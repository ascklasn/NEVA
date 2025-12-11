import os
import json
import yaml
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import GroupKFold
import argparse

import torch
import torch.cuda
import torch.backends.cudnn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy

from wsi_dataset import WSIDataModule
from models import MILModel, save_trainable_state_dict
import importlib
import copy

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_yaml(file_path: str) -> Dict[str, Any]:
    """Read YAML configuration file."""
    with open(file_path, mode="r", encoding='utf-8') as file:
        return yaml.load(file, Loader=yaml.Loader)


def read_config(config_name: str) -> Dict[str, Any]:
    """Read configuration from configs directory."""
    config_path = Path("./configs") / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return read_yaml(config_path)


def get_class_from_string(class_path: str, reload: bool = False) -> Any:
    """Dynamically import a class from a module string."""
    module_name, class_name = class_path.rsplit(".", 1)
    
    if reload:
        module = importlib.import_module(module_name)
        importlib.reload(module)
    
    return getattr(importlib.import_module(module_name), class_name)


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def create_folds(
    n_splits: int, 
    data_path: str, 
    output_dir: str, 
    seed: int, 
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, str]:
    """Create cross-validation folds with proper grouping."""
    # Read and prepare data
    df = pd.read_csv(data_path)
    
    # Create group key for stratification
    label_name = config.get('label_name', 'label')
    task_name = Path(config['dataframe']).stem
    if config.get('dataset_name') == "coxreg":
        df['group_key'] = df['patient_id'].astype(str) + '_' + df['status'].astype(str)
    else:
        df['group_key'] = df['patient_id'].astype(str) + '_' + df[label_name].astype(str)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Create folds using GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    
    # Initialize fold column
    df['fold'] = -1
    
    # Assign folds
    for fold, (_, test_idx) in enumerate(gkf.split(df, groups=df['group_key'])):
        df.loc[test_idx, 'fold'] = fold
    
    # Save fold information
    output_path = Path(output_dir) / f"{task_name}_{seed}_folds.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Created fold information at: {output_path}")
    logger.info(f"Fold distribution:\n{df['fold'].value_counts().sort_index()}")
    return df, str(output_path)


def setup_workspace(workspace_dir: str) -> None:
    """Create workspace directory if it doesn't exist."""
    Path(workspace_dir).mkdir(parents=True, exist_ok=True)


def prepare_trainer(
    config: Dict[str, Any], 
    workspace: str, 
    monitor_metric: str = "val_auc"
) -> pl.Trainer:
    """Prepare and configure the PyTorch Lightning Trainer."""
    # Determine mode based on metric
    mode = "max" if monitor_metric in ["val_auc", "val_cindex"] else "min"
    
    # Configure checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=workspace,
        filename=f"{{epoch}}-{{{monitor_metric}:.4f}}",
        save_top_k=1,
        monitor=monitor_metric,
        mode=mode,
        verbose=True,
        save_last=True,
    )
    
    # Configure early stopping
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        min_delta=0.00,
        patience=config["General"].get("patience", 10),
        mode=mode,
        verbose=True,
    )
    
    # Determine devices to use
    devices = config["General"].get("devices", 1)
    
    # Create and return trainer
    return pl.Trainer(
        accelerator="auto",
        devices=devices,
        # strategy="auto",
        strategy=DDPStrategy(find_unused_parameters=True),
        deterministic=True,
        precision=config["General"].get("precision", "16-mixed"),
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=config["General"]["epochs"],
        accumulate_grad_batches=config["General"]["acc_steps"],
        logger=False,
        enable_checkpointing=True,
        enable_model_summary=True,
        num_sanity_val_steps=0,  # Disable sanity check to avoid issues
        enable_progress_bar=True,
        log_every_n_steps=10,
    )


def save_results(results: Dict[str, Any], config_name: str) -> None:
    """Save training results to JSON file."""
    results_dir = Path("./workspace/results_MultiModal")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / f"{config_name}_results.json"
    
    # Load existing results if any
    existing_results = []
    if results_file.exists():
        try:
            with open(results_file, "r") as f:
                for line in f:
                    if line.strip():
                        existing_results.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning(f"Could not parse existing results file: {results_file}")
            existing_results = []
    
    # Append new results
    existing_results.append(results)
    
    # Save all results
    with open(results_file, "w") as f:
        for result in existing_results:
            json.dump(result, f)
            f.write("\n")
    
    logger.info(f"Results saved to: {results_file}")


def train_fold(
    config: Dict[str, Any],
    fold: int,
    seed: int,
    workspace: str,
    data_path: str,
    fold_df: pd.DataFrame
) -> float:
    """Train and evaluate a single fold."""
    # Set seed for reproducibility
    set_seed(seed)
    
    # Create a copy of the config with the updated data path
    config_copy = copy.deepcopy(config)
    config_copy['Data']['dataframe'] = data_path
    
    # Create data module
    data_module = WSIDataModule(
        config_copy,
        split_k=fold,
        dist=False,
        data_df=fold_df.copy(deep=True),
    )

    # Create model
    model_save_path = Path("./workspace/models") / config['name'] / f"seed={seed}_fold={fold}"
    setup_workspace(model_save_path)
    
    model = MILModel(config_copy, save_path=str(model_save_path))
    
    # Create trainer
    trainer = prepare_trainer(config_copy, workspace, config_copy['General']['metric'])
    
    # Train the model
    trainer.fit(model, datamodule=data_module)
    
    # Get best model path
    best_model_path = trainer.checkpoint_callback.best_model_path
    
    if not best_model_path or not Path(best_model_path).exists():
        logger.warning(f"No valid checkpoint found for fold {fold}")
        return 0.0
    
    logger.info(f"Best model for fold {fold}: {best_model_path}")
    
    # Test the model
    trainer.test(model, datamodule=data_module, ckpt_path="best")

    # Save only the model state dict
    model_state_path = model_save_path / f"fold_{fold}.pth"
    checkpoint = torch.load(best_model_path, map_location="cpu")
    save_trainable_state_dict(model, checkpoint['state_dict'], model_state_path)
 
    return getattr(model, 'test_performance', 0.0)


def main(config_name):
    """Main training function."""

    # Read configuration
    config = read_config(config_name)
    config['name'] = Path(config['Data']['dataframe']).stem
    
    # Display configuration
    logger.info(f"Configuration: {config_name}")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for k, v in value.items():
                logger.info(f"  {k}: {v}")
        else:
            logger.info(f"{key}: {value}")
    
    # Training parameters
    n_splits = 5  # 5-fold cross-validation
    seeds = config.get("seeds", [1234])
    data_path = config['Data']['dataframe']
    project_name = Path(data_path).stem
    
    # Get selected seed and fold if specified
    selected_seed, selected_fold = config.get('seed&fold', (None, None))
    
    # Process each seed
    for seed in seeds:
        # Skip if not the selected seed
        if selected_seed is not None and seed != selected_seed:
            continue
            
        logger.info(f"Starting training with seed: {seed}")
        
        # Create workspace directory
        workspace_dir = f"./outputs_{config_name}"
        setup_workspace(workspace_dir)
        
        # Create fold information (reuse existing splits if provided)
        df_fold = pd.read_csv(data_path)
        if 'fold' in df_fold.columns:
            logger.info("Detected precomputed folds in input dataframe; skipping fold generation.")
            fold_df = df_fold
            fold_data_path = data_path
        else:
            fold_df, fold_data_path = create_folds(
                n_splits, data_path, workspace_dir, seed, config['Data']
            )
        
        # Store results for this seed
        seed_results = []
        
        # Process each fold
        for fold in range(n_splits):
            # Skip if not the selected fold
            if selected_fold is not None and fold != selected_fold:
                continue
                
            logger.info(f"Training fold {fold} with seed {seed}")
        
            # Train and evaluate the fold
            performance = train_fold(
                config,
                fold,
                seed,
                workspace_dir,
                fold_data_path,
                fold_df,
            )
            seed_results.append(performance)
            
            logger.info(f"Fold {fold} completed with performance: {performance:.4f}")
            
    
        # Calculate and save results for this seed
        if seed_results:
            metric_name = 'cindex' if config['General']['metric'] == 'val_cindex' else 'auc'
            macro_avg = np.mean(seed_results)
            macro_std = np.std(seed_results)
            
            results = {
                project_name: {
                    "seed": seed,
                    f"macro_{metric_name}": round(macro_avg, 4),
                    "macro_std": round(macro_std, 4),
                    "folds": [round(r, 4) for r in seed_results],
                }
            }
            
            save_results(results, config_name)
            logger.info(f"Seed {seed} completed. Average {metric_name}: {macro_avg:.4f} ± {macro_std:.4f}")
        
        # # Clean up temporary files
        # try:
        #     os.remove(fold_data_path)
        #     logger.info(f"Removed temporary file: {fold_data_path}")
        # except OSError:
        #     pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV configurations.')
    parser.add_argument('--csv_list', nargs='+', 
    default=['p36', 'q23', 'alk', 'cmyc', 'hazard_level', 'mki', 'nmyc', 'shimada', 'subtype', 'os', 'pfs'], help='List of CSV names to process')
    args = parser.parse_args()
    
    for fname in args.csv_list:
        config_type = f'config_reg_{fname}' if fname in ['pfs', 'os'] else f'config_cls_{fname}'
        main(config_type)
