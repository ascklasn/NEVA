import pytorch_lightning as pl
import torch.cuda
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from wsi_dataset import WSIDataModule
import yaml
import importlib
from models import MILModel
import os
import random
import numpy as np
import torchmetrics.functional as tf
from models import compute_c_index   
import glob
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.strategies import DeepSpeedStrategy
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
from lightning.pytorch.accelerators import find_usable_cuda_devices
import oven
import argparse

def read_yaml(fpath):   # Read config configuration
    with open(fpath, mode="r",encoding='utf-8') as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return dict(yml)

def read_config(fname):
    return read_yaml(f"./configs/{fname}.yaml")


def get_obj_from_str(string, reload=False):      # Dynamically obtain the required model class
    module, cls = string.rsplit(".", 1)   
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

# seed everything
def fix_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)   # Set Python's hash seed to ensure reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.set_deterministic(True)  # torch < 1.8
    torch.use_deterministic_algorithms(True, warn_only=True)  # torch >= 1.8


def create_fold(N_SPLITS, fname, temp_dir, seed, config):  
    # Read data
    df = pd.read_csv(fname)  

    df.drop_duplicates(subset=['case_id'], inplace=True)  # Delete samples with the same case_id, and only the first sample that appears is preserved.
    # Create a composite key 'group_key' for grouping and stratification; ensure the same patient is in the same group.
    if config['dataset_name'] == "coxreg":
        df['group_key'] = df['patient_id'].astype(str) + '_' + df['status'].astype(str)
    else:
        df['group_key'] = df['patient_id'].astype(str) + '_' + df[config['label_name']].astype(str)

    # Shuffle the DataFrame if necessary
    np.random.seed(seed)  # Set seed for reproducibility
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame and reset the index, dropping the original index

    # Initialize GroupKFold
    gkf = GroupKFold(n_splits=N_SPLITS)  # Stratified cross-validation with 5 folds

    # Split data ensuring the same patient and similar status stays in the same fold
    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df['group_key'])):
        df.loc[test_idx.astype(int), 'fold'] = fold  # Set the fold for the test dataset based on its index

    # Reset index and save the dataframe with fold information

    print(f"{temp_dir}/temp.csv")
    print('##############################'*10)
    df.to_csv(f"{temp_dir}/{seed}_temp.csv",index=False)  # Save the dataset path
    print('Create new data CSV successfully')



def setup_workspace(workspace):
    os.makedirs(workspace, exist_ok=True)

ckpt_save_path=''
def prepare_trainer(config_yaml, num_gpus, workspace, monitor="val_auc"):

    # save_fname = "{epoch}-{val_auc:.3f}"
    checkpoint_cb = ModelCheckpoint( 
        dirpath=workspace,  
        filename="{epoch}-{val_auc:.4f}",  
        save_top_k=1,
        monitor=monitor,    # cls: val_auc    cox: val_cindex
        mode="max",
        verbose=True,
    )
    
    early_stop_cb = EarlyStopping(
        monitor=monitor,    # cls: val_auc    cox: val_cindex
        min_delta=0.00, 
        patience=10,
        mode="max"
    )

    return pl.Trainer(
        accelerator="cuda",
        devices=[0],   # your device id
        # strategy='ddp_find_unused_parameters_true',
        # strategy='ddp',
        deterministic=False,
        precision='16-mixed',

        callbacks=[checkpoint_cb, early_stop_cb],
        max_epochs=config_yaml["General"]["epochs"],
        accumulate_grad_batches=config_yaml["General"]["acc_steps"], 
        logger=False,
        inference_mode=True,  
    )

def save_results(results, fname):
    results_dir = f"./workspace/results_MultiModal"  # Directory to save results
    os.makedirs(results_dir, exist_ok=True)
    print('\n'*10)
    print("A new directory has been created, and the result of each seed contains 5 fold auc")
    results_file = os.path.join(results_dir, f"{fname}.json")
    with open(results_file, "a+") as f: 
        json.dump(results, f)  
        f.write('\n')

def main(type: str, task: str="alk"):
    
    MultiModal_list=['config_cls_alk_MultiModal','config_cls_cmyc_MultiModal',
                    'config_cls_hazard_level_MultiModal','config_cls_mki_MultiModal',
                    'config_cls_nmyc_MultiModal','config_cls_shimada_MultiModal',
                    'config_cls_subtype_MultiModal',
                    'config_cls_p36_MultiModal','config_cls_q23_MultiModal'
                    'config_reg_os_MultiModal','config_reg_pfs_MultiModal',]
    

    fname = f"config_{type}_{task}_MultiModal"  # e.g. config_cls_alk_MultiModal

    config_yaml = read_config(fname)  # read the configuration file
    
    seed,fold = config_yaml['seed&fold']

    for key, value in config_yaml.items():
        print(f"{key.ljust(30)}: {value}")

    num_gpus = 1
    dist = False # default : False

    original_csv = config_yaml['Data']['dataframe']   
    proj_name = config_yaml['Data']['label_name']   
    

    fix_seed(seed) # seed everything
    workspace = f"./outputs_{fname}"  
    rets_fold = []  # save the performance of each fold   

    setup_workspace(workspace) 


    dm = WSIDataModule(config_yaml, split_k=fold, dist=dist)  # datamodule Preparation dataset: training set, validation set, test set
    save_path = f"./workspace/models/{fname}/"    # Used to store logits, labels, and status files,
    setup_workspace(save_path) 


    model = MILModel(config_yaml, save_path=save_path)  #  pytorch_lighting 
    trainer = prepare_trainer(config_yaml, num_gpus, workspace, config_yaml['General']['metric'])

    trainer.fit(model, datamodule=dm, ckpt_path=None)  # trainer.fit: Training and Validation
    
    wts = trainer.checkpoint_callback.best_model_path  
    print('The best checkpoint path is：\n',wts)
    if os.path.exists(wts) == False:
        print('The best checkpoint does not exist')
    else:
        print('The best checkpoint exists')

    trainer.test(datamodule=dm, ckpt_path='best')    # trainer.test: Testing the model on the test set
    rets_fold.append(model.test_performance)   

    torch.save(torch.load(wts)['state_dict'], f'{save_path}/{proj_name}.pth')  # ! Used to save the model parameters
    os.system(f"rm -rf {workspace}")  

    macro_avg = np.mean(rets_fold)
    macro_std = np.std(rets_fold)
    metric_name = 'cindex' if config_yaml['General']['metric'] == 'val_cindex' else 'auc'
    results = {
        proj_name: {
            "seed": seed,
            f"macro_{metric_name}": round(macro_avg, 4),   
            "macro_std": round(macro_std, 4),  
            "folds": rets_fold,  
        }
    }
    save_results(results, fname) 


parser = argparse.ArgumentParser(description='Train NEVA model using multimodal data')
parser.add_argument('--type', type=str, choices=['cls', 'reg'], default='cls',
                    help='type of task (e.g. classification, Cox Regression)')
parser.add_argument('--task', type=str, choices=[ 'hazard_level', 'subtype','mki', 'shimada', 'alk', 'nmyc', 'cmyc', 'p36', 'q23', 'os', 'pfs'],
                    help='specific task to perform')

if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Arguments: {args}")
    print(f"Type: {args.type}, Task: {args.task}")
    if args.type not in ['cls', 'reg']:
        raise ValueError("Invalid type. Choose either 'cls' or 'reg'.")
    
    main(type=args.type, task=args.task)  # e.g. type='cls', task='alk'

