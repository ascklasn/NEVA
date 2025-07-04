import pytorch_lightning as pl
import torch.cuda
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
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
    return read_yaml(f"../3_mil_classifier/configs/{fname}.yaml")


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


def setup_workspace(workspace):
    os.makedirs(workspace, exist_ok=True)


def save_results(results, fname): 
    results_dir = f"./inference_outputs"  # Directory to save results
    os.makedirs(results_dir, exist_ok=True)
    print('\n'*10)
    print("A new directory has been created, and the result of each seed contains 5 fold auc")
    results_file = os.path.join(results_dir, f"{fname}.json")
    with open(results_file, "a+") as f: 
        json.dump(results, f)  
        f.write('\n')


def inference(type: str, task: str = "alk"):
    """
    Perform inference on the specified task and type.
    
    Args:
        type (str): Type of task, either 'cls' for classification or 'reg' for regression.
        task (str): Specific task to perform, e.g., 'alk', 'cmyc', etc.
    """
    # Read configuration
    config_yaml_name = f"config_{type}_{task}_MultiModal"  # e.g. config_cls_alk_MultiModal
    config = read_config(config_yaml_name)  # Read the configuration file
    seed, fold = config['seed&fold']  # Extract seed and fold from the configuration
    fix_seed(seed)  # Set the random seed for reproducibility

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wts_path = f"./model_weights/{task}.pth"
    wts = torch.load(wts_path)
    
    model = MILModel(config, save_path=None).to(device)
    model.load_state_dict(wts, strict=True).to(device)  # Load the model weights

    df = pd.read_csv(f"./datasets/{task}.csv")  # Read the dataframe from the configuration
    case_ids = df['case_id'].tolist()  # Extract case IDs from the dataframe
    filenames = df['filenames'].tolist()  # Extract case IDs from the dataframe
    
    if type == 'cls':
        labels = torch.tensor(df[f'{task}'].tolist())  # Extract labels from the dataframe
    elif type == 'reg':
        labels = torch.tensor(df[f'{task}'].tolist())
        status = torch.tensor(df[f'status'].tolist())

    logits = []
    
    for case_id, filename in zip(case_ids, filenames):
        report_feature_path = f"../2_extract_features/report_large/{case_id}.pt"  # Path to the report feature file
        image_feature_path = f".../2_extract_features/images/{filename}.pt"  # Path to the image feature file
        assert os.path.exists(report_feature_path) and os.path.exists(image_feature_path), \
            f"Missing feature file(s): {', '.join([p for p in [report_feature_path, image_feature_path] if not os.path.exists(p)])}"
        
        image_feature = torch.load(image_feature_path).unsqueeze(0).to(device)
        report_feature = torch.load(report_feature_path).unsqueeze(0).to(device)  # Load the report feature
        with torch.inference_mode():
            logit, results_dict = model((image_feature, report_feature))
        logits.append(logit.cpu().numpy())  # Append the logit to the list

    logits = torch.cat(logits, dim=0)  # Concatenate all logits into a single tensor

    if type == 'cls':
        probs = torch.softmax(logits, dim=1)  # Apply softmax to get probabilities
        preds = torch.argmax(probs, dim=1)  # Get the predicted classes
        print(f"Predicted classes: {preds}\n Labels: {labels}")
        results = {
            f'{task}': labels,
            'preds': preds,
        }
        torch.save(results, f"./inference_outputs/{task}_results.pth")  # Save the results

    elif type == 'reg':
        results = {
            f'{task}': labels,
            'status': status,
            'hazard scores': logits,
        }
        torch.save(results, f"./inference_outputs/{task}_results.pth")  # Save the results

args = argparse.ArgumentParser(description='Inference for NEVA model using multimodal data')
args.add_argument('--type', type=str, choices=['cls', 'reg'], default='cls',
                  help='Type of task (e.g. classification, regression)')
args.add_argument('--task', type=str, choices=['hazard_level', 'subtype', 'mki', 'shimada', 'alk', 'nmyc', 'cmyc', 'p36', 'q23', 'os', 'pfs'],
                  help='Specific task to perform') 



if __name__ == "__main__":
    args = args.parse_args()
    print(f"Arguments: {args}")
    print(f"Type: {args.type}, Task: {args.task}")

    if args.type not in ['cls', 'reg']:
        raise ValueError("Invalid type. Choose either 'cls' or 'reg'.")
    
    if args.task not in ['hazard_level', 'subtype', 'mki', 'shimada', 'alk', 'nmyc', 'cmyc', 'p36', 'q23', 'os', 'pfs']:
        raise ValueError("Invalid task. Choose a valid task from the list.")
    
    # Perform inference
    inference(type=args.type, task=args.task)  # e.g. type='cls',

    print(f"Results saved to './inference_outputs/{args.task}_results.pth'")  # Indicate where the results are saved


    