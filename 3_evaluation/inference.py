import pytorch_lightning as pl
import torch.cuda
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from wsi_dataset import WSIDataModule, get_dataset_fn
import argparse
import yaml
import importlib
from models import MILModel, compute_c_index
import os
import random
import numpy as np
import torchmetrics.functional as tf
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
from pytorch_lightning.strategies import DDPStrategy
from typing import Dict, List, Tuple, Optional, Any
from pprint import pprint as pp
import h5py
import multiprocessing
import torchvision.transforms.v2 as T
# multiprocessing.set_start_method('spawn', force=True)
# torch.cuda.empty_cache()
from metric import bootstrap_auc, bootstrap_cindex, compute_P_value
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision.io import read_image 
import logging
import torch.nn as nn
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def read_yaml(fpath): 
    with open(fpath, mode="r",encoding='utf-8') as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return dict(yml)

def read_config(config_path):
    return read_yaml(config_path)


def get_obj_from_str(string, reload=False):     
    module, cls = string.rsplit(".", 1) 
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

# seed everything
def fix_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)  # torch >= 1.8


def setup_workspace(workspace):
    os.makedirs(workspace, exist_ok=True)


def save_results(results_dict, dataset_name, results_dir="./eval_results"):
    # results_dir = os.path.join(results_dir, dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    print('\n'*2)
    results_file = os.path.join(results_dir, f"{dataset_name}.json")
    with open(results_file, "a+") as f: 
        json.dump(results_dict, f)  
        f.write('\n')

def create_case_dicts(dataframe,type: str = 'cls', dataset_name: str = 'subtype') -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """

    Args:
        dataframe: case_id、filenames、labels 
    
    Returns:
        tuple: (case_to_filenames_dict, case_to_label_dict)
    """

    case_to_filenames = {}

    case_to_label = {}
    
    for case_id, group in dataframe.groupby('case_id'):
        case_id_str = str(case_id)
        
        filenames_list = group['filename'].tolist()
        case_to_filenames[case_id_str] = filenames_list
        
        if type == 'cls':
            label = group[dataset_name].iloc[0] 
        elif type == 'coxreg':
            time = group[dataset_name].iloc[0]  
            status = group['status'].iloc[0] 
            label = (time, status)

        case_to_label[case_id_str] = label
    
    return case_to_filenames, case_to_label


class NEVA_InputLoader:
    def __init__(self, config, case_id,filenames):
        
        self.image_dir = config.get('image_dir',None)   
        self.report_dir = config.get('report_dir', None)  
        self.FM_type = config['FM_type']  # Patch_Multimodal
        self.FM_name = config['FM_name']  # neva
        self.dataset_type = config['dataset_type']
        self.dataset_name = config['dataset_name']
        self.cohort_name = config['cohort_name']  # Internal

        self.MAX_BAG = 200

        self.case_id = case_id
        if self.image_dir is not None:
            if self.cohort_name == 'Internal':
                high_mpp_dir = self.image_dir
                low_mpp_dir = self.image_dir.replace("high", "low")

    
                self.image_list = [os.path.join(high_mpp_dir, filename.replace('.h5','')) for filename in filenames if os.path.exists(os.path.join(high_mpp_dir, filename.replace('.h5','')))] + \
                                [os.path.join(low_mpp_dir, filename.replace('.h5','')) for filename in filenames if os.path.exists(os.path.join(low_mpp_dir, filename.replace('.h5','')))]
            else:
                self.image_list = [os.path.join(self.image_dir, filename.replace('.h5','')) for filename in filenames if os.path.exists(os.path.join(self.image_dir, filename.replace('.h5','')))]


            assert self.image_list, f"No image directories found for case_id {case_id} in cohort {self.cohort_name}."
            for image_path in self.image_list:
                assert os.path.exists(image_path), f"TOP200 Image dirs don't exist: {image_path}"
            self.image_list = sorted(self.image_list)  

        self.img_size = 224
        self.transform = T.Compose([
            T.Resize(self.img_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.CenterCrop((self.img_size, self.img_size)),


            T.ConvertImageDtype(torch.float32),

            T.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ])

    def _load_single_image_feature(self, image_dir: str) -> torch.Tensor:

        assert os.path.isdir(image_dir), f"{image_dir} is not a valid directory."
        patch_list = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        num_patches = len(patch_list)

        # Validate the number of patches
        if num_patches < 2:
            raise ValueError(f"Expected at least 2 patches, but found {num_patches} in {image_dir}")
        if num_patches < self.MAX_BAG:
            # Warn if padding is required but within acceptable range
            print(f"Warning: Only {num_patches} patches found in {image_dir}. Padding to {self.MAX_BAG}.")
        
        # Load available patches
        image_tensors = []
        for patch_path in patch_list[:self.MAX_BAG]:  # Load up to MAX_BAG patches
            image_tensors.append(read_image(patch_path))
        
        # Pad with blank images if necessary
        if num_patches < self.MAX_BAG:
            blank_image = torch.zeros_like(image_tensors[0])
            for _ in range(self.MAX_BAG - num_patches):
                image_tensors.append(blank_image)


        patch_bag_tensor = torch.stack(image_tensors, dim=0)  # [200, 3, H, W]
        patch_bag_tensor = self.transform(patch_bag_tensor)  # [200, 3, 224, 224]

        assert patch_bag_tensor.shape[0] == self.MAX_BAG, f"Expected {self.MAX_BAG} patches, got {patch_bag_tensor.shape[0]}"
        assert patch_bag_tensor.shape[1] == 3, f"Expected 3 channels, got {patch_bag_tensor.shape[1]}"
        expected_size = (self.img_size, self.img_size)
        assert patch_bag_tensor.shape[2:] == expected_size, f"Expected size {expected_size}, got {patch_bag_tensor.shape[2:]}"
        
        return patch_bag_tensor  # [200, 3, self.img_size, self.img_size] 比如[200, 3, 224, 224]

    @property  
    def load_full_image_feature(self) -> torch.Tensor:   

        image_feat = []
        filename_list = []
        for image_dir in self.image_list:
            filename = os.path.basename(image_dir) # 不包含.h5
            filename_list.append(filename)
            image_feat.append(self._load_single_image_feature(image_dir))
        stacked_images = torch.stack(image_feat, dim=0)  # [N, 200, 3, 768, 768]
        return stacked_images,filename_list  # [N, 200, 3, 768, 768]，N可以是1。 

    @property
    def load_report_feature(self) -> torch.Tensor:
        """load text feature"""
        assert self.report_dir is not None, "Report directory is not specified in the config."
        # load tokenzier for language input
        from transformers import XLMRobertaTokenizer
        from musk import utils, modeling
        report_csv = pd.read_csv(self.report_dir)
        case_id_list = report_csv['case_id'].tolist()
        report_en_list = report_csv['report_en'].tolist()  

        # load tokenzier for language input
        tokenizer = XLMRobertaTokenizer("./musk/models/tokenizer.spm")

        report_dict = {}
        import tqdm
        for case_id,report in tqdm(zip(case_id_list,report_en_list)):
            txt_ids, pad = utils.xlm_tokenizer(report, tokenizer, max_len=500)
            report_dict[str(case_id)] = {
                'text_ids': torch.tensor(txt_ids),
                'text_pads': torch.tensor(pad)
            }

        report_ids = report_dict[str(case_id)]['text_ids']  # shape: （500,）
        report_pads = report_dict[str(case_id)]['text_pads']

        return (report_ids, report_pads) 



def main_evaluate(FM_config=None):


    seed = FM_config['seed']
    fix_seed(seed)

    device= f"cuda:{FM_config['gpu']}" if torch.cuda.is_available() else "cpu"
    config_yaml = FM_config
    fold = FM_config['fold']
    pp(config_yaml)

    FM_name = config_yaml['Model'].get('FM_name')

    dataset_name = config_yaml['Data']['dataset_name']  # nmyc,cmyc,1p36
    task = dataset_name
    cohort_name = config_yaml['Data']['cohort_name']  # Prospective, Internal, PUFH, Shenzhen, GCI
    print(f"Cohort: {cohort_name} FM：{FM_name}，Dataset：{dataset_name}")  

    eval_df = config_yaml['Data'].get('eval_df', None)
    assert eval_df is not None and os.path.exists(eval_df), f"not found {eval_df}"
    if cohort_name == 'Internal':
        dataframe = pd.read_csv(eval_df)
        dataframe = dataframe[dataframe['fold'] == fold]
    else: 
        dataframe = pd.read_csv(eval_df)


    case_to_filenames, case_to_label = create_case_dicts(dataframe,type=config_yaml['Data']['dataset_type'],dataset_name=dataset_name)

    assert list(case_to_filenames.keys()) == list(case_to_label.keys()), "case_to_filenames and case_to_label do not match"

    model = MILModel(config_yaml, save_path='./').cpu().to(device) 
    wts_path = config_yaml['Model'].get('wts_path')

    if 'neva' in config_yaml['Data']['FM_name']:
        raw_state = torch.load(wts_path, map_location=device)
        state_dict = raw_state.get("state_dict", raw_state)
        incompatible_keys = model.load_state_dict(state_dict, strict=False)
        if incompatible_keys.missing_keys:
            logger.debug(
                "Fold %s missing %d keys when loading %s (expected for LoRA-only checkpoints). Example: %s",
                fold,
                len(incompatible_keys.missing_keys),
                wts_path,
                incompatible_keys.missing_keys[:5],
            )
        if incompatible_keys.unexpected_keys:

            logger.warning(
                "Unexpected parameters in checkpoint %s: %s",
                wts_path,
                incompatible_keys.unexpected_keys,
            )
    else:
        wts = torch.load(wts_path, map_location=device, weights_only=True)
        model.load_state_dict(wts)

    all_logits = []
    all_probs = []
    all_labels = []
    all_status = []  
    with torch.inference_mode():
        model.eval()
        N_all = len(case_to_filenames)
        count = 0
        print(f"Total {N_all} cases for evaluation.")
        for case_id in case_to_filenames.keys():
            filenames = case_to_filenames[case_id]  # xxx.h5
            if config_yaml['Data']['FM_name'] == 'neva':
                input = NEVA_InputLoader(FM_config['Data'], case_id, filenames)
                images_features, filename_list = input.load_full_image_feature
                images_features = images_features.to(device)
                report_ids, report_pads = input.load_report_feature  # (500,), (500,) 

                report_ids = report_ids[:100]
                report_pads = report_pads[:100]
                report_ids = report_ids.unsqueeze(0).expand(images_features.shape[0], -1)   # [N, 500]
                report_pads = report_pads.unsqueeze(0).expand(images_features.shape[0], -1) # [N, 500]
                report_features = (report_ids, report_pads)
                N = images_features.shape[0]
                logit = []
                for i in range(N):
                    image_feature = images_features[i].unsqueeze(0).to(device)  # [1, 50, 3, image_size, image_size]
                    report_feature = (report_ids[i].unsqueeze(0).to(device), report_pads[i].unsqueeze(0).to(device))  # ([1, 500], [1, 500])
                    logit_, _ = model((image_feature, report_feature))  # [1, n_classes]
                    logit.append(logit_)
                    del image_feature, report_feature, logit_
                logits = torch.cat(logit, dim=0)  # [N, n_classes]
                del report_ids, report_pads
            

            if config_yaml['Data']['dataset_type'] == 'cls':

                if FM_config['Data']['dataset_name'] == 'risk_group':
                    # The "risk group" is a binary classification task.
                    label = torch.tensor(case_to_label[case_id], dtype=torch.long, device=device)
                    label = torch.where(label != 0, torch.tensor(1, device=label.device), label)

                else:  # subtype, nmyc, cmyc
                    label = torch.tensor(case_to_label[case_id], dtype=torch.long, device=device)

                all_labels.append(label.cpu().tolist())
                logit = logits.mean(dim=0, keepdim=False)  # [1,]
                prob = torch.softmax(logit, dim=-1)
                all_probs.append(prob.cpu().tolist())
                all_logits.append(logit.cpu().tolist())

            elif config_yaml['Data']['dataset_type'] == 'coxreg':
                logit = logits.mean(dim=0, keepdim=False)  # [1,]
                all_logits.append(logit.cpu().tolist())  
                time, status = case_to_label[case_id]
                all_labels.append(time)  
                all_status.append(status)  

    if config_yaml['Data']['dataset_type'] == 'cls':
        metric_name = 'auroc'
        auc_mean, ci_lower, ci_upper, _, _ = bootstrap_auc(all_labels, all_probs, n_bootstrap=1000)
        save_dir = f'./eval_results/{cohort_name}/{task}/seed_{seed}/fold_{fold}/{FM_name}'
        os.makedirs(save_dir, exist_ok=True)
        torch.save(all_labels, os.path.join(save_dir, 'all_labels.pt'))
        torch.save(all_logits, os.path.join(save_dir, 'all_logits.pt'))

    elif config_yaml['Data']['dataset_type'] == 'coxreg':
        metric_name = 'cindex'
        cindex_mean, ci_lower, ci_upper, _, _ = bootstrap_cindex(all_logits, all_labels, all_status, n_bootstrap=1000)
        P_value = compute_P_value(all_logits, all_labels, all_status)
        save_dir = f'./eval_results/{cohort_name}/{task}/seed_{seed}/fold_{fold}/{FM_name}'
        os.makedirs(save_dir, exist_ok=True)
        torch.save(all_labels, os.path.join(save_dir, 'all_labels.pt'))
        torch.save(all_logits, os.path.join(save_dir, 'all_logits.pt'))
        torch.save(all_status, os.path.join(save_dir, 'all_status.pt'))

    metric_str = f"{auc_mean:.4f}({ci_lower:.4f}-{ci_upper:.4f})" if config_yaml['Data']['dataset_type'] == 'cls' else f"{cindex_mean:.4f}({ci_lower:.4f}-{ci_upper:.4f}), P-value: {P_value:.4e}"
    results = {
        dataset_name: {
            'cohort': cohort_name,
            'FM_name': FM_name,
            "seed": seed,
            'fold': fold,
            metric_name: metric_str, 
        }
    }
    save_results(results, dataset_name,results_dir="./eval_results")  


if __name__ == "__main__":

    model_dict = {
        'neva': {'FM_type': 'Patch_Multimodal',
                  'feat_dim': 1024,
                  },
    }

    parser = argparse.ArgumentParser(description='Mode: Evaluation. Process CSV configurations.')
    parser.add_argument('--csv_list', nargs='+',  
                        default=['1p36', '11q23', 'alk', 'cmyc', 
                        'risk_group', 'mki', 'nmyc', 'shimada', 
                        'subtype', 'os', 'pfs'], help='List of CSV names to process')
    parser.add_argument('--dataset_type', type=str, default='cls', choices=['cls', 'coxreg'],)
    parser.add_argument('-FM', '--FoundationModel', type=str, nargs='+', default=['neva'],
                                                                       help='Foundation Models to evaluate')
    parser.add_argument('--cohort_name', type=str, default='Internal', choices=['Internal', 'Prospective_1', 'Prospective_2', 'PUFH', 'Shenzhen', 'GCI'], help='Evaluation datasets to use')      
    parser.add_argument('--image_dir', type=str, default=None, help='Path to images  directory')        
    parser.add_argument('--eval_df', type=str, default=None, help='Path to eval csv file')      
    parser.add_argument('--cohort_name', type=str, default='Internal', choices=['Internal', 'Prospective_1', 'Prospective_2', 'PUFH', 'Shenzhen', 'GCI'], help='Evaluation datasets to use')      
    parser.add_argument('--gpu', type=int, default=1, choices=[0, 1], help='GPU id to use')

    args = parser.parse_args()

    for FM_name in args.FoundationModel:
        
        FM_config = read_yaml(f"./configs/NEVA.yaml"
        
        FM_config['Data']['dataset_type'] = args.dataset_type    # cls or coxreg
        FM_config['Data']['cohort_name'] = args.cohort_name      # Internal, Prospective_1, Prospective_2
        FM_config['Data']['FM_name'] = FM_name                   # neva
        FM_config['Data']['FM_type'] = model_dict[FM_name]['FM_type']


        FM_config['Data']['image_dir'] = args.image_dir 

        FM_config['Data']['report_dir'] = args.eval_df

        if 'Multimodal' in model_dict[FM_name]['FM_type']:
            assert FM_config['Data']['image_dir'] is not None and FM_config['Data']['report_dir'] is not None, "Multimodal need both image_dir and report_dir"

        if 'neva' in FM_name:
            FM_config['Model']['name'] = 'models.neva.NEVA'

        FM_config['Model']['FM_name'] = FM_name
        FM_config['Model']['FM_type'] = model_dict[FM_name]['FM_type']
        
        FM_config['Model']['params']['feat_dim'] = model_dict[FM_name].get('feat_dim', None)
        FM_config['Model']['params']['report_feat_dim'] = model_dict[FM_name].get('report_dim', None)
        FM_config['Model']['params']['align_dim'] = model_dict[FM_name].get('align_dim', None)
        FM_config['Model']['params']['FM_type'] = model_dict[FM_name]['FM_type']
        FM_config['Model']['params']['FM_name'] = FM_name
        FM_config['Model']['params']['FM_type'] = model_dict[FM_name]['FM_type']


        FM_config['General']['metric'] = 'val_auc' if args.dataset_type == 'cls' else 'val_cindex'

        FM_config['gpu'] = args.gpu

        for task in args.csv_list:
            FM_config['Data']['dataset_name'] = task
            match task:
                case 'risk_group':
                    FM_config['seed'] = 2345
                    FM_config['fold'] = 2
                    FM_config['Model']['params']['num_classes'] = 2
                case 'subtype':
                    FM_config['seed'] = 42
                    FM_config['fold'] = 2
                    FM_config['Model']['params']['num_classes'] = 3
                case 'shimada':
                    FM_config['seed'] = 3456
                    FM_config['fold'] = 0
                    FM_config['Model']['params']['num_classes'] = 2
                case 'mki':

                    FM_config['seed'] = 42
                    FM_config['fold'] = 4
                    FM_config['Model']['params']['num_classes'] = 3
                case 'alk':
                    FM_config['seed'] = 42
                    FM_config['fold'] = 2
                    FM_config['Model']['params']['num_classes'] = 2
                case 'cmyc':
                    FM_config['seed'] = 42
                    FM_config['fold'] = 3
                    FM_config['Model']['params']['num_classes'] = 2
                case 'nmyc':
                    FM_config['seed'] = 2345
                    FM_config['fold'] = 4
                    FM_config['Model']['params']['num_classes'] = 2
                case '1p36':
                    FM_config['seed'] = 42
                    FM_config['fold'] = 3
                    FM_config['Model']['params']['num_classes'] = 2
                case '11q23':
                    FM_config['seed'] = 42
                    FM_config['fold'] = 4
                    FM_config['Model']['params']['num_classes'] = 2
                case 'pfs':
                    FM_config['seed'] = 2345
                    FM_config['fold'] = 2
                    FM_config['Model']['params']['num_classes'] = 1
                    FM_config['Loss']['name'] = "models.model_utils.CoxSurvLoss"
                case 'os':
                    FM_config['seed'] = 2345
                    FM_config['fold'] = 0
                    FM_config['Model']['params']['num_classes'] = 1
                    FM_config['Loss']['name'] = "models.model_utils.CoxSurvLoss"
                
            FM_config['Data']['eval_df'] = args.eval_df 
            assert FM_config['Data']['eval_df'] is not None, "Mode:Evaluate, eval_df should not be None"

            FM_config['Model']['wts_path'] = os.path.join(f"./neva_weights/{task}/seed={FM_config['seed']}_fold={FM_config['fold']}/fold_{FM_config['fold']}.pth")

            assert os.path.exists(FM_config['Model']['wts_path']), f"Weigts not found：{FM_config['Model']['wts_path']}"
            main_evaluate(FM_config)