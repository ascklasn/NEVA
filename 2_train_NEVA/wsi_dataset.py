import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data.distributed import DistributedSampler
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import pandas as pd
from copy import deepcopy
import glob
import pandas as pd
from torch.utils.data import Sampler
import numpy as np
import h5py
import math
import numpy as np
import torchvision
from transformers import XLMRobertaTokenizer
from musk import utils, modeling
from PIL import Image
from torchvision.io import read_image  # 直接将png图像读成Tensor[C,H,W]，通常比PIL快
import torchvision.transforms.v2 as T
import pandas as pd
from musk import utils as mutils

class CoxRegDataset(Dataset):  # cox-regression  预后回归任务
    def __init__(self, df, config):
        """
        Initialize the CoxRegDataset.

        Parameters:
        - df (DataFrame): DataFrame containing the dataset.
        - config (dict): Configuration dictionary with 'image_dir', 'report_dir', and 'wsi_batch' keys.
        """
        super(CoxRegDataset, self).__init__()

        self.df = df
        self.config = config
        self.image_dir = config.get('image_dir')
        self.report_dir = config.get('report_dir')
        self.wsi_batch = config.get('wsi_batch', False)  

        # 选择 TOP Patches
        self.MAX_BAG = 100  # Maximum bag size for WSI batch 
        self.image_dir = config.get('image_dir', None)  # 存放TOP_50个patch的文件夹的路径
        self.report_dir = config.get('report_dir', None)  # 存放英文报告的csv文件的路径

        # 定义图像的转换方式, mpp=1.0, pacth_size=224
        self.img_size = 224
        self.transform = T.Compose([
            # 调整大小 & 裁剪
            T.Resize(self.img_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.CenterCrop((self.img_size, self.img_size)),

            # 把 uint8 (0–255) 转为 float32 (0–1)
            T.ConvertImageDtype(torch.float32),

            # 按照 ImageNet 规范化
            T.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ])

        self.image_dict = self._load_image_path_dict() if self.image_dir is not None else None
        self.report_dict = pd.read_csv(self.report_dir) if self.report_dir is not None else None
        self.tokenizer = XLMRobertaTokenizer("./musk/models/tokenizer.spm")

    def _load_image_path_dict(self):
        """
        从父文件夹中读取所有子文件夹的名称和路径
        """
        parent_dir = self.image_dir
        # low_mpp_dir = self.image_dir.replace("high", "low")
        path_dict = {}
        # for parent_dir in [high_mpp_dir, low_mpp_dir]:
        #     image_dir_list =  [os.path.join(parent_dir, name) for name in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, name))]
        #     path_dict.update({os.path.basename(x): x for x in image_dir_list})
        image_dir_list =  [os.path.join(parent_dir, name) for name in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, name))]
        path_dict.update({os.path.basename(x): x for x in image_dir_list})
        return path_dict

    def _load_image(self, image_filename: str):
        """
        Load the top patches for the given image filename, apply transformations, and ensure exactly
        self.MAX_BAG patches. If there are between 20 and 49 patches, pad with blank images. If fewer
        than 20, raise an error.
        """
        # Retrieve the directory containing top patches
        image_dir = self.image_dict.get(image_filename)
        assert image_dir is not None, f"no patches for {image_filename}."
        assert os.path.isdir(image_dir), f"{image_dir} is not a valid directory."
        
        # Gather all patch files and sort them to maintain order
        all_patches = sorted(glob.glob(os.path.join(image_dir, f"{image_filename}_*.png")))
        num_patches = len(all_patches)
        
        # Validate the number of patches
        if num_patches < 2:
            raise ValueError(f"Expected at least 2 patches, but found {num_patches} in {image_dir}")
        if num_patches < self.MAX_BAG:
            # Warn if padding is required but within acceptable range
            print(f"Warning: Only {num_patches} patches found in {image_dir}. Padding to {self.MAX_BAG}.")
        
        # Load available patches
        image_tensors = []
        for patch_path in all_patches[:self.MAX_BAG]:  # Load up to MAX_BAG patches
            image_tensors.append(read_image(patch_path))
        
        # Pad with blank images if necessary
        if num_patches < self.MAX_BAG:
            blank_image = torch.zeros_like(image_tensors[0])
            for _ in range(self.MAX_BAG - num_patches):
                image_tensors.append(blank_image)
        
        # Apply transformations and validate the output tensor
        images_bag_tensor = torch.stack(image_tensors, dim=0)
        images_bag_tensor = self.transform(images_bag_tensor)
        
        assert images_bag_tensor.shape[0] == self.MAX_BAG, f"Expected {self.MAX_BAG} patches, got {images_bag_tensor.shape[0]}"
        assert images_bag_tensor.shape[1] == 3, f"Expected 3 channels, got {images_bag_tensor.shape[1]}"
        expected_size = (self.img_size, self.img_size)
        assert images_bag_tensor.shape[2:] == expected_size, f"Expected size {expected_size}, got {images_bag_tensor.shape[2:]}"
        
        return images_bag_tensor
        
    def _load_report(self, case_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        report = self.report_dict.loc[self.report_dict['case_id'] == case_id, 'report_en'].iloc[0]
        text_ids, text_pads = mutils.xlm_tokenizer(report, self.tokenizer, max_len=100)
        text_ids_tensor = torch.as_tensor(text_ids, dtype=torch.long)
        text_pads_tensor = torch.as_tensor(text_pads, dtype=torch.bool)
        return text_ids_tensor, text_pads_tensor

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        self.df['case_id']
        return len(self.df['case_id'])


    def __getitem__(self, idx):

        # ! 根据 filename 获得的图像，根据case_id获得英文报告
        """
        Get a single sample from the dataset.

        Parameters:
        - idx (int): Index of the sample.

        Returns:
        - tuple: Tuple containing image tensors, report ids/pads tensors, time, and status.
        """
        item = self.df.iloc[idx]

        case_id = str(item['case_id'])   #  case_id同时也是report的名字
        image_filename = str(item["filename"].replace(".pt",""))
        # print(f"*********   {image_filename}   *********")

        image = self._load_image(image_filename) if self.image_dir is not None else torch.zeros(100)
        report = self._load_report(case_id) if self.report_dir is not None else (torch.zeros(100), torch.zeros(100)) 

        time = torch.tensor(item.time)  # PFS：Progression-Free Survival     OS：Overall Survival
        status = torch.tensor(item.status)  # 时间是否发生，PFS：病情是否发生后进展， OS：是否死亡

        return image, report, time, status, case_id


class CLSDataset(CoxRegDataset):   # 分类任务
    def __init__(self, df, config):
        super().__init__(df, config)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Parameters:
        - idx (int): Index of the sample.

        Returns:
        - tuple: Tuple containing image tensors, report ids/pads tensors, labels.
        """
        item = self.df.iloc[idx]

        case_id = str(item['case_id'])
        image_filename = str(item["filename"].replace(".pt",""))

        image = self._load_image(image_filename)  if self.image_dir is not None else torch.zeros(100)
        report = self._load_report(str(case_id)) if self.report_dir is not None else (torch.zeros(100), torch.zeros(100))
        
        if self.config['label_name']=='hazard_level':
            # merge hazard_level labels to 2-cls
            label = torch.tensor(0 if item[self.config['label_name']] == 0 else 1).long()
        else:
            label = torch.tensor(item[self.config['label_name']]).long()  

        return image, report, torch.tensor([0]).float(), label, case_id



def get_dataset_fn(dataset_name='cls'):
    assert dataset_name in ['coxreg', 'cls']
    if dataset_name == 'coxreg':  # cox-regression dataset  用于做回归任务
        return CoxRegDataset 
    elif dataset_name == 'cls':  # cox-regression dataset
        return CLSDataset
    else:
        raise NotImplementedError


"""
创建Datamodule  继承自LightningDataModule  用于处理数据集的加载和预处理
self.MAX_BAG = 6000   
一张WSI可以获得N个patch，若N<6000，则补全到6000个patch；若N>6000，则随机采样或选择前6000个patch
将6000个patch作为输出送入NEVA(MUSK+MIL)
"""

class WSIDataModule(LightningDataModule):
    def __init__(self, config, split_k=0, dist=True, data_df=None):   # dist 是否采用分布式训练策略
        super(WSIDataModule, self).__init__()
        """
        prepare datasets and samplers  
        """
        if data_df is not None:
            df = data_df.copy()
        else:
            df = pd.read_csv(config["Data"]["dataframe"])

        train_index = df[df["fold"] != split_k].index     # 5折交叉验证   4折是训练集、1折是测试集
        train_df = df.loc[train_index].reset_index(drop=True)  # 重置索引并将原来的索引列删去

        val_index = df[df["fold"] == split_k].index  # 留下一折当作验证集
        val_df = df.loc[val_index].reset_index(drop=True)

        # independent test cohort
        if config["Data"]["test_df"] is not None:  #  读取测试集
            test_df = pd.read_csv(config["Data"]["test_df"])
        # cross-validation test cohort; same as validation.
        else:
            test_df = deepcopy(val_df)  # 测试集从验证集copy出来

        dfs = [train_df, val_df, test_df]  # get training, test and validation datasets

        self.dist = dist  

        # get train, val, test dataset
        dataset_name = 'basic'
        if 'dataset_name' in config['Data'].keys():  # cls 或者 coxreg   cls 是只看
            dataset_name = config['Data']['dataset_name']

        # 定义 get item 和 len 函数
        self.datasets = [get_dataset_fn(dataset_name)(df, config["Data"]) for df in dfs] 

        self.dataset_name = dataset_name
        self.config = config

        self.batch_size = config["Data"]["batch_size"]
        self.num_workers = config["Data"]["num_workers"]

    def prepare_data(self) -> None: # 用于下载数据
        pass

    def setup(self, stage):  # setup是在多个GPU上    prepare_data 是在单个GPU上，一般用于下载原数据集
        
        # for training balanced sampler
        if self.dataset_name == 'cls':
            label_key = self.config["Data"].get("label_name")
            if label_key is None:
                raise KeyError("`label_name` must be provided in config['Data'] for classification datasets.")
            labels_list = self.datasets[0].df[label_key].to_numpy().astype(np.int64, copy=False)
        else:
            if 'status' not in self.datasets[0].df.columns:
                raise KeyError("`status` column is required for Cox regression datasets.")
            labels_list = self.datasets[0].df['status'].to_numpy().astype(np.int64, copy=False)

        if self.dist:   # 默认为True
            train_sampler = DistributedBalancedSampler(labels_list)
            val_sampler = DistributedSampler(
                self.datasets[1],
                num_replicas=train_sampler.num_replicas,
                rank=train_sampler.rank,
                shuffle=False,
            ) if len(self.datasets[1]) > 0 else None
            test_sampler = DistributedSampler(
                self.datasets[2],
                num_replicas=train_sampler.num_replicas,
                rank=train_sampler.rank,
                shuffle=False,
            ) if len(self.datasets[2]) > 0 else None
            self.samplers = [train_sampler, val_sampler, test_sampler]
        else:
            self.samplers = [BalancedSampler(labels_list), None, None]  # balanced samplers

    def train_dataloader(self):
        loader = DataLoader(
            self.datasets[0],
            shuffle=False,  
            batch_size=self.batch_size,
            sampler=self.samplers[0],  # 为None
            num_workers=self.num_workers,
            drop_last=False,     # ! 如果最后一个batch不满足batch_size，决定是否drop掉最后一个batch，这里建议不drop
            pin_memory=True,
            persistent_workers=True,     # 复用 worker，避免每个 epoch 重启 
        )
        return loader

    def val_dataloader(self):  # 在验证阶段，
        loader = DataLoader(
            self.datasets[1],
            shuffle=False,
            batch_size=self.batch_size,
            sampler =self.samplers[1],
            num_workers=self.num_workers,
            drop_last=False,   # ! 如果最后一个batch不满足batch_size，决定是否drop掉最后一个batch，这里建议不drop
            pin_memory=True,
            persistent_workers=True,     # 复用 worker，避免每个 epoch 重启
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.datasets[2],
            shuffle=False,
            batch_size=self.batch_size,
            sampler=self.samplers[2],
            num_workers=self.num_workers,
            drop_last=False,    # ! 如果最后一个batch不满足batch_size，决定是否drop掉最后一个batch，这里建议不drop
            pin_memory=True,
            persistent_workers=True,     # 复用 worker，避免每个 epoch 重启
        )
        return loader


class DistributedBalancedSampler(Sampler):
    """Distributed sampler that keeps per-class sampling balanced across ranks."""

    def __init__(self, dataset_labels, shuffle: bool = True, seed: int = 0):
        self.labels = np.asarray(dataset_labels)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0

        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in np.unique(self.labels)}
        if not self.label_to_indices:
            raise ValueError("DistributedBalancedSampler requires non-empty labels.")

        self.max_class_size = max(len(indices) for indices in self.label_to_indices.values())
        self.base_length = self.max_class_size * len(self.label_to_indices)

        self.num_samples = int(math.ceil(self.base_length / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)

        balanced_indices = []
        for indices in self.label_to_indices.values():
            if len(indices) == 0:
                continue
            sampled = rng.choice(indices, self.max_class_size, replace=True)
            balanced_indices.append(sampled)

        if not balanced_indices:
            raise ValueError("DistributedBalancedSampler could not sample any indices.")

        balanced_indices = np.concatenate(balanced_indices)

        if self.shuffle:
            rng.shuffle(balanced_indices)

        if len(balanced_indices) < self.total_size:
            padding = rng.choice(balanced_indices, self.total_size - len(balanced_indices), replace=True)
            balanced_indices = np.concatenate([balanced_indices, padding])
        else:
            balanced_indices = balanced_indices[:self.total_size]

        partitioned = balanced_indices[self.rank:self.total_size:self.num_replicas]
        return iter(partitioned.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class BalancedSampler(Sampler):   # 可以处理类别不平衡样本
    def __init__(self, dataset_labels):
        self.indices = []
        self.num_samples = 0

        # Create a list of indices for each class
        label_to_indices = {label: np.where(dataset_labels == label)[0] for label in np.unique(dataset_labels)}

        # Find the maximum size among the classes to balance
        largest_class_size = max(len(indices) for indices in label_to_indices.values())

        # Extend indices of smaller classes by sampling with replacement
        for indices in label_to_indices.values():
            indices_balanced = np.random.choice(indices, largest_class_size, replace=True)
            self.indices.append(indices_balanced)
            self.num_samples += largest_class_size

        # Flatten list and shuffle
        self.indices = np.random.permutation(np.hstack(self.indices))

    def __iter__(self):
        return iter(self.indices)


    def __len__(self):
        return self.num_samples
