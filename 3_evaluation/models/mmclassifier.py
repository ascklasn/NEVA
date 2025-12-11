import torch
from timm.models import create_model
from musk import utils, modeling
from PIL import Image
from transformers import XLMRobertaTokenizer
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import torchvision
import os
import pandas as pd
import numpy as np
from IPython.display import display
import h5py
import matplotlib.pyplot as plt
# import opensdpc   
# # todo sdpc 这个包目前安装失败，参考  https://github.com/WonderLandxD/opensdpc
import glob
import openslide
import re
import oven
import logging
import torchvision
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import huggingface_hub
from huggingface_hub import login, hf_hub_download
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
import timm
from tqdm import tqdm
from torchvision import transforms
import importlib
import os, re, torch, torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft import get_peft_model_state_dict


# ========= 枚举将注入 LoRA 的 Linear =========

import re
import torch.nn as nn

def collect_lora_targets_last_layers(root: nn.Module,
                                     num_last_layers: int = 1):
    """
    仅在 beit3.encoder.layers 的最后 num_last_layers 个 block 中收集 Linear 作为 LoRA 注入目标。
    目标 Linear 的命名仍限定为：
      self_attn.{q_proj|k_proj|v_proj|out_proj} 或 ffn.*.{fc1|fc2}
    兼容 Multiway(A/B) 命名，例如：
      ...self_attn.q_proj.A / ...self_attn.q_proj.B / ...ffn.A.fc1 / ...ffn.B.fc2

    参数
    ----
    root : nn.Module
        模型根模块（包含 beit3.encoder.layers）
    num_last_layers : int
        只选最后多少个 encoder block（默认 1）
    """
    # 1) 先尽量直接从属性拿到 ModuleList（最可靠）
    last_indices = None
    layers = None
    try:
        layers = getattr(getattr(getattr(root, "beit3"), "encoder"), "layers")
    except Exception:
        layers = None

    if isinstance(layers, nn.ModuleList) and len(layers) > 0:
        L = len(layers)
        num = max(1, min(num_last_layers, L))
        last_indices = list(range(L - num, L))
    else:
        # 2) 兜底：从 named_modules() 里解析出最大层号
        idxs = set()
        for name, _ in root.named_modules():
            m = re.match(r"^beit3\.encoder\.layers\.(\d+)(?:\.|$)", name)
            if m:
                idxs.add(int(m.group(1)))
        if idxs:
            L = max(idxs) + 1
            num = max(1, min(num_last_layers, L))
            last_indices = list(range(L - num, L))
        else:
            # 找不到就直接返回空
            return []

    # 3) 构造这些“最后层”的名前缀，用于快速过滤
    prefixes = tuple(f"beit3.encoder.layers.{i}." for i in last_indices)

    # 4) 保持原有线性层匹配规则（attn 的 q/k/v/out 与 ffn 的 fc1/fc2）
    pat = re.compile(r"(self_attn\.(q_proj|k_proj|v_proj|out_proj))|(ffn\..*\.(fc1|fc2))")

    keys = []
    for name, module in root.named_modules():
        if not name.startswith(prefixes):
            continue
        if isinstance(module, nn.Linear) and pat.search(name):
            keys.append(name)

    return sorted(set(keys))

def collect_lora_targets(root: nn.Module):
    """
    仅挑 Linear，名字里含 q_proj/k_proj/v_proj/out_proj 或 ffn.fc1/ffn.fc2。
    兼容 Multiway(A/B) 的命名，例如：
    beit3.encoder.layers.0.self_attn.q_proj.A / ...B / ffn.A.fc1 / ffn.B.fc2
    """
    keys = []
    for name, module in root.named_modules():
        if isinstance(module, nn.Linear):
            if re.search(r"(self_attn\.(q_proj|k_proj|v_proj|out_proj))|(ffn\..*\.(fc1|fc2))", name):
                keys.append(name)
    return sorted(set(keys))

def get_musk_lora(musk_config = "musk_large_patch16_384" ):

    musk_config = "musk_large_patch16_384"
    musk = create_model(musk_config)
    utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", musk, 'model|module', '')
    # target_modules = collect_lora_targets(musk)
    target_modules = collect_lora_targets_last_layers(musk,num_last_layers=1)  # ! 只对最后一个encoder进行lora
    
    lora_cfg = LoraConfig(
        r=8,                # rank，可按显存改为 4/16
        lora_alpha=16,      # scaling
        lora_dropout=0.05,  # 训练时 dropout
        bias="none",
        target_modules=target_modules,   # 精确到 Linear 的路径
        task_type="FEATURE_EXTRACTION",  # 非 LM 任务，用特征/分类更合适
    )
    musk_lora = get_peft_model(musk, lora_cfg)
    musk_lora.print_trainable_parameters()  # 验证只有 LoRA 权重可训练, 输出musk_lora的训练参数

    return musk_lora


def get_obj_from_str(image_mil_name, reload=False):  # 动态导入所需要的类
    module, cls = image_mil_name.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class Pooler(nn.Module):  # 先LN层归一化、后线性映射、再激活函数
    def __init__(self, input_features, output_features, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        # x of shape [batch_size, feat_dim]
        cls_rep = self.norm(x)
        pooled_output = self.dense(cls_rep)
        # pooled_output = self.activation(pooled_output)
        return pooled_output


class MMClassifier(nn.Module):
    """
    Multi-Modal classifier combining vision and language for outcome prediction.
    常用的 参数解释：
    image_mil_name: "models.CLAM_Batch"
    mil_params:
        hidden_feat: 128
        gate: true
        size_arg: 'small'
        dropout: true
        instance_eval: false
        subtyping: false
        k_sample: 16
    feat_dim: 768
    num_classes: 2
    """

    def __init__(self, image_mil_name, mil_params, feat_dim, num_classes):
        super(MMClassifier, self).__init__()

        self.musk_lora = get_musk_lora()    # 获得 musk_lora 模型

        target_dim = mil_params['hidden_feat']    

        self.image_mil = get_obj_from_str(image_mil_name)(   # 读取图像编码器  CLAM_Batch类
            feat_dim=feat_dim,  # 2048
            n_classes=num_classes,  
            **mil_params
        )

        self.feat_dim = feat_dim  
        self.fc_vision = Pooler(target_dim, target_dim)  # 1
        
        report_feat_dim = 1024
        self.fc_report = Pooler(report_feat_dim, target_dim)    #  文本特征从1024维度映射到 target_dim ，让图像特征和文本特征对齐

        # Classifiers
        self.classifier_vision = nn.Linear(target_dim, num_classes)   # hidden=512 --> num_classes
        self.classifier_report = nn.Linear(target_dim, num_classes)   # hidden=512 --> num_classes


        # 图像特征和文本特征融合之后，再使用 classifier_final 进行分类/回归
        self.classifier_final = nn.Sequential(nn.LayerNorm(target_dim),
                                            nn.Linear(target_dim,256),    # 先到256  再到num_class
                                            nn.Sigmoid(),
                                            # nn.Dropout(0.1),
                                            nn.Linear(256,128), 
                                            nn.Linear(128, num_classes))   # 128 --> num_classes
        
        # 多模态特征融合时，图像和文本的初始权重均为 0.5
        self.w1 = nn.Parameter(torch.tensor(0.5))  # 初始值为 0.5
        self.w2 = nn.Parameter(torch.tensor(0.5))  # 初始值为 0.5

    def forward(self, batch):
        """
        这里的 batch 包含 images 和reports
        images ： [MAX_BAG, 3, 768, 768]
        reports包含了 reports_ids 和 reports_pads
        """

        images,reports = batch
        images = torch.squeeze(images, dim=0)
        reports_ids, reports_pads = reports

        # 使用 musk_lora 获得 图像特征和文本特征，注意musk是 patch-level的foundation model
        image_embeddings,reports_embeddings  = self.musk_lora(
            image=images,
            with_head=False, # We only use the retrieval head for image-text retrieval tasks.
            out_norm=True,
            ms_aug=True,  # by default it is False, `image_embeddings` will be 1024-dim; if True, it will be 2048-dim.
            scales=[0.5,1],
            max_split_size=384,
            return_global=True,
            text_description=reports_ids,
            padding_mask=reports_pads,
            )  # return (vision_cls, text_cls)

        image_embeddings = torch.unsqueeze(image_embeddings, dim=0)  # [MAX_BAG,2048] --> [1, MAX_BAG,2048],batch_size=1

        feat_vision = None
        feat_report = None
        
        # Aggregate WSI   进行aggregation  使用了 clam  实际上是Batch版本的abmil   因为instance_eval=False，不计算instance损失
        if image_embeddings is not None and image_embeddings.any():  # images.any() 只要至少存在一个非零值则返回True
            feat_vision, _ = self.image_mil((image_embeddings, None), return_global_feature=True)  # global_feature 512 维度
            # global_feature=True   那么这时候feat_vision 的形状 [b,d]  b :batch_size  d:每个bag的最终向量
            feat_vision = self.fc_vision(feat_vision)     # 768-->768
        
        # Aggregate report
        if reports_embeddings is not None and reports_embeddings.any():
            feat_report = self.fc_report(reports_embeddings)  # 1024->768
        
        results_dict = {}
        
        # multimodal 
        if feat_report is not None and feat_vision is not None:
            weight_sum = self.w1 + self.w2
            w1_normalized = self.w1 / weight_sum
            w2_normalized = self.w2 / weight_sum
            global_feat = w1_normalized * feat_vision + w2_normalized * feat_report  # 多模态特征加权求和

            # Additional loss
            logits_vision = self.classifier_vision(feat_vision)     # 128 --> num_classes
            logits_report = self.classifier_report(feat_report)     # 128 --> num_classes
            results_dict.update({"logits_vision": logits_vision, "logits_report": logits_report})  # 更新字典

        # report-only
        elif feat_report is not None:
            global_feat = feat_report
            logits_report = self.classifier_report(feat_report)   # 128--> num_classes
            results_dict.update({"logits_report": logits_report})
        # vision-only
        elif feat_vision is not None:
            logits_vision = self.classifier_vision(feat_vision)  # 128--> num_classes
            global_feat = feat_vision
            results_dict.update({"logits_vision": logits_vision})
        else:
            raise NotImplementedError
        
        logits = self.classifier_final(global_feat)

        return logits, results_dict


