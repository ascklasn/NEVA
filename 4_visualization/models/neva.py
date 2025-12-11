import torch
from timm.models import create_model
from musk import utils, modeling
from huggingface_hub import login
# Use your own Huggingface TOKEN
# my_token = 'hf_xxx'
# login(my_token)
import re
import torch.nn as nn
import importlib
from peft import LoraConfig, get_peft_model, TaskType
from einops import rearrange

# ========= Enumerate Linear layers to inject LoRA =========
def collect_lora_targets(
    root: nn.Module,
    mode: str = "qv",          # "qv", "qk", "attn", "ffn", "attn+ffn", "all"
    num_last_layers: int = None,
    include_multiway: bool = True,
    as_regex: bool = False,    # if True, return a single regex instead of explicit names
):
    patterns = {
        "qk": r"self_attn\.(q_proj|k_proj)",
        "qv": r"self_attn\.(q_proj|v_proj)",
        "attn": r"self_attn\.(q_proj|k_proj|v_proj|out_proj)",
        "ffn": r"ffn\..*\.(fc1|fc2)",
        "attn+ffn": r"(self_attn\.(q_proj|k_proj|v_proj|out_proj))|(ffn\..*\.(fc1|fc2))",
        "all": r"(self_attn\.(q_proj|k_proj|v_proj|out_proj))|(ffn\..*\.(fc1|fc2))",
    }
    if as_regex:
        return patterns[mode]

    prefixes = None
    if num_last_layers:
        prefixes = _get_last_layer_prefixes(root, num_last_layers)

    pat = re.compile(patterns[mode])
    keys = []
    for name, module in root.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if prefixes and not name.startswith(prefixes):
            continue
        if pat.search(name):
            if include_multiway or not re.search(r"\.(A|B)(\.|$)", name):
                keys.append(name)
    return sorted(set(keys))

def _get_last_layer_prefixes(root: nn.Module, num_last_layers: int):
    idxs = set()
    for name, _ in root.named_modules():
        m = re.match(r"^beit3\.encoder\.layers\.(\d+)(?:\.|$)", name)
        if m:
            idxs.add(int(m.group(1)))
    if not idxs:
        return tuple()
    L = max(idxs) + 1
    num = max(1, min(num_last_layers, L))
    return tuple(f"beit3.encoder.layers.{i}." for i in range(L - num, L))


def get_musk_lora(musk_config="musk_large_patch16_384"):
    # Fixed model configuration
    musk_config = "musk_large_patch16_224"
    musk = create_model(musk_config)
    
    # Load pre-trained weights
    utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", musk, 'model|module', '')
    
    # Target modules for LoRA injection (only last encoder layer)
    target_modules = collect_lora_targets(musk, mode="qv", num_last_layers=3)  # Q+V, last 3; Most efficient strong baseline
    # target_modules = collect_lora_targets(musk, mode="attn", num_last_layers=4)  # Q,K,V,O; A bit more capacity

    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,  # encoder/CLIP-style usage
        inference_mode=False,
        r=8,                 # rank for attention
        lora_alpha=16,       # ~2×r is a good starting point
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
    )

    # Apply LoRA to the model
    musk_lora = get_peft_model(musk, lora_cfg)
    musk_lora.print_trainable_parameters()  # Verify only LoRA weights are trainable
    
    return musk_lora


def get_obj_from_str(image_mil_name, reload=False):
    """Dynamically import the required class"""
    module, cls = image_mil_name.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class MLPPooler(nn.Module):
    """Lightweight MLP Pooler with minimal parameters"""
    def __init__(self, input_features, output_features):
        super().__init__()
        # Single linear layer with LayerNorm and GELU activation
        self.norm = nn.LayerNorm(input_features)
        self.linear = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()

    def forward(self, x):
        # x of shape [batch_size, feat_dim]
        x = self.norm(x)
        x = self.linear(x)
        x = self.activation(x)
        return x

class ConcatMLPFusion(nn.Module):
    """Concatenate features then mix them with a lightweight MLP."""
    def __init__(self, feature_dim, hidden_multiplier: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden_dim = feature_dim * hidden_multiplier
        self.norm_vision = nn.LayerNorm(feature_dim)
        self.norm_report = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, vision_feat, report_feat):
        vision_norm = self.norm_vision(vision_feat)
        report_norm = self.norm_report(report_feat)
        fused = torch.cat([vision_norm, report_norm], dim=-1)
        fused = self.mlp(fused)
        return self.dropout(fused)


class NEVA(nn.Module):
    """
    Multi-Modal classifier combining vision and language for outcome prediction.
    Common parameter explanations:
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

    def __init__(self, config):
        super(NEVA, self).__init__()
        self.return_feat = config.get('return_feat', False)

        image_mil_name = config['image_mil_name']  # "models.clam.CLAM_Batch"
        mil_params = config['mil_params']          # mil的参数
        feat_dim = config['feat_dim']              # 1024
        num_classes = config['num_classes']        # 2
        self.image_only = config.get('image_only', False)
        self.report_only = config.get('report_only', False)
        self.patch_probs = config.get('patch_probs', False)

        self.musk_lora = get_musk_lora()    # Get musk_lora model

        target_dim = mil_params['hidden_feat']    

        # Load image encoder (CLAM_Batch class)
        self.image_mil = get_obj_from_str(image_mil_name)(
            feat_dim=feat_dim, 
            n_classes=num_classes,  
            **mil_params
        )

        self.feat_dim = feat_dim  # 1024
        # self.fc_vision = MLPPooler(target_dim, target_dim)  
        self.fc_report = MLPPooler(feat_dim, target_dim)    

        # Concatenation fusion module
        self.fusion_head = ConcatMLPFusion(target_dim)

        # Classifiers
        self.classifier_vision = nn.Linear(target_dim, num_classes)   # hidden=512 --> num_classes
        self.classifier_report = nn.Linear(target_dim, num_classes)   # hidden=512 --> num_classes

        # Final classifier after fusing image and text features
        self.classifier_final = nn.Linear(target_dim, num_classes)  

    def forward(self, batch):
        """
        Batch contains images and reports
        images: [batch_size, bag_size, 3, 768, 768]
        reports contains reports_ids and reports_pads
        """
        images, reports = batch
        reports_ids, reports_pads = reports

        if self.report_only:
        # Encode reports using musk_lora
            _, reports_embeddings = self.musk_lora(
                with_head=False, 
                out_norm=False,
                ms_aug=False, 
                return_global=True,
                text_description=reports_ids,  # [bsz, 100] tensor 
                padding_mask=reports_pads,     # [bsz, 100] tensor
            )  # returns (vision_cls, text_cls)
    
            # Process report features
            if reports_embeddings is not None and torch.any(reports_embeddings):
                feat_report = self.fc_report(reports_embeddings)  # 1024->target_dim

            logits_report = self.classifier_report(feat_report)   # 128--> num_classes
            results_dict = {}
            results_dict.update({"logits_report": logits_report})

            return logits_report, results_dict
        
        if self.image_only:
            bsz, bag_size = images.shape[0], images.shape[1]
                # Rearrange images for processing
            images_in = rearrange(images, "b g c h w -> (b g) c h w")
            
            # Encode images using musk_lora
            image_embeddings, _ = self.musk_lora(
                image=images_in,
                with_head=False, 
                out_norm=False,
                ms_aug=False, 
                return_global=True,
            )  # returns (vision_cls, text_cls)
            
            image_embeddings = rearrange(image_embeddings, "(b g) d -> b g d", b=bsz)
            # Aggregate WSI features using CLAM (essentially Batch ABMIL)
            if image_embeddings is not None and torch.any(image_embeddings):
                feat_vision, _ = self.image_mil(
                    (image_embeddings, None), 
                    return_global_feature=True
                )  
            logits_vision = self.classifier_vision(feat_vision)  # 128--> num_classes
            results_dict = {}
            results_dict.update({"logits_vision": logits_vision})
            return logits_vision, results_dict

        else:        
            bsz, bag_size = images.shape[0], images.shape[1]
            # Rearrange images for processing
            images_in = rearrange(images, "b g c h w -> (b g) c h w")
            
            # Encode images using musk_lora
            image_embeddings, _ = self.musk_lora(
                image=images_in,
                with_head=False, 
                out_norm=False,
                ms_aug=False, 
                return_global=True,
            )  # returns (vision_cls, text_cls)
            
            image_embeddings = rearrange(image_embeddings, "(b g) d -> b g d", b=bsz)
            
            # Encode reports using musk_lora
            _, reports_embeddings = self.musk_lora(
                with_head=False, 
                out_norm=False,
                ms_aug=False, 
                return_global=True,
                text_description=reports_ids,  # [bsz, 100] tensor 
                padding_mask=reports_pads,     # [bsz, 100] tensor
            )  # returns (vision_cls, text_cls)

            feat_vision = None
            feat_report = None
            
            # Aggregate WSI features using CLAM (essentially Batch ABMIL)
            if image_embeddings is not None and torch.any(image_embeddings):
                feat_vision, patch_probs = self.image_mil(
                    (image_embeddings, None), 
                    return_global_feature=True
                )
                if self.patch_probs:
                    return patch_probs['patch_probs']  
                
            # Process report features
            if reports_embeddings is not None and torch.any(reports_embeddings):
                feat_report = self.fc_report(reports_embeddings)  # 1024->target_dim
            
            results_dict = {}
            if self.return_feat:
                results_dict.update({"feat_vision": image_embeddings, "feat_report": reports_embeddings})
            
            # Multimodal fusion
            if feat_report is not None and feat_vision is not None:
                # Use concatenation-based fusion instead of simple addition
                global_feat = self.fusion_head(feat_vision, feat_report)

                # Additional losses
                logits_vision = self.classifier_vision(feat_vision)     # 128 --> num_classes
                logits_report = self.classifier_report(feat_report)     # 128 --> num_classes
                results_dict.update({"logits_vision": logits_vision, "logits_report": logits_report})
            
            # Report-only mode
            elif feat_report is not None:
                global_feat = feat_report
                logits_report = self.classifier_report(feat_report)   # 128--> num_classes
                results_dict.update({"logits_report": logits_report})
            
            # Vision-only mode
            elif feat_vision is not None:
                logits_vision = self.classifier_vision(feat_vision)  # 128--> num_classes
                global_feat = feat_vision
                results_dict.update({"logits_vision": logits_vision})
            
            else:
                raise NotImplementedError("At least one modality must be available")
            
            # Final classification
            logits = self.classifier_final(global_feat)

            return logits, results_dict
