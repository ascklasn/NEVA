import torch
from torch import nn
from models.clam import *
from torchscale.architecture.encoder import Encoder     
from torchscale.architecture.config import EncoderConfig
import importlib
def get_obj_from_str(image_mil_name, reload=False): 
    module, cls = image_mil_name.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)



class Slide_FM(nn.Module):  # TITAN、PRISM、
    def __init__(self,config):
        super().__init__()
        self.FM_type = config.get('FM_type')  # FMs Type，CHIEF, HIPT, SAM, DINOv2, MAE, etc.
        self.feat_dim = config.get('feat_dim', None)  # dim of image features
        self.report_feat_dim = config.get('report_feat_dim', None)  # dim of report features
        self.align_feat_dim = config.get('align_dim', None)  # align feature dim for multimodal fusion
        self.num_classes = config['num_classes']
        if self.FM_type == 'Slide_Multimodal' :
            self.align_image = nn.Linear(self.feat_dim, self.align_feat_dim)
            self.align_report = nn.Linear(self.report_feat_dim, self.align_feat_dim)
            self.classifier = nn.Linear(self.align_feat_dim, self.num_classes)
        elif self.FM_type == 'Slide_Vision':  # Slide_Vision
            self.classifier = nn.Linear(self.feat_dim, self.num_classes)

    def forward(self, x):

        image, report = x  # img: [N, image_dim], text: [N, text_dim]
        if self.FM_type == 'Slide_Multimodal':
            image = self.align_image(image)  # [N, align_dim]
            report = self.align_report(report)  # [N, align_dim]
            assert image.shape == report.shape, f"Image and report features must have the same shape, but got {image.shape} and {report.shape}"
            global_feat = image + report
        elif self.FM_type == 'Slide_Vision':
            assert torch.any(image != 0), "Report features cannot be all zeros for Slide_Report FM_type"
            global_feat = image

        logits = self.classifier(global_feat)  # [N, n_classes]
        return logits
    

class Patch_FM(nn.Module):  # UNI、Virchow、CONCH、HOPT、PLIP、ResNet50
    def __init__(self,config):
        super().__init__()
        self.FM_type = config.get('FM_type')  # FM types，UNI、CONCH_v1.5、HOPT、PLIP、ResNet50
        self.feat_dim = config['feat_dim']
        self.report_dim = config.get('report_dim', None)  # dim of report features
        self.align_dim = config.get('align_dim', None) # align feature dim for multimodal fusion
        self.num_classes = config['num_classes']
        self.image_mil_name = config['image_mil_name']  # "models.clam.CLAM_Batch"
        self.mil_params = config['mil_params']  

        self.image_mil = get_obj_from_str(self.image_mil_name)(   
            feat_dim=self.feat_dim, 
            n_classes=self.num_classes,  
            **self.mil_params)
        if self.report_dim is not None and self.align_dim is not None:
            self.align_image = nn.Linear(self.feat_dim, self.align_dim)
            self.align_report = nn.Linear(self.report_dim, self.align_dim)
            self.classifier = nn.Linear(self.align_dim, self.num_classes)

    def forward(self, x):
        image, report = x  # img: [N, image_dim], text: [text_dim]
        if self.report_dim is not None and self.align_dim is not None:
            image_feats, _, _, _ = self.image_mil((image, None))  #  [N, feat_dim]
            image_aligned = self.align_image(image_feats)  # [N, align_dim]
            report_aligned = self.align_report(report)  # [align_dim]
            assert image_aligned.shape[1] == report_aligned.shape[0], f"Image and report features must have the same shape, but got {image_aligned.shape} and {report_aligned.shape}"
            global_feat = image_aligned + report_aligned  # [N, align_dim]
            logits = self.classifier(global_feat)  # [N, n_classes]
        else:
            logits, _, _, _ = self.image_mil((image, None))
        return logits