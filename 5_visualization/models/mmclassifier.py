import torch
import torch.nn as nn
import importlib
from einops import rearrange
import torch.nn.functional as F

"""
We proposen-lang to use visiouage features for outcome prediction.
"""

def get_obj_from_str(image_mil_name, reload=False):  # Dynamic import required classes
    module, cls = image_mil_name.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class Pooler(nn.Module):  # First, LN layer normalization, then linear mapping, and reactivate function
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
    Commonly used parameter explanation:
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

        target_dim = mil_params['hidden_feat']  

        self.image_mil = get_obj_from_str(image_mil_name)(   # Read image encoder CLAM_Batch class
            feat_dim=feat_dim,  
            n_classes=num_classes,  
            **mil_params
        )

        self.feat_dim = feat_dim  
        self.fc_vision = Pooler(target_dim, target_dim)  
        
        report_feat_dim = 1024
        self.fc_report = Pooler(report_feat_dim, target_dim)    

        # Classifiers
        self.classifier_vision = nn.Linear(target_dim, num_classes)   
        self.classifier_report = nn.Linear(target_dim, num_classes)   

        self.classifier_final = nn.Sequential(nn.LayerNorm(target_dim),
                                            nn.Linear(target_dim,256),    
                                            nn.Sigmoid(),
                                            nn.Linear(256,128), 
                                            nn.Linear(128, num_classes)) 
        
        self.w1 = nn.Parameter(torch.tensor(0.5))  # The initial value is 0.5
        self.w2 = nn.Parameter(torch.tensor(0.5))  # The initial value is 0.5

    def forward(self, batch):
        # print(batch.size())
        images,reports = batch
        feat_vision = None
        feat_report = None
        
        if images is not None and images.any():  
            feat_vision, _ = self.image_mil((images, None), return_global_feature=True) 
            feat_vision = self.fc_vision(feat_vision)     
        
        # Aggregate report
        if reports is not None and reports.any():
            feat_report = self.fc_report(reports)  
        
        results_dict = {}
        
        # multimodal 
        if feat_report is not None and feat_vision is not None:

            weight_sum = self.w1 + self.w2
            w1_normalized = self.w1 / weight_sum
            w2_normalized = self.w2 / weight_sum
            global_feat = w1_normalized * feat_vision + w2_normalized * feat_report

            # Additional loss
            logits_vision = self.classifier_vision(feat_vision)     
            logits_report = self.classifier_report(feat_report)    
            results_dict.update({"logits_vision": logits_vision, "logits_report": logits_report}) 

        # report-only
        elif feat_report is not None:
            global_feat = feat_report
            logits_report = self.classifier_report(feat_report)   
            results_dict.update({"logits_report": logits_report})
        # vision-only
        elif feat_vision is not None:
            logits_vision = self.classifier_vision(feat_vision) 
            global_feat = feat_vision
            results_dict.update({"logits_vision": logits_vision})
        else:
            raise NotImplementedError
        
        logits = self.classifier_final(global_feat)
        
        return logits, results_dict
