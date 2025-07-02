import torch
from torch import nn
from models.clam import *
from torchscale.architecture.encoder import Encoder      # 一般的vit的encoder_block
from torchscale.model.LongNet import LongNetEncoder      # longvit 的encoder_block
from torchscale.architecture.config import EncoderConfig
# from longvit_dilated_attention import *


class Pooler(nn.Module):  # LN layer normalization first, linear mapping, and reactivate functionalization, then linear mapping, and reactivate function
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








class vision_only(nn.Module): 
    def __init__(self,**kwargs):
        super(vision_only, self).__init__()
        self.model = CLAM_Batch(**kwargs)  

    def forward(self, x):
        logits, Y_prob, Y_hat, results_dict = self.model(x)
        return logits,Y_prob, Y_hat, results_dict  


