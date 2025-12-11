import torch.nn as nn
from collections import  OrderedDict
import torch
import torchvision.models as models


class MoCoys(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07,
                    mlp=True, normalize=False, attention=False, mpp=False):

        super(MoCoys, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.normalize = normalize

        # create the encoders
        # num_classes is the output fc dimension
        if attention:
          self.encoder_q = base_encoder(num_classes=dim, attention=attention)
          self.encoder_k = base_encoder(num_classes=dim, attention=attention)
        else:
          self.encoder_q = base_encoder(num_classes=dim)
          self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        if mpp:
            self.mpp_predictor_q = nn.Linear(dim, 2)
            self.mpp_predictor_k = nn.Linear(dim, 2)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self,im_q):
        q = self.encoder_q(im_q)  # querys: NxC
        if self.normalize:
            q = nn.functional.normalize(q, dim=1)
        return q


def create_mocoys(pretrained_weight):
    net = MoCoys(models.__dict__['resnet50'], dim=128)
    pretext_model = torch.load(pretrained_weight, map_location="cpu")['state_dict']
    td = OrderedDict()
    for key, value in pretext_model.items():
        k = key[7:]
        td[k] = value
    net.load_state_dict(td)
    net.encoder_q.fc = nn.Identity()
    net.encoder_q.instDis = nn.Identity()
    net.encoder_q.groupDis = nn.Identity()

    return net