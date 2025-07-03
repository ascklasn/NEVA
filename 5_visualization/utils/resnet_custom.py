# modified from Pytorch official resnet.py
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import torchvision.models as models
from collections import OrderedDict

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Bottleneck_Baseline(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_Baseline, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Baseline(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet_Baseline, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


def resnet50_baseline(pretrained=False):
    """Constructs a Modified ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_Baseline(Bottleneck_Baseline, [3, 4, 6, 3])
    td = torch.load("/mnt/group-ai-medical-sz/private/jinxixiang/code/huayin/"
                    "extract_feat/models/resnet50-19c8e357.pth", map_location="cpu")
    model.load_state_dict(td, strict=False)

    return model


def load_pretrained_weights(model, name):
    pretrained_dict = model_zoo.load_url(model_urls[name])
    model.load_state_dict(pretrained_dict, strict=False)
    return model



class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=True, avgF=False, ext="k"):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        avgF: if extract average pooling layer features of encoders
        ext: which encoder to extract features, 'k' or 'q'
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.avgF = avgF
        self.ext = ext
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, im_k):
        if self.ext == "k":
            encoder = self.encoder_k
        elif self.ext == "q":
            encoder = self.encoder_q
        else:
            print("error encoder type for feature extraction")
            raise TypeError

        if self.avgF:
            output = encoder.conv1(im_k)
            output = encoder.bn1(output)
            output = encoder.relu(output)
            output = encoder.maxpool(output)
            output = encoder.layer1(output)
            output = encoder.layer2(output)
            output = encoder.layer3(output)
            output = encoder.layer4(output)
            output = encoder.avgpool(output)
            # print("avg size", output.size())
            output_avg = output.squeeze(-1).squeeze(-1)
            output_avg_norm = nn.functional.normalize(output_avg, dim=1)
            output_fc = encoder.fc(output_avg)
            # print("fc size", output_fc.size())
            output_fc_norm = nn.functional.normalize(output_fc, dim=1)

            return output_avg_norm

        else:
            k = encoder(im_k)  # keys: NxC
            k1 = nn.functional.normalize(k, dim=1)

            return k, k1
            # return k1


class MoCoys(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07,
                    mlp=True, mgd=False, descriptors=None, p=3, normalize=False, attention=False, mpp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
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
        # if mgd and descriptors is not None:
        #     dim_mlp = self.encoder_q.fc.weight.shape[1]
        #     self.encoder_q.avgpool = MGD(dim_mlp, dim_mlp, descriptors=descriptors, p=p)
        #     self.encoder_k.avgpool = MGD(dim_mlp, dim_mlp, descriptors=descriptors, p=p)

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
        # print(q.size())
        return q


def create_moco():
    net = MoCo(models.__dict__['resnet50'], dim=128, avgF=True)
    pretext_model = torch.load('./self_sup/checkpoint_0039.pth.tar')['state_dict']

    td = OrderedDict()
    for key, value in pretext_model.items():
        k = key[7:]
        td[k] = value
    net.load_state_dict(td)

    return net


def create_mocoys():
    net = MoCoys(models.__dict__['resnet50'], dim=128)
    pretext_model = torch.load('./self_sup/checkpoint_0039.pth.tar', map_location='cpu')['state_dict']
    td = OrderedDict()
    for key, value in pretext_model.items():
        k = key[7:]
        td[k] = value
    net.load_state_dict(td)

    net.encoder_q.fc = nn.Identity()
    net.encoder_q.instDis = nn.Identity()
    net.encoder_q.groupDis = nn.Identity()

    return net

from timm.models.layers.helpers import to_2tuple

class ConvStem(nn.Module):
    """
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 4, 'ConvStem only supports patch size of 16'
        assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 8 for ConvStem'

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


if __name__ == "__main__":
    net = create_mocoys()
    x = torch.randn(1, 3, 256, 256)
    y = net(x)
    print(f"y shape: {y.shape}")