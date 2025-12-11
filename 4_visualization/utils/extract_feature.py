import torch
import torch.nn as nn
import h5py
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.utils import collate_features
from .moco import create_mocoys
from .resnet_custom import resnet50_baseline

import torch, torchvision
import torch.nn as nn
import timm
from timm.models.layers.helpers import to_2tuple
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

import sys
# sys.path.append("/home/huruizhen/mil_dataset_1024/MUSK/musk")

class ConvStem(nn.Module):
    """
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


def load_feature_model(device, pretrained="moco"):
    print('loading model checkpoint')

    if pretrained == "moco":
        model = create_mocoys("/mnt/group-ai-medical-sz/private/jinxixiang/code/PANDA/"
                              "prostate_releasev1/weights/encoder.pth.tar")
    elif pretrained == "imagenet":
        model = resnet50_baseline(pretrained=True)

    elif pretrained == "swin_mocov3":
        model = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
        model.head = nn.Identity()
        pretrained_weight = r'/mnt/group-ai-medical-sz/private/scusenyang/mocov3/' \
                            r'convert_path/swin_conv_best/checkpoint_0030.pth'
        td = torch.load(pretrained_weight)["model"]
        model.load_state_dict(td, strict=True)

    elif pretrained == "OMAP":
        from timm.models import create_model
        import bertpath.utils as butils
        import bertpath.modeling_finetune

         # >>>>>>>>>>>> load model >>>>>>>>>>>> #
        model_config = "bertpath_large_patch16_384_retrieval"
        model = create_model(model_config, vocab_size=64010).eval()
        model_path = "/mnt/sdd/vl_bertpath/3_contrastive_finetuning/scripts/results/bertpath_large_384/model.pth"
        butils.load_model_and_may_interpolate(model_path, model, 'model|module', '')

    elif pretrained == "Musk":  # 还有问题需要修改
        # login('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')
        # os.environ["http_proxy"] = "http://127.0.0.1:7890" # 代理设置
        # os.environ["https_proxy"] = "http://127.0.0.1:7890" # 代理设置
        # >>>>>>>>>>>> load model >>>>>>>>>>>> #
        # model_config = "musk_large_patch16_384"
        # from timm.models import create_model
        # model = create_model(model_config).eval()
        # utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, 'model|module', '')
        # device = torch.device("cuda:0")
        # model.to(device, dtype=torch.float16)
        # model.eval()
        print('模型加载完毕')
        return 
        # 之后还要加载musk模型的权重
        # <<<<<<<<<<<< load model <<<<<<<<<<<< #


    else:
        raise NotImplementedError

    model = model.to(device, non_blocking=True)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()
    return model


def eval_transforms(target_size):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    trans = transforms.Compose(
        [transforms.Resize(target_size),
         transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std)
         ]
    )
    return trans


class WholeSlidePatchBag(Dataset):
    def __init__(self, patch_coord, wsi, patch_level=0, patch_size=256,
                 custom_downsample=1, target_patch_size=-1, transforms=None):
        self.wsi = wsi
        self.coords = patch_coord

        self.patch_level = patch_level
        self.patch_size = patch_size

        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size,) * 2
        elif custom_downsample > 1:
            self.target_patch_size = (self.patch_size // custom_downsample,) * 2
        else:
            self.target_patch_size = None
        
        if transforms is None:
            self.roi_transforms = eval_transforms(self.target_patch_size)
        else:
            self.roi_transforms = transforms

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        coord = self.coords[idx]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord


def extract_feature_with_coord(wsi_slide, patch_coord, batch_size=256, verbose=1, print_every=1,
                               custom_downsample=1, patch_size=256, target_patch_size=-1, device=None, model=None):
    
    if model.__class__.__name__ == "BERTPath":
        trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize(target_patch_size, interpolation=3, antialias=True),
            torchvision.transforms.CenterCrop((target_patch_size, target_patch_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])
    else:
        trans = None

    patch_dataset = WholeSlidePatchBag(patch_size=patch_size,
        patch_coord=patch_coord, wsi=wsi_slide, custom_downsample=custom_downsample,
        target_patch_size=target_patch_size, transforms=trans)

    if device == "cuda":
        patch_loader = DataLoader(
            dataset=patch_dataset, batch_size=batch_size,
            num_workers=32, pin_memory=True, collate_fn=collate_features)
    else:
        patch_loader = DataLoader(
            dataset=patch_dataset, batch_size=batch_size,
            collate_fn=collate_features)

    if verbose > 0:
        print('processing total of {} batches'.format(len(patch_loader)))

    feature_bag = []

    for count, (batch, coords) in enumerate(patch_loader):
        with torch.no_grad():
            print('batch {}/{}, {} files processed'.format(count, len(patch_loader), count * batch_size))
            batch = batch.to(device, non_blocking=True)

            if model is not None:

                if model.__class__.__name__ == "BERTPath":
                    # This setting must be consistent with the features used for training MIL
                    features = model(image=batch, with_head=True, out_norm=True)[0]  

                else:
                    features = model(batch)

            else:
                features = torch.zeros(1)
            feature_bag.append(features)
    
    feature_bag = torch.cat(feature_bag)

    return feature_bag
