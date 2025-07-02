import os
import glob
import random
import base64
from io import BytesIO
from PIL import Image
import logging
from typing import Any, Tuple

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from timm.models import create_model
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import musk.utils as butils
import h5py
from musk import utils, modeling
import openslide

from huggingface_hub import login
my_token = 'hf_xxxxx' # replace with your Hugging Face token
login(my_token)

logging.basicConfig(level=logging.INFO)


class WSIBag(Dataset):
    def __init__(self,
                 file_path,
                 wsi_path,
                 custom_downsample=1,
                 target_patch_size=-1,
                 transform=None
                 ):
        """
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			custom_transforms (callable, optional): Optional transform to be applied on a sample
			custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
			target_patch_size (int): Custom defined image size before embedding
		"""
        self.wsi = openslide.open_slide(wsi_path)
        self.file_path = file_path
        self.trans = transform

        self.hdf5_file = h5py.File(file_path, "r", driver='core', backing_store=False)  # Memory-mapped
        dset = self.hdf5_file['coords']
        self.patch_level = dset.attrs['patch_level']
        self.patch_size = dset.attrs['patch_size']
        self.length = len(dset)
        
        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size,) * 2
        elif custom_downsample > 1:
            self.target_patch_size = (self.patch_size // custom_downsample,) * 2
        else:
            self.target_patch_size = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # with h5py.File(self.file_path, 'r') as hdf5_file:
        
        coord = self.hdf5_file['coords'][idx]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        
        if self.trans is None:
            img = torchvision.transforms.functional.pil_to_tensor(img).unsqueeze(0)
        else:
            img = self.trans(img)

        return img


def load_model(model_config: str, model_path: str, device: torch.device) -> Any:
    """
    Load the model with the given configuration and path.

    Args:
        model_config (str): Configuration name of the model.
        model_path (str): Path to the model file.
        device (torch.device): Device to load the model on.

    Returns:
        Any: The loaded model.
    """
    model = create_model(model_config, vocab_size=64010).eval()
    butils.load_model_and_may_interpolate(model_path, model, 'model|module', '')
    model.to(device, dtype=torch.float16)
    model.eval()
    return model


def process_slide(slide: str, h5_file: str, transform: torchvision.transforms.Compose, model: Any, device: torch.device, output_root: str) -> None:
    """
    Process a single slide, extracting features and saving them.

    Args:
        slide (str): The path to the slide file.
        transform (torchvision.transforms.Compose): Transformations to apply to the images.
        model (Any): The model to use for feature extraction.
        device (torch.device): Device to run the model on.
        output_root (str): Directory to save the output features.
    """
    slide_name = os.path.basename(slide).replace('.svs', '')   # or tiff format
    save_dir = os.path.join(output_root, f"{slide_name}.pt")

    if os.path.exists(save_dir):
        logging.info(f"Skipping {slide_name}, already processed.")
        return

    dataset = WSIBag(wsi_path=slide, file_path=h5_file, transform=transform, target_patch_size=img_size)
    loader = DataLoader(dataset, batch_size=200, shuffle=False, pin_memory=True, num_workers=4, drop_last=False) # Resize batch_size according to your GPU memory
    
    features = []

    with torch.inference_mode():
        for batch in loader:
            feat = model(
                image=batch.to(device, dtype=torch.float16),
                return_global=True,
                with_head=False,
                out_norm=True
            )[0]
            features.append(feat.cpu())

    features = torch.cat(features)
    torch.save(features, save_dir)
    logging.info(f"Saved features for {slide_name} to {save_dir}")


if __name__ == "__main__":
    process_idx = 1
    device = torch.device(f"cuda:{process_idx}")
    output_root = "./outputs/images"
    os.makedirs(output_root, exist_ok=True)

    # >>>>>>>>>>>> load model >>>>>>>>>>>> #
    model_config = "musk_large_patch16_384"
    model = create_model(model_config).eval()
    utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, 'model|module', '')
    model.to(device, dtype=torch.float16)
    model.eval()
    # <<<<<<<<<<<< load model <<<<<<<<<<<< #

    img_size = 384 if '384' in model_config else 224
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size, interpolation=3, antialias=True),
        torchvision.transforms.CenterCrop((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
    ])
    
    # path to the directory containing H5 files
    h5_root = "../1_process_wsi/outputs/patches"  # path to the directory containing H5 files
    h5_files =  list(glob.glob(f"{h5_root}/*.h5"))
    h5_dict = dict()

    for h5_file in h5_files:
        k = os.path.basename(h5_file)[:-len(".h5")]
        v = h5_file
        h5_dict.update({k: v})

    # Define the directory to search for H5 files
    slide_root = '../1_process_wsi/WSIs_source' 
    
    # List to store the paths of H5 files
    svs_files = []

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(slide_root):
        for file in files:
            if file.endswith('.svs'): # Also supports other formats files such as .tiff
                svs_files.append(os.path.join(root, file))

    svs_dict = dict()

    for svs_file in svs_files:
        k = os.path.basename(svs_file)[:-len(".svs")]
        v = svs_file
        svs_dict.update({k: v})

    slides_all = sorted(list(svs_dict.keys()), reverse=False)

    for slide in tqdm(slides_all):
        try: 
            slide_path = svs_dict[slide]
            h5_path = h5_dict[slide]
        except:
            continue

        process_slide(slide_path, h5_path, transform, model, device, output_root)
