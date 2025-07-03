import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import yaml
from addict import Dict
from tifffile import TiffWriter
from pathlib import Path
import cv2
import os
from PIL import Image


def read_yaml(fpath="./configs/sample.yaml"):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)


def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]


# **********************************
# Author: kuantian                 #
# Email:  kuantian@tencent.com     #
# **********************************
class WsiWriter(object):
    def __init__(self, wsi_slide, mmap_tmp_dir, save_tiff_name, custom_level):
        self.wsi_slide = wsi_slide
        self.mmap_tmp_dir = mmap_tmp_dir
        self.tile_size = 256

        self.resolution_unit = None
        self.x_resolution = None
        self.y_resolution = None

        self.save_tiff_name = save_tiff_name
        self.custom_level = custom_level
        self.level_numpy_list = self.init_mmap()

    def init_mmap(self):
        [w, h] = self.wsi_slide.level_dimensions[self.custom_level]

        img_h, img_w = h, w
        level_numpy_list = []
        for level in range(self.wsi_slide.level_count - self.custom_level):
            memmap_filename = Path(self.mmap_tmp_dir, "level{}.mmap".format(level))
            level_numpy = np.memmap(
                str(memmap_filename), dtype=np.uint8, mode="w+", shape=(img_h, img_w, 4)
            )
            level_numpy[:] = 255
            level_numpy_list.append(level_numpy)

            img_w = round(img_w / 2)
            img_h = round(img_h / 2)

            if max(img_w, img_h) < self.tile_size:
                break
        return level_numpy_list

    def write_level0(self, image):
        # image_rgb = image[:, :, :3]
        # self.level_numpy_list[0][:, :, :3] = image_rgb[:, :, ::-1]
        # self.level_numpy_list[0][:, :, 3] = image[:, :, 3]

        self.level_numpy_list[0] = image

    def generate_pyramid(self):
        for index in range(1, len(self.level_numpy_list)):
            src_arr = self.level_numpy_list[index - 1]
            target_arr = self.level_numpy_list[index]

            target_arr[:] = cv2.resize(
                src_arr, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
            )

    def generate_properties(self):
        properties = self.wsi_slide.properties
        if (properties.get("tiff.ResolutionUnit")
                and properties.get("tiff.XResolution")
                and properties.get("tiff.YResolution")):
            resolution_unit = properties.get("tiff.ResolutionUnit")
            x_resolution = float(properties.get("tiff.XResolution"))
            y_resolution = float(properties.get("tiff.YResolution"))
        else:
            resolution_unit = properties.get("tiff.ResolutionUnit", "inch")
            if properties.get("tiff.ResolutionUnit", "inch").lower() == "inch":
                numerator = 25400  # Microns in Inch
            else:
                numerator = 10000  # Microns in CM
            x_resolution = int(numerator // float(properties.get("openslide.mpp-x", 1)))
            y_resolution = int(numerator // float(properties.get("openslide.mpp-y", 1)))

        downsample = self.wsi_slide.level_downsamples[self.custom_level]
        x_resolution = x_resolution / downsample
        y_resolution = y_resolution / downsample
        print("x_resolution, y_resolution:", x_resolution, y_resolution)
        self.resolution_unit = resolution_unit
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution

    def write_tiff(self):

        print('generate_pyramid')
        self.generate_pyramid()
        print('generate_properties')
        self.generate_properties()
        subfiletype_none = 0
        subfiletype_reducedimage = 1

        with TiffWriter(self.save_tiff_name, bigtiff=True) as tif:
            for level in range(len(self.level_numpy_list)):  # save from smaller image
                src_arr = self.level_numpy_list[level]

                tif.save(
                    src_arr,
                    software="Glencoe/Faas pyramid",
                    metadata={"axes": "YXC"},
                    tile=(self.tile_size, self.tile_size),
                    photometric="RGB",
                    planarconfig="CONTIG",
                    resolution=(
                        self.x_resolution // 2 ** level,
                        self.y_resolution // 2 ** level,
                        self.resolution_unit,
                    ),
                    # compression=6,
                    subfiletype=subfiletype_reducedimage if level else subfiletype_none,
                )


if __name__ == "__main__":
    img = cv2.imread("../heatmap.jpg")
    import openslide

    wsi_name = "/mnt/group-ai-medical-sz/private/jinxixiang/data/PANDA" \
               "/train_images/07aa24f15ce062d65979b6a8bc7eb3f0.tiff"
    wsi_slide = openslide.OpenSlide(wsi_name)
    wsi_writer = WsiWriter(wsi_slide, "../mmap_dir", "heatmap.tiff")
    wsi_writer.write_patch(img)
    wsi_writer.write_tiff()
