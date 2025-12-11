from torch.utils.data import Dataset
import numpy as np
import openslide


class PatchBag(Dataset):
    def __init__(self, slide, coords, patch_level, patch_size, target_size):
        self.slide = slide
        self.coords = coords
        self.patch_level = patch_level
        self.patch_size = patch_size
        self.target_size = (target_size,)*2

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, item):
        region = self.slide.read_region(self.coords[item], self.patch_level,
                                        (self.patch_size, self.patch_size)).convert("RGB")

        if self.target_size is not None:
            region = region.resize(self.target_size)

        region = np.array(region) / 255
        region = np.transpose(region, (2, 0, 1))

        return region


