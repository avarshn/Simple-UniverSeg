"""
Code is taken from https://github.com/JJGO/UniverSeg/blob/main/example_data/oasis.py
OASIS dataset processed at https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md
"""

import pathlib
import subprocess
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import nibabel as nib
import PIL
from PIL import ImageEnhance

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms

import random
from utils.const import DATA_FOLDER
from utils.elastic_deformation import get_displacement

def process_img(path: pathlib.Path, size: Tuple[int, int]):
    img = (nib.load(path).get_fdata() * 255).astype(np.uint8).squeeze()
    img = PIL.Image.fromarray(img)
    img = img.resize(size, resample=PIL.Image.BILINEAR)
    img = img.convert("L")
    img = np.array(img)
    img = img.astype(np.float32) / 255
    img = np.rot90(img, -1)
    return img.copy()


def process_seg(path: pathlib.Path, size: Tuple[int, int]):
    seg = nib.load(path).get_fdata().astype(np.int8).squeeze()
    seg = PIL.Image.fromarray(seg)
    seg = seg.resize(size, resample=PIL.Image.NEAREST)
    seg = np.array(seg)
    seg = seg.astype(np.float32)
    seg = np.rot90(seg, -1)
    return seg.copy()

def intask_augmentation(img, seg):


    # Spatial Transforms (Affine Shift, Elastic Warp, Flip) - Should be Applied both on image and segmentation label
    
    # Affine Shift
    if np.random.rand() < 0.5:
        degree = random.randint(-15, 15)
        translate_x = random.uniform(0, 0.2)
        translate_y = random.uniform(0, 0.2)
        scale = random.uniform(0.8, 1.1)
        shear = 0
        img = transforms.functional.affine(img, degree, [translate_x, translate_y], scale, shear)
        seg = transforms.functional.affine(seg, degree, [translate_x, translate_y], scale, shear)
        

    # Elastic Warp
    # if np.random.rand() < 0.1:
    #     alpha = random.uniform(1, 2.5)
    #     sigma = random.uniform(7, 8)
    #     _, height, width = transforms.functional.get_dimensions(img)
    #     displacement = get_displacement(alpha, sigma, [height, width])
    #     img = transforms.functional.elastic(img, displacement, transforms.InterpolationMode.BILINEAR)
    #     seg = transforms.functional.elastic(seg, displacement, transforms.InterpolationMode.NEAREST)

    
    # Brightness Contrast Change
    if np.random.rand() < 0.25:
        # Adjust brightness and contrast
        jitter = transforms.ColorJitter(brightness=0.1, contrast=0.5)
        img = jitter(img)
        
    # Sharpness Change
    if np.random.rand() < 0.25:
        sharpness_factor = random.uniform(0, 5)
        # Adjust sharpness
        img = transforms.functional.adjust_sharpness(img, sharpness_factor)

    # Gaussian Blur
    if np.random.rand() < 0.25:
        sigma_val = random.uniform(0.1, 1.1)
        img = transforms.functional.gaussian_blur(img, kernel_size = 5, sigma = sigma_val)

    # Gaussian Noise
    if np.random.rand() < 0.25:
        mean = random.uniform(0, 0.005)
        std = random.uniform(0, 0.06)  # std_square = [0, 0.0036]

        # noise = std * torch.randn_like(img) + mean
        img = img + std * torch.randn_like(img) + mean
    
    return img, seg
    
def load_folder(path: pathlib.Path, size: Tuple[int, int] = (128, 128)):
    data = []
    for file in sorted(path.glob("*/slice_norm.nii.gz")):
        img = process_img(file, size=size)
        seg_file = pathlib.Path(str(file).replace("slice_norm", "slice_seg24"))
        seg = process_seg(seg_file, size=size)
        data.append((img, seg))
    return data


def require_download_oasis():
    dest_folder = pathlib.Path("/tmp/universeg_oasis/")

    if not dest_folder.exists():
        tar_url = "https://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.2d.v1.0.tar"
        subprocess.run(
            [
                "curl",
                tar_url,
                "--create-dirs",
                "-o",
                str(dest_folder / "neurite-oasis.2d.v1.0.tar"),
            ],
            stderr=subprocess.DEVNULL,
            check=True,
        )

        subprocess.run(
            [
                "tar",
                "xf",
                str(dest_folder / "neurite-oasis.2d.v1.0.tar"),
                "-C",
                str(dest_folder),
            ],
            stderr=subprocess.DEVNULL,
            check=True,
        )

    return dest_folder


@dataclass
class OASISDataset(Dataset):
    split: Literal["support", "test"]
    label: int
    support_frac: float = 0.154

    def __post_init__(self):
        T = torch.from_numpy
        self._data = [(T(x)[None], T(y)) for x, y in load_folder(DATA_FOLDER)]
        if self.label is not None:
            self._ilabel = self.label
        self._idxs = self._split_indexes()

    def _split_indexes(self):
        rng = np.random.default_rng(42)
        N = len(self._data)
        p = rng.permutation(N)
        i = int(np.floor(self.support_frac * N))
        return {"support": p[:i], "test": p[i:]}[self.split]

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, idx):
        img, seg = self._data[self._idxs[idx]]
        if self.label is not None:
            seg = (seg == self._ilabel)[None]
        return img, seg


@dataclass
class My_OASISDataset(Dataset):
    split: Literal["support", "dev", "test"]
    label: int

    def __init__(self, split, label, train_frac=0.6, dev_frac=0.2, do_in_task_augmentation = False):
        self.split = split
        self.label = label
        self.train_frac = train_frac
        self.dev_frac = dev_frac
        self.do_in_task_augmentation = do_in_task_augmentation
        # self._idxs = None

        # Convert Numpy array to Pytorch tensor
        T = torch.from_numpy
        self._data = [(T(x)[None], T(y)) for x, y in load_folder(DATA_FOLDER)]
        if self.label is not None:
            # print("label not none")
            self._ilabel = self.label
        self._idxs = self._split_indexes()

    # def __post_init__(self):
    #     T = torch.from_numpy
    #     self._data = [(T(x)[None], T(y)) for x, y in load_folder(DATA_FOLDER)]
    #     if self.label is not None:
    #         print("label not none")
    #         self._ilabel = self.label
    #     self._idxs = self._split_indexes()

    def _split_indexes(self):
        # print("in split index")
        rng = np.random.default_rng(42)
        N = len(self._data)
        p = rng.permutation(N)
        i = int(np.floor(self.train_frac * N))
        j = int(np.floor(self.dev_frac * N))
        return {"support": p[:i], "dev": p[i : i + j], "test": p[i + j :]}[self.split]

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, idx):
        img, seg = self._data[self._idxs[idx]]
        if self.label is not None:
            seg = (seg == self._ilabel)[None]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # # Perform Image preprocesing on GPU
        # img = img.to(device)
        # seg = seg.to(device)

        # In-task augmentation - For training
        if self.do_in_task_augmentation:
            img, seg = intask_augmentation(img, seg)

        return img, seg

def create_data_loaders_v2(labels_idx, batch_size, num_workers, split, do_in_task_augmentation = False):
    data_loaders = []
    datasets = []
    for task_idx in labels_idx:
        d_support = My_OASISDataset(split, task_idx, do_in_task_augmentation = do_in_task_augmentation)
        supportloader = DataLoader(
            d_support, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        data_loaders.append(supportloader)
        datasets.append(d_support)
    return data_loaders, datasets