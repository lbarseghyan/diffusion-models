import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T

import sys
sys.path.append('denoising-diffusion-pytorch')

from denoising_diffusion.utils import *

# Custom Dataset for Image Folder
class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.images = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        image = Image.open(path).convert("RGB")
        if self.transform is None:
            np_img = np.array(image)  # shape [H, W, C]
            tensor_img = torch.tensor(np_img).float() / 255.0
            return {"image": tensor_img}
        else:
            img = self.transform(image)
            if img.dim() == 3:
                # Permute tensor from CHW to HWC if needed
                img = img.permute(1, 2, 0)
            return {"image": img}

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)        

class ImageConditionalDataset(Dataset):
    """
    This dataset expects a folder structure:
        root/
            condition/  --> edge maps (e.g., "99_A.png")
            target/     --> corresponding shoe images (e.g., "99_B.png")
    The pairing is done based on the numeric prefix (e.g., "99").
    """
    def __init__(self, 
                 root, 
                 image_size,
                 augment_horizontal_flip=False,
                 convert_image_to=None):
        self.root = Path(root)
        self.image_size = image_size
        
        # Get list of condition images
        self.cond_folder = self.root / "condition"
        self.target_folder = self.root / "target"
        self.cond_paths = sorted(list(self.cond_folder.glob("*.*")))
        
        # Define transform:
        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.cond_paths)
    
    def __getitem__(self, index):
        # Get condition file path
        cond_path = self.cond_paths[index]
        # Extract numeric prefix to match with target image.
        # Assuming condition image names like "99_A.png" and target "99_B.png"
        prefix = cond_path.stem.split("_")[0]
        target_path = self.target_folder / f"{prefix}_B.jpg"
        
        # Load images
        cond_img = Image.open(cond_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")
        
        cond = self.transform(cond_img)
        target = self.transform(target_img)
        return target, cond