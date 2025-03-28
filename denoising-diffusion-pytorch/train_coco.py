import os
from pathlib import Path
import json
from datetime import datetime
from random import random
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, utils
from PIL import Image
from tqdm.auto import tqdm
from denoising_diffusion_pytorch.utils import *
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_text_conditional import Unet, GaussianDiffusion, Trainer


#########################
# Dataset for edges2shoes
#########################

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


import clip

class TextConditionalDataset(Dataset):
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

        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.model_clip, self.preprocess = clip.load("ViT-B/32", device=self.device) 

    def get_text_embedding(self, text):
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_emb = self.model_clip.encode_text(tokens)
        return text_emb.cpu()  # shape: (1, 512) for ViT-B/32

    def __len__(self):
        return len(self.cond_paths)
    
    def __getitem__(self, index):
        # Get condition file path
        cond_path = self.cond_paths[index]
        # Extract numeric prefix to match with target image.
        # Assuming condition image names like "99_A.png" and target "99_B.png"
        prefix = cond_path.stem.split(".")[0]
        target_path = self.target_folder / f"{prefix}.jpg"
        
        # Load images
        with open(cond_path, 'r') as file:
            cond_text = file.read()
        cond = self.get_text_embedding(cond_text)
        target_img = Image.open(target_path).convert("RGB")
        target = self.transform(target_img)
        return target, cond

#########################
# Custom Trainer for Conditional Training
#########################


class ConditionalTrainer(Trainer):
    """
    Extend the Trainer to work with a DataLoader that returns (target, cond) tuples.
    """
    def __init__(self, diffusion, dataset, **kwargs):
        # We pass a dummy folder to the base Trainer (it won't be used)
        super().__init__(diffusion, folder='../data/coco/train', **kwargs)
        dl = DataLoader(dataset, batch_size =  self.batch_size, shuffle = True, pin_memory = True)
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

#########################
# Hyperparameters and Setup
#########################

# Path to your edges2shoes dataset root folder.
dataset_root = '../data/coco/train'
image_size = 64  # You can change this based on your needs

# Create dataset and dataloader.
dataset = TextConditionalDataset(dataset_root, image_size=image_size)
# train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

# Instantiate the conditional Unet.
# Make sure to set cond_channels=3 for an RGB conditioning image.
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    dropout = 0.1,
    channels = 3,
    self_condition = False, # disable self-conditioning when using external conditioning
)

# Create the GaussianDiffusion wrapper.
diffusion = GaussianDiffusion(
    model,
    image_size = image_size,
    timesteps = 1000,           # number of diffusion steps
    condition_text_folder=dataset_root+'/condition'
)

# Instantiate the trainer.
trainer = ConditionalTrainer(
    diffusion,
    dataset,
    train_batch_size = 16,
    train_lr = 2e-4,
    train_num_steps = 800000,
    calculate_fid = True,
    save_and_sample_every = 10000,
    num_fid_samples = 1000,
    results_folder = f'./results/conditional_ddpm/coco/28-03-2025_{image_size}x{image_size}'
)

#########################
# Training Script
#########################

if __name__ == '__main__':
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print("Starting training on edges2shoes conditional dataset...")
    trainer.train()
