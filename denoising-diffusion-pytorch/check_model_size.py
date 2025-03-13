from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os
import csv
import numpy as np
from tqdm import tqdm
from datetime import datetime

results_folder = 'results/800k_steps_fid_samples_false_07_03_25'

model = Unet(
    dim = 128,
    dim_mults = (1, 2, 2, 2),
    dropout = 0.1,
    # flash_attn = True        default=False
)


diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    timesteps = 1000,           # number of steps
    hybrid_loss = True
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")