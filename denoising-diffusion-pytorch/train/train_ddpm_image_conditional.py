import sys
sys.path.append('denoising-diffusion-pytorch')

import argparse
import yaml
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from denoising_diffusion.denoising_diffusion_image_conditional import Unet, ImageConditionalDenoisingDiffusion, ImageConditionalTrainer
from denoising_diffusion.utils import *
from utils.data import ImageConditionalDataset


# ─── Load config ─────────────────────────────────────────────────────────

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml',
                    help='Path to YAML config file')
args = parser.parse_args()
cfg = load_config(args.config)


# ─── Dataset Setup ────────────────────────────────────────────────────────────

dataset_cfg = cfg['dataset']
dataset = ImageConditionalDataset(dataset_cfg['dataset_root'], 
                                  image_size = dataset_cfg['image_size'])


# ─── Unet Setup ────────────────────────────────────────────────────────────

unet_cfg = cfg['unet']
unet = Unet(
    dim            = unet_cfg['dim'],
    dim_mults      = tuple(unet_cfg["dim_mults"]),
    dropout        = unet_cfg['dropout'],
    channels       = unet_cfg['channels'],
    cond_channels  = unet_cfg['cond_channels'],      
    self_condition = unet_cfg['self_condition'],    
)

# ─── DenoisingDiffusion Setup ───────────────────────────────────────────────

diffusion_cfg = cfg['diffusion']
diffusion = ImageConditionalDenoisingDiffusion(
    model                 = unet,
    image_size            = diffusion_cfg['image_size'],
    timesteps             = diffusion_cfg['timesteps'],        
    sampling_timesteps    = diffusion_cfg['sampling_timesteps'],           
    condition_data_folder = dataset_cfg['dataset_root']+'/condition'
)

# ─── Trainer Setup ─────────────────────────────────────────────────────────

trainer_cfg = cfg['trainer']
trainer = ImageConditionalTrainer(
    diffusion_model       = diffusion,
    dataset               = dataset,
    train_batch_size      = trainer_cfg['train_batch_size'],
    train_lr              = float(trainer_cfg['train_lr']),
    train_num_steps       = trainer_cfg['train_num_steps'],
    calculate_fid         = trainer_cfg['calculate_fid'],
    calculate_is          = trainer_cfg['calculate_is'],
    save_and_sample_every = trainer_cfg['save_and_sample_every'],
    num_fid_samples       = trainer_cfg['num_fid_samples'],
    results_folder        = trainer_cfg['results_folder']            
)


if __name__ == '__main__':
    print(f"Total parameters: {sum(p.numel() for p in unet.parameters())}")
    print("Starting training on edges2shoes conditional dataset...")
    trainer.train()