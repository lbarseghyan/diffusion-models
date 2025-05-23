import sys
sys.path.append('denoising-diffusion-pytorch')

import argparse
import yaml

import torch.nn.functional as F
from torch.utils.data import DataLoader
from denoising_diffusion.utils import *
from denoising_diffusion.denoising_diffusion_text_conditional import Unet, TextConditionalDenoisingDiffusion, TextConditionalTrainer
from utils.data import TextConditionalDataset


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
dataset = TextConditionalDataset(dataset_cfg['dataset_root'], 
                                 image_size = dataset_cfg['image_size'])

# ─── Unet Setup ────────────────────────────────────────────────────────────

unet_cfg = cfg['unet']
unet = Unet(
    dim            = unet_cfg['dim'],
    dim_mults      = tuple(unet_cfg["dim_mults"]),
    dropout        = unet_cfg['dropout'],
    channels       = unet_cfg['channels'],
    self_condition = unet_cfg['self_condition'],    
    text_condition = unet_cfg['text_condition'],
    use_cross_attn = unet_cfg['use_cross_attn']
)

# ─── DenoisingDiffusion Setup ───────────────────────────────────────────────

diffusion_cfg = cfg['diffusion']
diffusion = TextConditionalDenoisingDiffusion(
    model                 = unet,
    image_size            = diffusion_cfg['image_size'],
    timesteps             = diffusion_cfg['timesteps'], 
    sampling_timesteps    = diffusion_cfg['sampling_timesteps'],         
    embedding_file        = dataset.embedding_file
)

# --- Trainer Setup ----------------------------------------------

# class ConditionalTrainer(Trainer):
#     """
#     Extend the Trainer to work with a DataLoader that returns (target, cond) tuples.
#     """
#     def __init__(self, diffusion_model, dataset, **kwargs):
#         # We pass a dummy folder to the base Trainer (it won't be used)
#         super().__init__(diffusion_model, folder=dataset_cfg['dataset_root'], **kwargs)
#         dl      = DataLoader(dataset, batch_size =  self.batch_size, shuffle = True, pin_memory = True)
#         dl      = self.accelerator.prepare(dl)
#         self.dl = cycle(dl)


trainer_cfg = cfg['trainer']
trainer = TextConditionalTrainer(
    diffusion_model       = diffusion,
    dataset               = dataset,
    train_batch_size      = trainer_cfg['train_batch_size'],
    train_lr              = float(trainer_cfg['train_lr']),
    train_num_steps       = trainer_cfg['train_num_steps'],
    calculate_fid         = trainer_cfg['calculate_fid'],
    calculate_is          = trainer_cfg['calculate_is'],
    save_and_sample_every = trainer_cfg['save_and_sample_every'],
    num_fid_samples       = trainer_cfg['num_fid_samples'],
    results_folder        = trainer_cfg['results_folder'],
)


if __name__ == '__main__':
    print(f"Total parameters: {sum(p.numel() for p in unet.parameters())}")
    print("Starting training on edges2shoes conditional dataset...")
    trainer.train()
