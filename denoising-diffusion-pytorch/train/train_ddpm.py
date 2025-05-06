import sys
sys.path.append('denoising-diffusion-pytorch')

import argparse
import yaml
from denoising_diffusion import Unet, DenoisingDiffusion, Trainer

# ─── Load config ─────────────────────────────────────────────────────────

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml',
                    help='Path to YAML config file')
args = parser.parse_args()
cfg = load_config(args.config)

# ─── Unet Setup ────────────────────────────────────────────────────────────

unet_cfg = cfg['unet']
unet = Unet(
    dim       = unet_cfg['dim'],
    dim_mults = tuple(unet_cfg["dim_mults"]),
    dropout   = unet_cfg['dropout'],
)

# ─── DenoisingDiffusion Setup ───────────────────────────────────────────────

diffusion_cfg = cfg['diffusion']
diffusion = DenoisingDiffusion(
    model              = unet,
    image_size         = diffusion_cfg['image_size'],
    timesteps          = diffusion_cfg['timesteps'],           # number of steps
    sampling_timesteps = diffusion_cfg['sampling_timesteps'],         
)

# ─── Trainer Setup ─────────────────────────────────────────────────────────

trainer_cfg = cfg['trainer']
trainer = Trainer(
    diffusion_model       = diffusion,
    folder                = trainer_cfg['data_path'],
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
    print()
    trainer.train()  