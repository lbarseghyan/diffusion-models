import sys
import torch 
import argparse
import yaml


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
from utils.data import ImageConditionalDataset

dataset_cfg = cfg['dataset']
dataset = ImageConditionalDataset(dataset_cfg['dataset_root'], 
                                  image_size = dataset_cfg['image_size'])

# ─── Import and Instantiate Your VQModels ─────────────────────────────────────────────────────────

sys.path.append('./latent-diffusion')
from ldm.models.autoencoder import VQModel  

condition_vae = VQModel(
    ddconfig   = cfg['ddconfig'],
    lossconfig = cfg['lossconfig'],
    n_embed    = cfg['n_embed'],
    embed_dim  = cfg['embed_dim'],
    monitor    = "val/rec_loss"
)

condition_vae.learning_rate = cfg['base_learning_rate']

condition_checkpoint = torch.load(cfg['condition_checkpoint_path'], map_location="cpu")

if "state_dict" in condition_checkpoint:
    state_dict = condition_checkpoint["state_dict"]
else:
    state_dict = condition_checkpoint

condition_vae.load_state_dict(state_dict)
condition_vae.eval()  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
condition_vae.to(device)

# Target VAE

target_vae = VQModel(
    ddconfig   = cfg['ddconfig'],
    lossconfig = cfg['lossconfig'],
    n_embed    = cfg['n_embed'],
    embed_dim  = cfg['embed_dim'],
    monitor    = "val/rec_loss"
)

target_vae.learning_rate = cfg['base_learning_rate']

target_checkpoint = torch.load(cfg['target_checkpoint_path'], map_location="cpu")

if "state_dict" in target_checkpoint:
    state_dict = target_checkpoint["state_dict"]
else:
    state_dict = target_checkpoint

target_vae.load_state_dict(state_dict)
target_vae.eval()  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_vae.to(device)


# ─── Unet Setup ─────────────────────────────────────────────────────────

sys.path.append('./denoising-diffusion-pytorch')
from denoising_diffusion.utils import *
from denoising_diffusion.denoising_diffusion_image_conditional import Unet, ImageConditionalTrainer

unet_cfg = cfg['unet']
unet = Unet(
    dim            = unet_cfg['dim'],
    dim_mults      = tuple(unet_cfg["dim_mults"]),
    dropout        = unet_cfg['dropout'],
    channels       = unet_cfg['channels'],
    cond_channels  = unet_cfg['cond_channels'],      
)

# ─── Diffusion Setup ─────────────────────────────────────────────────────────

from ldm.models.latent_diffusion_image_conditional import ImageConditionalLatentDiffusion  

shape = condition_vae.decoder.z_shape
latent_shape = (shape[1], shape[2], shape[3])

latentdiffusion_cfg=cfg['latentdiffusion']
diffusion = ImageConditionalLatentDiffusion(
    unet,
    vae          = target_vae,
    latent_shape = latent_shape,
    cond_vae     = condition_vae,
    timesteps    = latentdiffusion_cfg['timesteps'],
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




# ─── Main Training Routine ──────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"Total parameters: {sum(p.numel() for p in unet.parameters())}")
    print()
    trainer.train()  