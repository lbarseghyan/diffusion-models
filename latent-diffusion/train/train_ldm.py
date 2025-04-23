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



# ─── Import and Instantiate Your VQModel ─────────────────────────────────────────────────────────

sys.path.append('/home/user1809/Desktop/diffusion-models/latent-diffusion')
from ldm.models.autoencoder import VQModel  

vae = VQModel(
    ddconfig   = cfg['ddconfig'],
    lossconfig = cfg['lossconfig'],
    n_embed    = cfg['n_embed'],
    embed_dim  = cfg['embed_dim'],
    monitor    = "val/rec_loss"
)

vae.learning_rate = cfg['base_learning_rate']

checkpoint = torch.load(cfg['checkpoint_path'], map_location="cpu")

if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

vae.load_state_dict(state_dict)
vae.eval()  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)



# ─── Unet Setup ─────────────────────────────────────────────────────────

sys.path.append('/home/user1809/Desktop/diffusion-models/denoising-diffusion-pytorch')
from denoising_diffusion_pytorch.utils import *
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

unet_cfg = cfg['Unet_config']

model = Unet(
    dim       = unet_cfg['dim'],
    dim_mults = tuple(unet_cfg['dim_mults']),
    dropout   = unet_cfg['dropout'],
)



# ─── Diffusion Setup ─────────────────────────────────────────────────────────

from ldm.models.latent_diffusion import LatentDiffusion  

shape = vae.decoder.z_shape
latent_shape = (shape[1], shape[2], shape[3])

diffusion = LatentDiffusion(
    model,
    vae,
    latent_shape,
    timesteps = cfg['diffusion_timesteps'],          
)


# ─── Trainer Setup ─────────────────────────────────────────────────────────

trainer_cfg = cfg['trainer_config']
print('\n ++++++++++++++')
print(trainer_cfg)
print('\n ++++++++++++++')

trainer = Trainer(
    diffusion,
    folder                = trainer_cfg['dataset_root'],
    image_size            = trainer_cfg['image_size'],
    train_batch_size      = trainer_cfg['train_batch_size'],
    train_lr              = float(trainer_cfg['train_lr']),
    train_num_steps       = trainer_cfg['train_num_steps'],           
    save_and_sample_every = trainer_cfg['save_and_sample_every'],
    num_fid_samples       = trainer_cfg['num_fid_samples'],         
    results_folder        = trainer_cfg['results_folder'],
    calculate_is          = trainer_cfg['calculate_is'],
    calculate_fid         = trainer_cfg['calculate_fid']        
)

# ─── Main Training Routine ──────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print()
    trainer.train()  