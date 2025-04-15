# VAE
import sys

sys.path.append('/home/user1809/Desktop/diffusion-models/latent-diffusion')
from ldm.models.autoencoder import VQModel  
import torch 

# 1. Define your model configuration (should be the same as used for training)
ddconfig = {
    "double_z": False,
    "z_channels": 3,
    "resolution": 32,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 64,
    "ch_mult": [1, 2, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [],
    "dropout": 0.0
}

lossconfig = {
    "target": "taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
    "params": {
        "disc_conditional": False,
        "disc_in_channels": 3,
        "disc_start": 0,
        "disc_weight": 0.75,
        "codebook_weight": 1.0,
    }
}

embed_dim = 3
n_embed = 8192
base_learning_rate = 4.5e-06

# 2. Instantiate your model (import your VQModel accordingly)
vae = VQModel(
    ddconfig=ddconfig,
    lossconfig=lossconfig,
    n_embed=n_embed,
    embed_dim=embed_dim,
    monitor="val/rec_loss"
)
vae.learning_rate = base_learning_rate

# 3. Load the checkpoint
checkpoint_path = "/home/user1809/Desktop/vae_checkpoint/model_epochepoch=9999.ckpt"  # update with your checkpoint path
checkpoint = torch.load(checkpoint_path, map_location="cpu")
# For Lightning checkpoints you might have "state_dict" key
if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

vae.load_state_dict(state_dict)
vae.eval()  # set to evaluation mode

# # Move tensor to same device as model if necessary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)

sys.path.append('/home/user1809/Desktop/diffusion-models/denoising-diffusion-pytorch')
from denoising_diffusion_pytorch.utils import *
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# Path to your edges2shoes dataset root folder.
dataset_root = '../data/cifar_small/train_images'
image_size = 32  # You can change this based on your needs

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    dropout = 0.1,
)

from ldm.models.latent_diffusion import LatentDiffusion  

shape = vae.decoder.z_shape
latent_shape = (shape[1], shape[2], shape[3])

diffusion = LatentDiffusion(
    model,
    vae,
    # image_size = image_size,
    latent_shape = latent_shape,
    timesteps = 1000,           # number of steps
)


trainer = Trainer(
    diffusion,
    dataset_root,
    image_size=image_size,
    train_batch_size = 64,
    train_lr = 2e-4,
    train_num_steps = 1000000,           
    # calculate_fid = True,              
    save_and_sample_every = 2500,
    num_fid_samples = 1000,         
    results_folder='results/ldm/cifar_10_overfit',
    calculate_is=False,
    calculate_fid=False        
)

if __name__ == '__main__':
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print()
    trainer.train()  