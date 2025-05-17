import yaml
import sys
sys.path.append('./denoising-diffusion-pytorch')


import argparse
from denoising_diffusion import Unet, DenoisingDiffusion, Trainer, Dataset
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import cpu_count
from denoising_diffusion.utils import *
from pathlib import Path
import torch
from torchvision import transforms as T, utils
from tqdm.auto import tqdm
import os
import re
from ema_pytorch import EMA
from denoising_diffusion.inception_score_evaluation import InceptionScoreEvaluation
from denoising_diffusion.fid_evaluation import FIDEvaluation
from torch.utils.data import DataLoader


# ─── Load config ─────────────────────────────────────────────────────────

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
config = '/home/user1809/Desktop/diffusion-models/latent-diffusion/train/configs/ldm_cifar.yaml'
cfg = load_config(config)




# ─── Import and Instantiate Your VQModel ─────────────────────────────────────────────────────────
import sys
sys.path.append('./latent-diffusion')
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

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    dropout = 0.1,
)

# ─── Diffusion Setup ─────────────────────────────────────────────────────────

from ldm.models.latent_diffusion import LatentDiffusion  

shape = vae.decoder.z_shape
latent_shape = (shape[1], shape[2], shape[3])

diffusion = LatentDiffusion(
    model,
    vae,
    latent_shape,
    timesteps = 1000,          
)
# ------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ema_decay = 0.995
ema_update_every = 10
ema = EMA(diffusion, beta = ema_decay, update_every = ema_update_every)
ema.to(device)

num_samples = 25
batch_size = 64
inception_block_idx =2048  #from trainer
image_size = 32            #from trainer
# images_folder = '../data/cifar-10/train_images'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run DDIM sampling for diffusion model")

    parser.add_argument(
        "--trained_models_folder",
        type=str,
        default='./results',
        help="Folder for trained models"
    )

    parser.add_argument(
        "--images_folder",
        type=str,
        default='../data/cifar-10/train_images',
        help="Folder for real images"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="If set, use trained model with the given number."
    )

    parser.add_argument(
        "--generation_results_folder",
        type=str,
        default=None,
        help="Folder for DDIM results (default: auto-generated)"
    )

    # parser.add_argument(
    #     "--ddim_sampling",
    #     action="store_false",
    #     help="If set, use DDIM sampling; otherwise, use DDPM sampling."
    # )

    parser.add_argument(
        "--ddim_sampling_timesteps",
        type=int,
        default=10,
        help="Number of timesteps for DDIM sampling (default: 200)"
    )

    parser.add_argument(
        "--calculate_is",
        action="store_false",
        help="If set, calculate IS."
    )

    parser.add_argument(
        "--calculate_fid",
        action="store_false",
        help="If set, calculate FID."
    )

    parser.add_argument(
        "--num_fid_samples",
        type=int,
        default=1000,
        help="Number of images generated for FID evaluation"
    )

    args = parser.parse_args()

    # Use the command-line argument for ddim_sampling_timesteps
    trained_models_folder = args.trained_models_folder
    images_folder = args.images_folder
    ddim_sampling_timesteps = args.ddim_sampling_timesteps      # 200
    generation_results_folder = args.generation_results_folder              # None
    # ddim_sampling = args.ddim_sampling   # True
    ddim_sampling = False
    sampling_model = args.model 
    calculate_fid = args.calculate_fid 
    calculate_is = args.calculate_is 
    num_fid_samples = args.num_fid_samples 


    # Find the model numbers
    if sampling_model is not None:
        milestones = [sampling_model]
    else:
        pattern = re.compile(r"model-(\d+)\.pt")
        milestones = []
        for filename in os.listdir(trained_models_folder):
            match = pattern.fullmatch(filename)
            if match:
                milestones.append(int(match.group(1)))
        milestones.sort() 

    if generation_results_folder is not None:
        generation_results_folder = Path(generation_results_folder)
        generation_results_folder.mkdir(parents=True, exist_ok = True)
    else:
        generation_results_folder = Path("./results_ddim")  / f"{os.path.basename(trained_models_folder)}_{ddim_sampling_timesteps}"

        counter = 1
        while generation_results_folder.exists():
            generation_results_folder = Path("./results_ddim") / f"{os.path.basename(trained_models_folder)}_{ddim_sampling_timesteps}_{counter}"
            counter += 1
        generation_results_folder.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(generation_results_folder / "tensorboard_logs"))

    if calculate_fid:
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(diffusion.channels)
            augment_horizontal_flip = True  # from trainer
            shuffle = True
            ds = Dataset(images_folder, image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
            dl = DataLoader(ds, batch_size = batch_size, shuffle = shuffle, pin_memory = True, num_workers = cpu_count())
            dl = cycle(dl)

    for milestone in tqdm(milestones): 
        data = torch.load(str(Path(trained_models_folder) / f'model-{milestone}.pt'), map_location=device, weights_only=True)
        ema.load_state_dict(data["ema"])
        ema.ema_model.eval()
        step = data['step']

        (h, w), channels = ema.ema_model.image_size, ema.ema_model.channels

        with torch.inference_mode():
            batches = num_to_groups(num_samples, batch_size)
            if ddim_sampling:
                all_images_list = list(map(lambda n: ema.ema_model.ddim_sample((n, channels, h, w), sampling_timesteps = ddim_sampling_timesteps, return_all_timesteps = False), batches))     # if ddim
            else:
                all_images_list = list(map(lambda n: ema.ema_model.sample(batch_size=n), batches))    
           
        all_images = torch.cat(all_images_list, dim = 0)
        utils.save_image(all_images, str(generation_results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(num_samples)))


        # Log generated samples to TensorBoard
        # Assuming all_images is in [C, H, W] format per sample (batched as [B, C, H, W])
        writer.add_images("Samples", all_images, step, dataformats="NCHW")

        if calculate_fid or calculate_is:
            # Generate num_fid_samples 
            print(f"Generating {num_fid_samples} sample for calculating FID and IS.")
            all_fake_samples_list = [] 
            with torch.inference_mode():
                batches = num_to_groups(num_fid_samples, batch_size)
                for batch in tqdm(batches):
                    # Generate a batch of images.
                    if ddim_sampling:
                        fake_samples =  ema.ema_model.ddim_sample((batch, channels, h, w), sampling_timesteps = ddim_sampling_timesteps, return_all_timesteps = False).to(device)     # if ddim
                    else:
                        fake_samples =  ema.ema_model.sample(batch_size=batch).to(device)    

                    all_fake_samples_list.append(fake_samples)

            all_fake_samples = torch.cat(all_fake_samples_list, dim=0)

        # whether to calculate fid
        if calculate_fid:
            fid_scorer = FIDEvaluation(
                    batch_size=batch_size,
                    dl=dl,
                    sampler=ema.ema_model,
                    channels=diffusion.channels,
                    stats_dir=trained_models_folder,
                    device=device,
                    num_fid_samples=num_fid_samples,
                    inception_block_idx=inception_block_idx
                )

            fid_score = fid_scorer.fid_score(all_fake_samples)
            print(f'FID score: {fid_score}')
            writer.add_scalar("Eval/FID", fid_score, step)

        # whether to calculate IS
        if calculate_is:
            is_scorer = InceptionScoreEvaluation(
                batch_size=batch_size,
                sampler=ema.ema_model,
                channels=diffusion.channels,
                stats_dir=trained_models_folder,
                device=device,
                num_samples=num_fid_samples  # Use same count as FID for consistency
            )

            is_score = is_scorer.calculate_inception_score(all_fake_samples)
            print(f'Inception Score: {is_score:.4f}')
            writer.add_scalar("Eval/IS", is_score, step)