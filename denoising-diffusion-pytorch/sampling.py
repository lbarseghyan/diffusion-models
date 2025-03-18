import argparse
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import cpu_count
from denoising_diffusion_pytorch.utils import *
from pathlib import Path
import torch
from torchvision import transforms as T, utils
from datetime import datetime
from functools import partial
from tqdm.auto import tqdm
import os
import re

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    dropout = 0.1,
)


diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    timesteps = 1000,           # number of steps
)

training_results_folder = 'results/14-03-2025_models'

trainer = Trainer(
    diffusion,
    '../data/cifar-10/train_images',
    train_batch_size = 64,
    train_lr = 2e-4,
    train_num_steps = 800000,           
    calculate_fid = True,              
    save_and_sample_every = 5,
    num_fid_samples = 10,    
    results_folder = training_results_folder        
)

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run DDIM sampling for diffusion model")
    parser.add_argument(
        "--ddim_sampling_timesteps",
        type=int,
        default=10,
        help="Number of timesteps for DDIM sampling (default: 200)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="If set, use trained model with the given number."
    )

    parser.add_argument(
        "--ddim_results_folder",
        type=str,
        default=None,
        help="Folder for DDIM results (default: auto-generated)"
    )

    parser.add_argument(
        "--ddim_sampling",
        action="store_false",
        help="If set, use DDIM sampling; otherwise, use DDPM sampling."
    )

    args = parser.parse_args()

    # Use the command-line argument for ddim_sampling_timesteps
    ddim_sampling_timesteps = args.ddim_sampling_timesteps      # 200
    ddim_results_folder = args.ddim_results_folder              # None
    ddim_sampling = args.ddim_sampling   # True
    sampling_model = args.model 

    # Find the model numbers
    if sampling_model is not None:
        milestones = [sampling_model]
    else:
        pattern = re.compile(r"model-(\d+)\.pt")
        milestones = []
        for filename in os.listdir(training_results_folder):
            match = pattern.fullmatch(filename)
            if match:
                milestones.append(int(match.group(1)))
        milestones.sort(reverse=True) 

    if ddim_results_folder is not None:
        ddim_results_folder = Path(ddim_results_folder)
        ddim_results_folder.mkdir(parents=True, exist_ok = True)
    else:
        # experiment_date = datetime.now().strftime("%d-%m-%Y")

        ddim_results_folder = Path("./results_ddim")  / f"{os.path.basename(training_results_folder)}_{ddim_sampling_timesteps}"

        counter = 1
        while ddim_results_folder.exists():
            ddim_results_folder = Path("./results_ddim") / f"{os.path.basename(training_results_folder)}_{ddim_sampling_timesteps}_{counter}"
            counter += 1
        ddim_results_folder.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(ddim_results_folder / "tensorboard_logs"))


    for milestone in milestones:   
        accelerator = trainer.accelerator

        trainer.load(milestone)

        trainer.ema.ema_model.eval()

        (h, w), channels = trainer.ema.ema_model.image_size, trainer.ema.ema_model.channels

        with torch.inference_mode():
            batches = num_to_groups(trainer.num_samples, trainer.batch_size)
            if ddim_sampling:
                all_images_list = list(map(lambda n: trainer.ema.ema_model.ddim_sample((n, channels, h, w), sampling_timesteps = ddim_sampling_timesteps, return_all_timesteps = False), batches))     # if ddim
            else:
                all_images_list = list(map(lambda n: trainer.ema.ema_model.sample(batch_size=n), batches))    

        all_images = torch.cat(all_images_list, dim = 0)

        utils.save_image(all_images, str(ddim_results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(trainer.num_samples)))

        # Log generated samples to TensorBoard
        # Assuming all_images is in [C, H, W] format per sample (batched as [B, C, H, W])
        writer.add_images("Samples", all_images, trainer.step, dataformats="NCHW")

        # Generate num_fid_samples 
        accelerator.print(f"Generating {trainer.fid_scorer.n_samples} sample for calculating FID and IS.")
        all_fake_samples_list = [] 
        with torch.inference_mode():
            batches = num_to_groups(trainer.fid_scorer.n_samples, trainer.batch_size)
            for batch in tqdm(batches):
                # Generate a batch of images.
                if ddim_sampling:
                    fake_samples =  trainer.ema.ema_model.ddim_sample((batch, channels, h, w), sampling_timesteps = ddim_sampling_timesteps, return_all_timesteps = False).to(trainer.device)     # if ddim
                else:
                    fake_samples =  trainer.ema.ema_model.sample(batch_size=batch).to(trainer.device)    

                all_fake_samples_list.append(fake_samples)

        all_fake_samples = torch.cat(all_fake_samples_list, dim=0)

        # whether to calculate fid
        if trainer.calculate_fid:
            fid_score = trainer.fid_scorer.fid_score(all_fake_samples)
            accelerator.print(f'FID score: {fid_score}')
            writer.add_scalar("Eval/FID", fid_score, trainer.step)

        # whether to calculate IS
        if trainer.calculate_is:
            is_score = trainer.is_scorer.calculate_inception_score(all_fake_samples)
            trainer.accelerator.print(f'Inception Score: {is_score:.4f}')
            writer.add_scalar("Eval/IS", is_score, trainer.step)
