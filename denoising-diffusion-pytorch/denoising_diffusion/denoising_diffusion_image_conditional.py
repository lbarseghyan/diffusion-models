from pathlib import Path
import random
import math
from functools import partial
from collections import namedtuple
from typing import Optional, Tuple
from einops import rearrange, reduce
from tqdm.auto import tqdm


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T, utils
from PIL import Image

from denoising_diffusion.denoising_diffusion import Unet, DenoisingDiffusion, Trainer, extract
from denoising_diffusion.utils import (
    exists, default, cycle, unnormalize_to_zero_to_one, num_to_groups, identity, divisible_by)

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# --------------------------------------------------------------------------
# 1.  U‑Net that understands an *extra* image‑condition tensor
# --------------------------------------------------------------------------

class Unet(Unet):
    """U‑Net that accepts an additional *image* condition.

     Parameters
     ----------
     *unet_args, **unet_kwargs
     cond_channels : int, optional
         Number of channels in the conditioning image (0 disables
         conditioning).  For RGB images this is *3*.
    """

    def __init__(self, *unet_args, cond_channels = 0, **unet_kwargs):
        self.cond_channels = cond_channels
        super().__init__(*unet_args, **unet_kwargs)

        in_ch = self.init_conv.in_channels + cond_channels
        out_ch, k, p = self.init_conv.out_channels, self.init_conv.kernel_size, self.init_conv.padding
        self.init_conv = nn.Conv2d(in_ch, out_ch, k, padding=p)

    # Override the forward pass only to concatenate the conditioning map.
    def forward(self, x, time, *, cond = None, x_self_cond = None):
        if exists(cond):
            assert cond.shape[0] == x.shape[0], "batch mismatch between x and cond"
            x = torch.cat((x, cond), dim=1)
        return super().forward(x, time, x_self_cond=x_self_cond)


# --------------------------------------------------------------------------
# 2.  Diffusion process that forwards the condition through the Unet
# --------------------------------------------------------------------------

class ImageConditionalDenoisingDiffusion(DenoisingDiffusion):
    """DenoisingDiffusion wrapper that is aware of an image condition.
     It simply pass *cond* down to the underlying model everywhere it matters.

     Parameters
     ----------
     condition_data_folder : str or Path, optional
         Path to a directory with images that can serve as conditioning inputs
         during sampling.

     All other parameters are forwarded to the parent constructor unchanged.
     """
    def __init__(self, *args, condition_data_folder, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition_data_folder = condition_data_folder

    def model_predictions(self, x, t, cond=None, x_self_cond = None,clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, cond=cond, x_self_cond=x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)
    
    # propagate cond through the private helper that compute mean/var
    def p_mean_variance(self, x, t, cond = None, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, cond=cond, x_self_cond=x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, cond = None, x_self_cond = None):   # change
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, cond = cond, x_self_cond = x_self_cond, clip_denoised = True)   # change
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    def get_random_condition(self, batch, device):
        """
        Randomly sample a batch of condition images from the training condition folder.
        Assumes that self.training_condition_data_folder is set to the folder path.
        """
        # Define a default transform if not already defined
        if not hasattr(self, "cond_transform"):
            self.cond_transform = T.Compose([
                T.Lambda(nn.Identity()),
                T.Resize(self.image_size),
                nn.Identity(),
                T.CenterCrop(self.image_size),
                T.ToTensor()
            ])
            
        # List all images in the training condition folder and randomly choose 'batch' number of images
        condition_folder = Path(self.condition_data_folder)
        condition_paths = list(condition_folder.glob("*.*"))
        selected_paths = random.choices(condition_paths, k = batch)

        cond_images = []
        for p in selected_paths:
            img = Image.open(p).convert("RGB")
            img = self.cond_transform(img)
            cond_images.append(img)

        cond_batch = torch.stack(cond_images, dim=0).to(device)

        # cond_img = Image.open(selected_path).convert("RGB")
        # cond_img = self.cond_transform(cond_img)
        return cond_batch

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_condition_image=False, return_all_timesteps = False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        # Sample a batch of condition images
        cond = self.get_random_condition(batch, device)     # add

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, cond, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)

        if return_condition_image:
            return (cond, ret)
        
        return ret
    
    @torch.inference_mode()
    def ddim_sample(self, shape, sampling_timesteps=None, cond = None, return_all_timesteps = False):
        if sampling_timesteps is None:
            sampling_timesteps = self.sampling_timesteps
        batch, device, total_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size = 16, return_condition_image=False, return_all_timesteps = False):
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, h, w), return_condition_image, return_all_timesteps = return_all_timesteps)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, cond = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, cond, self_cond)

        return img

   # Update p_losses to pass cond along:
    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None, cond = None):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, cond=cond).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, cond=cond, x_self_cond=x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)

        if self.hybrid_loss:
            # Get the model's reverse distribution parameters:
            model_mean, _, model_log_variance, _ = self.p_mean_variance(x=x, t=t, x_self_cond=x_self_cond, clip_denoised=True)

            # Get the true posterior parameters:
            posterior_mean, posterior_variance, posterior_log_variance_clipped = self.q_posterior(x_start, x, t)

            # Compute KL divergence per sample (elementwise):
            kl = 0.5 * (
                posterior_log_variance_clipped - model_log_variance +
                (torch.exp(model_log_variance) + (model_mean - posterior_mean)**2) / posterior_variance - 1
            )
            # Average KL over non-batch dimensions:
            kl = reduce(kl, 'b ... -> b', 'mean')

            # Optionally, only consider KL for t > 0:
            mask = (t > 0).float()
            kl = (kl * mask).sum() / (mask.sum() + 1e-8)

            loss = loss + 0.001 * kl

        return loss.mean()

    def forward(self, img, cond=None, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, cond=cond, *args, **kwargs)

    
 # -------------------------------------------------------------------------
 # 3.  Trainer that feeds the condition tensor
 # -------------------------------------------------------------------------

class ImageConditionalTrainer(Trainer):
    """
    Extend the Trainer to work with a DataLoader that returns (target, cond) tuples.

    Usage is identical to the parent class, except that the ``dataset`` you
    provide must yield two tensors: the ground‑truth image and its conditioning
    counterpart.  These are automatically moved to the correct device and
    passed to the diffusion model as ``cond=...``.
    """
    def __init__(self, diffusion_model, dataset, **kwargs):
        # We pass a dummy folder to the base Trainer (it won't be used)
        super().__init__(diffusion_model, folder='.', **kwargs)
        dl = DataLoader(dataset, batch_size =  self.batch_size, shuffle = True, pin_memory = True)
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

    # override one training step to unpack (target, cond)
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        if accelerator.is_main_process:
            self.save_training_params(str(self.results_folder))

        writer = SummaryWriter(log_dir=str(self.results_folder / "tensorboard_logs"))

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    target, condition = next(self.dl)      # adjusted for conditional
                    target, condition = target.to(device), condition.to(device) 

                    with self.accelerator.autocast():
                        loss = self.model(target, condition)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    writer.add_scalar("Train/Loss", total_loss, self.step)
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: torch.cat(self.ema.ema_model.sample(batch_size=n, return_condition_image=True), dim=3), batches))    # change

                        all_images = torch.cat(all_images_list, dim = 0)

                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                        # Log generated samples to TensorBoard
                        # Assuming all_images is in [C, H, W] format per sample (batched as [B, C, H, W])
                        writer.add_images("Samples", all_images, self.step, dataformats="NCHW")

                        # Generate num_fid_samples 
                        if self.calculate_fid or self.calculate_is:
                            accelerator.print(f"Generating {self.fid_scorer.n_samples} sample for calculating FID and IS.")
                            all_fake_samples_list = [] 
                            with torch.inference_mode():
                                batches = num_to_groups(self.fid_scorer.n_samples, self.batch_size)
                                for batch in tqdm(batches):
                                    # Generate a batch of images.
                                    fake_samples =  self.ema.ema_model.sample(batch_size=batch).to(self.device)
                                    all_fake_samples_list.append(fake_samples)

                            all_fake_samples = torch.cat(all_fake_samples_list, dim=0)

                        # whether to calculate fid
                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score(all_fake_samples)
                            accelerator.print(f'FID score: {fid_score}')
                            writer.add_scalar("Eval/FID", fid_score, self.step)

                        # whether to calculate IS
                        if self.calculate_is:
                            is_score = self.is_scorer.calculate_inception_score(all_fake_samples)
                            self.accelerator.print(f'Inception Score: {is_score:.4f}')
                            writer.add_scalar("Eval/IS", is_score, self.step)


                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                        self.accelerator.print()

                pbar.update(1)
            
            torch.cuda.empty_cache()

        accelerator.print('training complete')
        writer.close()
