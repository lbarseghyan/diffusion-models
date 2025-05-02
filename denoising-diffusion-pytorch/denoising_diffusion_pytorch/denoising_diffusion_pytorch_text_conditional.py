import os
import math
import random     
import pickle
from pathlib import Path
from functools import partial
from collections import namedtuple
from tqdm.auto import tqdm
from einops import rearrange, reduce

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import  DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T, utils


from denoising_diffusion_pytorch.attend import Attend
from denoising_diffusion_pytorch.version import __version__
from denoising_diffusion_pytorch.utils import *
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


class RMSNorm1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        # For 2D/3D input, we want a parameter of shape (1, dim)
        self.g = nn.Parameter(torch.ones(1, dim))
        
    def forward(self, x):
        # x is expected to be of shape (batch, n, dim) or (batch, dim)
        return F.normalize(x, dim=-1) * self.g * self.scale

class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            RMSNorm1D(dim)
        )
        
    def forward(self, x, context):
        # x: image features (batch, n, dim)
        # context: text tokens (batch, m, context_dim) or (batch, context_dim)
        if context.ndim == 2:
            context = context.unsqueeze(1)  # convert (batch, context_dim) -> (batch, 1, context_dim)
            
        b, n, _ = x.shape
        b, m, _ = context.shape
        
        q = self.to_q(x)   # (b, n, inner_dim)
        k = self.to_k(context)   # (b, m, inner_dim)
        v = self.to_v(context)   # (b, m, inner_dim)
        
        # reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b m (h d) -> b h m d', h=self.heads)
        v = rearrange(v, 'b m (h d) -> b h m d', h=self.heads)
        
        # scaled dot-product attention
        attn_scores = torch.einsum('b h n d, b h m d -> b h n m', q, k) * self.scale
        attn = attn_scores.softmax(dim=-1)
        
        out = torch.einsum('b h n m, b h m d -> b h n d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

  

# --------------------------------------------------------------------------
# 1.  U‑Net that understands an *extra* image‑condition tensor
# --------------------------------------------------------------------------

class Unet(Unet):
    r"""Extends the baseline ``Unet`` with optional text conditioning.

    Two mechanisms are offered (choose at construction time):

    1. **Time‑embedding concatenation** – project the pooled text embedding to
       the same dimensionality as the time embedding and fuse them.
    2. **Cross‑attention** – inject token‑level features at three strategic
       points (down‑, mid‑, and up‑sampling paths).
    """

    def __init__(self, *, dim, init_dim = None, dim_mults = (1, 2, 4, 8), text_condition = True, text_emb_dim = 512, use_cross_attn = False, attn_dim_head=32, **base_kwargs):
        # Remove kwargs that the base class doesn't understand
        super().__init__(dim=dim, init_dim = init_dim, dim_mults = dim_mults, attn_dim_head=attn_dim_head, **base_kwargs)

        # Save options
        self.text_condition = text_condition
        self.use_cross_attn = use_cross_attn
        # self._text_emb_dim = text_emb_dim

        time_dim = dim * 4  #  ← _BaseUnet constructs this likewise12

        if text_condition and not use_cross_attn:
            # Simple concatenation pathway
            self.text_proj = nn.Sequential(
                nn.Linear(text_emb_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
            self.text_concat_proj = nn.Linear(time_dim * 2, time_dim)

        init_dim = default(init_dim, dim)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]

        if text_condition and use_cross_attn:
            # We create the cross‑attention modules **after** the base network
            # is built so that we know the channel dimensionalities.
            self.cross_attn      = CrossAttention(dim=dims[-1], context_dim=text_emb_dim, heads=4, dim_head=attn_dim_head)
            self.cross_attn_down = CrossAttention(dim=dims[-1], context_dim=text_emb_dim, heads=4, dim_head=attn_dim_head)
            self.cross_attn_up   = CrossAttention(dim=dims[-1], context_dim=text_emb_dim, heads=4, dim_head=attn_dim_head)

    # ------------------------------------------------------------------
    # Forward pass – copied from the base implementation but with hooks
    # ------------------------------------------------------------------

    def forward(self, x, time, text_emb = None, x_self_cond = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        # -------------------------------------------------------------
        # Text embedding – simple concatenation variant
        # -------------------------------------------------------------
        if self.text_condition and exists(text_emb) and (not self.use_cross_attn):
            if text_emb.dim() == 3 and text_emb.size(1) == 1:
                text_emb = text_emb.squeeze(1)
            text_emb = text_emb.to(t.dtype)  # Ensure text_emb is in the same dtype as t
            text_features = self.text_proj(text_emb)  # shape: (batch, time_dim)
            combined = torch.cat((t, text_features), dim=1)  # shape: (batch, 2*time_dim)
            t = self.text_concat_proj(combined)  # shape: (batch, time_dim)
  
        # --- Downsample path ------------------------------------------------

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        # ----------------------------------------
        # Cross‑attention variant
        # ----------------------------------------

        # Inject cross‑attention before bottleneck
        if self.text_condition and exists(text_emb) and self.use_cross_attn:
            b, c, h_sp, w_sp = x.shape
            x_flat = x.view(b, c, h_sp * w_sp).permute(0, 2, 1)     # (b, n, c)
            x_flat = self.cross_attn_down(x_flat, text_emb)        # (b, n, c)
            x = x_flat.permute(0, 2, 1).view(b, c, h_sp, w_sp)

        # --- Bottleneck ----------------------------------------------------
        x = self.mid_block1(x, t)

        # Apply cross-attention-based text conditioning if selected ---
        if self.text_condition and exists(text_emb) and self.use_cross_attn:
            # Here, text_emb should be of shape (batch, token_len, text_emb_dim)
            b, c, h_sp, w_sp = x.shape
            x_flat = x.view(b, c, h_sp * w_sp).permute(0, 2, 1)  # (b, n, c)
            x_flat = self.cross_attn(x_flat, text_emb)            # (b, n, c)
            x = x_flat.permute(0, 2, 1).view(b, c, h_sp, w_sp)
            
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        # Inject again after bottleneck 
        if self.text_condition and exists(text_emb) and self.use_cross_attn:
            b, c, h_sp, w_sp = x.shape
            x_flat = x.view(b, c, h_sp * w_sp).permute(0, 2, 1)
            x_flat = self.cross_attn_up(x_flat, text_emb)
            x = x_flat.permute(0, 2, 1).view(b, c, h_sp, w_sp)

        # --- Upsample path --------------------------------------------------
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
    
# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# --------------------------------------------------------------------------
# 2.  Diffusion process that forwards the condition through the Unet
# --------------------------------------------------------------------------

class GaussianDiffusion(GaussianDiffusion):
    """Wraps the base diffusion class, piping *text_emb* through the model."""

    def __init__(self, *, model, embedding_file, **kwargs):
        super().__init__(model=model, **kwargs)
        assert os.path.exists(
            embedding_file
        ), "Pre‑computed caption embeddings file not found."
        self.embedding_file = Path(embedding_file)

    def model_predictions(self, x, t, text_emb = None, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, text_emb = text_emb, x_self_cond=x_self_cond)
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

    def p_mean_variance(self, x, t, text_emb = None, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, text_emb = text_emb, x_self_cond=x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, text_emb = None, x_self_cond = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, text_emb = text_emb, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start


    # add
    def get_random_text_condition(self, batch, device):
        """
        Randomly sample a batch of text embeddings and their corresponding captions from the precomputed embeddings.
        Assumes that self.precomputed_embeddings is stored in a pickle file,
        where keys are the base filenames and values are dictionaries with:
          - 'captions': a list of strings
          - 'embeddings': a numpy array of shape (num_captions, embedding_dim)

        Returns:
            cond_batch: a tensor of shape (batch, embedding_dim)
            texts: a list of strings corresponding to the selected captions
        """
        with open(self.embedding_file, "rb") as f:
            precomputed_embeddings = pickle.load(f)
        print(f"Loaded precomputed embeddings from {self.embedding_file}")

        # Get all keys available in the precomputed embeddings dictionary.
        keys = list(precomputed_embeddings.keys())
        
        # Randomly sample 'batch' number of keys.
        selected_keys = random.choices(keys, k=batch)
        
        cond_embeddings = []
        texts = []
        
        for key in selected_keys:
            # Retrieve both captions and embeddings for the current key.
            data = precomputed_embeddings[key]
            embeddings_array = data["embeddings"]  # shape: (num_captions, embedding_dim)
            captions = data["captions"]              # list of captions
            num_captions = embeddings_array.shape[0]
            
            # Randomly select one caption embedding.
            chosen_index = random.randint(0, num_captions - 1)
            chosen_embedding = embeddings_array[chosen_index]
            chosen_caption = captions[chosen_index]
            
            # Convert the numpy array to a torch tensor.
            cond_embeddings.append(torch.tensor(chosen_embedding, dtype=torch.float))
            texts.append(chosen_caption)
        
        # Stack the embeddings into a tensor of shape (batch, embedding_dim) and move to the correct device.
        cond_batch = torch.stack(cond_embeddings, dim=0).to(device)
        return cond_batch, texts


    @torch.inference_mode()
    def p_sample_loop(self, shape, save_path_for_text=None, return_all_timesteps = False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        text_emb, texts = self.get_random_text_condition(batch, device)     # add

        # Save the corresponding captions if a path is provided.
        if exists(save_path_for_text):
            mode = 'a' if os.path.exists(save_path_for_text) else 'w'
            with open(save_path_for_text, mode) as txt_file:
                for text in texts:
                    txt_file.write(text + "\n")

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, text_emb, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)

        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, save_path_for_text = None, sampling_timesteps = None, return_all_timesteps = False):
    
        if sampling_timesteps is None:
            sampling_timesteps = self.sampling_timesteps
        batch, device, total_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        text_emb, texts = self.get_random_text_condition(batch, device)

        # Save the corresponding captions if a path is provided.
        if exists(save_path_for_text):
            mode = 'a' if os.path.exists(save_path_for_text) else 'w'
            with open(save_path_for_text, mode) as txt_file:
                for text in texts:
                    txt_file.write(text + "\n")

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, text_emb = text_emb, x_self_cond = self_cond, clip_x_start = True, rederive_pred_noise = True)

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
    def sample(self, batch_size = 16, save_path_for_text=None, return_all_timesteps = False):
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, h, w), save_path_for_text, return_all_timesteps = return_all_timesteps)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, text_emb = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, text_emb, self_cond)

        return img

    def p_losses(self, x_start, t, text_emb = None, noise = None, offset_noise_strength = None):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, text_emb = text_emb).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, text_emb = text_emb, x_self_cond=x_self_cond)

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
        # loss.mean()

        if self.hybrid_loss:
            # Get the model's reverse distribution parameters:
            model_mean, _, model_log_variance, _ = self.p_mean_variance(x=x, t=t, text_emb = text_emb, x_self_cond=x_self_cond, clip_denoised=True)

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

    def forward(self, img, text_emb = None, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, text_emb = text_emb, *args, **kwargs)
    

 # -------------------------------------------------------------------------
 # 3.  Trainer that feeds the condition tensor
 # -------------------------------------------------------------------------

class Trainer(Trainer):
    """
    Extend the Trainer to work with a DataLoader that returns (target, cond) tuples.
    """
    def __init__(self, diffusion_model, dataset, **kwargs):
        # We pass a dummy folder to the base Trainer (it won't be used)
        super().__init__(diffusion_model, folder='.', **kwargs)
        dl      = DataLoader(dataset, batch_size =  self.batch_size, shuffle = True, pin_memory = True)
        dl      = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

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
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n, save_path_for_text=str(self.results_folder / f'sample-{milestone}.txt')), batches))

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