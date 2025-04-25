import torch
import torch.nn as nn

# Import the base GaussianDiffusion (DDPM) and VAE (VQModel) from the repository
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch.utils import identity
from ldm.models.autoencoder import VQModel  # VAE with .encode() and .decode() methods

class LatentDiffusion(GaussianDiffusion):
    def __init__(self, model: nn.Module, vae: nn.Module, latent_shape: tuple, timesteps: int = 1000, **kwargs):
        """
        Latent Diffusion Model: Diffusion in VAE latent space.
        :param model: The U-Net or diffusion model operating on latent space.
        :param vae: Pre-trained VAE (with encoder & decoder) to compress images.
        :param latent_shape: Shape of the VAE latent (channels, height, width).
        :param timesteps: Number of diffusion timesteps (e.g., 1000).
        :param kwargs: Additional arguments for GaussianDiffusion (e.g., beta schedule).
        """
        # Call base class __init__ using image_size (spatial dimension) from latent_shape,
        # but do not pass channels as a separate argument.
        super().__init__(model, image_size=latent_shape[1], timesteps=timesteps, **kwargs)
        # Save the VAE and latent shape.
        self.vae = vae
        self.latent_channels = latent_shape[0]  # Save latent channels if needed.
        # Optionally, you can override the model's channels attribute if required.
        self.model.channels = self.latent_channels
        self.normalize = identity 
        self.unnormalize = identity    

        # Freeze VAE parameters if not fine-tuning
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False


    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space using the VAE encoder."""
        with torch.no_grad():
            # VQModel.encode may return a tuple (latents, indices); assume first is latent tensor
            latents = self.vae.encode(images)
            if isinstance(latents, tuple):
                latents = latents[0]
        return latents


    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent representations back to image space using the VAE decoder."""
        with torch.no_grad():
            images = self.vae.decode(latents)
        return images


    def forward(self, real_images: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion loss on a batch of real images (for training).
        1. Encode images to latents.
        2. Compute diffusion loss in latent space using GaussianDiffusion.
        """
        # Encode images to latent space
        latents = self.encode(real_images)
        # Use GaussianDiffusion's forward (or loss computation) on latents
        return super().forward(latents)


    @torch.inference_mode()
    def sample(self, batch_size=16, return_all_timesteps=False):
        # Get latent spatial dimensions and channels:
        # In __init__, self.image_size was set to the spatial size of the latent (e.g. (8, 8) or (2, 2))
        # and self.channels is available (or you can use self.latent_channels)
        (h, w), channels = self.image_size, self.channels  
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        # sample_fn returns latents (the same shape as defined)
        latents = sample_fn((batch_size, channels, h, w), return_all_timesteps=return_all_timesteps)
        # Decode the latent codes back to pixel space using the VAE decoder
        return self.decode(latents)