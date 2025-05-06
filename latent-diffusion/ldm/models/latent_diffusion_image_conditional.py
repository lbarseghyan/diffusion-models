import os
import torch
from torch import nn

# Import the base ImageConditionalDenoisingDiffusion (DDPM) and VAE (VQModel) from the repository
from denoising_diffusion.denoising_diffusion_image_conditional import ImageConditionalDenoisingDiffusion
from denoising_diffusion.utils import identity

from ldm.models.autoencoder import VQModel  # VAE with .encode() and .decode() methods

class ImageConditionalLatentDiffusionImage(ImageConditionalDenoisingDiffusion):
    def __init__(self, model, vae, latent_shape, condition_data_folder, cond_vae=None,  **kwargs):
        """
        Latent Diffusion Model for image conditioning.
        
        This version inherits directly from ImageConditionalDenoisingDiffusion (the image-conditional version)
        and adds VAE encoding/decoding to work in the latent space.
        
        :param model: A U-Net (or diffusion model) that already supports a text_emb argument.
        :param vae: Pretrained VAE for encoding images to and decoding images from latent space.
        :param latent_shape: Tuple (channels, height, width) describing a single latent sample.
        :param cond_vae: (Optional) A separate VAE trained for the conditioning image. If None,
                             the same VAE is used.
        :param condition_data_folder: Path of the folder for conditional images.
        :param kwargs: Additional keyword arguments for ImageConditionalDenoisingDiffusion.
        """

        # ImageConditionalDenoisingDiffusion expects an image_size (spatial dims) from latent_shape.
        super().__init__(model, image_size=latent_shape[1], **kwargs)
        
        self.vae = vae
        self.cond_vae = cond_vae if cond_vae is not None else self.vae

        self.latent_channels = latent_shape[0]   # Save latent channels if needed.
        # Optionally, you can override the model's channels attribute if required.  
        self.model.channels = self.latent_channels

        self.normalize = identity 
        self.unnormalize = identity

        self.condition_data_folder = condition_data_folder  

        # Freeze the VAE (do not compute gradients for its parameters)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        if cond_vae is not None:
            self.cond_vae.eval()
            for param in self.cond_vae.parameters():
                param.requires_grad = False
        
        
    def encode(self, images, cond = False): 
        """Encode images to latent space using the VAE encoder."""
        with torch.no_grad():
            if cond:
                latents = self.cond_vae.encode(images)
            else:
                latents = self.vae.encode(images)
            # VQModel.encode may return a tuple (latents, indices); assume first is latent tensor
            if isinstance(latents, tuple):
                latents = latents[0]
        return latents


    def decode(self, latents, cond = False): 
        """Decode latent representations back to image space using the VAE decoder."""
        with torch.no_grad():
            if cond:
                images = self.cond_vae.encode(latents)
            else:
                images = self.vae.encode(latents)
        return images
    

    def forward(self, target, cond):
        """
        Forward pass computes the diffusion loss on target images conditioned on image.
        
        :param target: Batch of target images.
        :param cond: Batch of condition images.
        :return: Diffusion loss in the latent space.
        """
        # Encode the target images into latent space.
        target_latents = self.encode(target)
        cond_latents = self.encode(cond, cond=True)
        # Use ImageConditionalDenoisingDiffusion's forward (or loss computation) on latents
        return super().forward(target_latents, cond=cond_latents)
    

    @torch.inference_mode()
    def sample(self, batch_size = 16, return_condition_image=False, return_all_timesteps = False):
        """
        Generate samples conditioned on the provided image.
        
        :param text_emb: Batch of text embeddings.
        :param batch_size: Number of images to generate.
        :param return_condition_image: Whether to return the conditional images.
        :param return_all_timesteps: Whether to return intermediate samples.
        :return: Generated images (decoded from latent samples).
        
        The method uses the appropriate sampling routine (p_sample_loop or ddim_sample)
        on the latent space and then decodes the generated latents via the VAE.
        """
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        
        samples = sample_fn((batch_size, channels, h, w), return_condition_image, return_all_timesteps = return_all_timesteps)

        if return_condition_image:
            # Decode latent samples back to image space.
            image_samples = self.decode(samples[1])
            cond = samples[0]
            return (image_samples, cond)
        else:
            image_samples = self.decode(samples)
            return image_samples
