import os
import torch
from torch import nn

# Import the base GaussianDiffusion (DDPM) and VAE (VQModel) from the repository
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_text_conditional import GaussianDiffusion
from denoising_diffusion_pytorch.utils import identity

from ldm.models.autoencoder import VQModel  # VAE with .encode() and .decode() methods

class LatentDiffusionText(GaussianDiffusion):
    def __init__(self, model, vae, latent_shape, text_emb_dim = 512, **kwargs):
        """
        Latent Diffusion Model for text conditioning.
        
        This version inherits directly from GaussianDiffusion (the text-conditional version)
        and adds VAE encoding/decoding to work in the latent space.
        
        :param model: A U-Net (or diffusion model) that already supports a text_emb argument.
        :param vae: Pretrained VAE for encoding images to and decoding images from latent space.
        :param latent_shape: Tuple (channels, height, width) describing a single latent sample.
        :param text_emb_dim: Dimensionality of the text embeddings (e.g. from a pretrained CLIP text encoder).
        :param kwargs: Additional keyword arguments for GaussianDiffusion.
        """

        # GaussianDiffusion expects an image_size (spatial dims) from latent_shape.
        super().__init__(model, image_size=latent_shape[1],  **kwargs)
        
        self.vae = vae
        self.latent_channels = latent_shape[0]  # Save latent channels if needed.
        # Optionally, you can override the model's channels attribute if required.  
        self.model.channels = self.latent_channels

        self.text_emb_dim = text_emb_dim

        self.normalize = identity 
        self.unnormalize = identity   

        # Freeze the VAE (do not compute gradients for its parameters)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        
        
    def encode(self, images):
        """Encode images to latent space using the VAE encoder."""
        with torch.no_grad():
            # VQModel.encode may return a tuple (latents, indices); assume first is latent tensor
            latents = self.vae.encode(images)
            if isinstance(latents, tuple):
                latents = latents[0]
        return latents


    def decode(self, latents):
        """Decode latent representations back to image space using the VAE decoder."""
        with torch.no_grad():
            images = self.vae.decode(latents)
        return images
    

    def forward(self, target, text_emb):
        """
        Forward pass computes the diffusion loss on target images conditioned on text.
        
        :param target: Batch of target images.
        :param text_emb: Batch of text embeddings (precomputed via a CLIP encoder, for instance).
        :return: Diffusion loss in the latent space.
        """

        # Encode the target images into latent space.
        target_latents = self.encode(target)
        if isinstance(target_latents, tuple):
            target_latents = target_latents[0]
        # Use GaussianDiffusion's forward (or loss computation) on latents
        return super().forward(target_latents, text_emb=text_emb)
    
    
    @torch.inference_mode()
    def sample(self, batch_size = 16, save_path_for_text=None, return_all_timesteps = False):
        """
        Generate samples conditioned on the provided text embeddings.
        
        :param text_emb: Batch of text embeddings.
        :param batch_size: Number of images to generate.
        :save_path_for_text: Path for saving the text for which the samples are generated. 
        :param return_all_timesteps: Whether to return intermediate samples.
        :return: Generated images (decoded from latent samples).
        
        The method uses the appropriate sampling routine (p_sample_loop or ddim_sample)
        on the latent space and then decodes the generated latents via the VAE.
        """
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample 

        # Sample latent codes using the GaussianDiffusion sampling function.
        latent_samples = sample_fn((batch_size, channels, h, w), save_path_for_text, return_all_timesteps = return_all_timesteps)
        image_samples = self.decode(latent_samples)

        return image_samples