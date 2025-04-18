import os
import torch
from torch import nn

# Import the base GaussianDiffusion (DDPM) and VAE (VQModel) from the repository
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_text_conditional import GaussianDiffusion
from ldm.models.autoencoder import VQModel  # VAE with .encode() and .decode() methods

class LatentDiffusionText(GaussianDiffusion):
    def __init__(
        self,
        model: nn.Module,
        vae: nn.Module,
        latent_shape: tuple,    # (channels, height, width) for one latent sample
        timesteps: int = 1000,
        # conditioning_mode: str = "concat",  # "concat" or "cross-attn"
        text_emb_dim: int = 512,
        **kwargs
    ):
        """
        Latent Diffusion Model for text conditioning.
        
        This version inherits directly from GaussianDiffusion (the text-conditional version)
        and adds VAE encoding/decoding to work in the latent space.
        
        :param model: A U-Net (or diffusion model) that already supports a text_emb argument.
        :param vae: Pretrained VAE for encoding images to and decoding images from latent space.
        :param latent_shape: Tuple (channels, height, width) describing a single latent sample.
        :param timesteps: Number of diffusion timesteps.
        :param conditioning_mode: Specifies text conditioning method.
                                  "concat" applies a projection on text embeddings and concatenates with a time embedding;
                                  "cross-attn" passes the text embedding directly to the network (which should use cross-attention).
        :param text_emb_dim: Dimensionality of the text embeddings (e.g. from a pretrained CLIP text encoder).
        :param kwargs: Additional keyword arguments for GaussianDiffusion.
        """
        # Call the parent constructor.
        # GaussianDiffusion expects an image_size (spatial dims) from latent_shape.
        super().__init__(model, image_size=latent_shape[1], timesteps=timesteps, **kwargs)
        
        self.vae = vae
        self.latent_channels = latent_shape[0]
        # self.conditioning_mode = conditioning_mode
        self.text_emb_dim = text_emb_dim
        # Freeze the VAE (do not compute gradients for its parameters)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # In text-conditional DDPM the model already has a time embedding network (e.g. self.model.time_mlp).
        # When using the "concat" conditioning mode, we will process the text embedding and combine it with a fixed time embedding.

        # if self.conditioning_mode == "concat":
        #     self.text_proj = nn.Sequential(
        #         nn.Linear(text_emb_dim, self.model.time_mlp[-1].out_features),
        #         nn.GELU(),
        #         nn.Linear(self.model.time_mlp[-1].out_features, self.model.time_mlp[-1].out_features)
        #     )
        #     self.concat_proj = nn.Linear(self.model.time_mlp[-1].out_features * 2, self.model.time_mlp[-1].out_features)

        # In "cross-attn" mode we assume the diffusion model (Unet) already handles text tokens via cross-attention,
        # so no additional projection is applied here.

        
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
    

    def forward(self, target: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the diffusion loss on target images conditioned on text.
        
        :param target: Batch of target images.
        :param text_emb: Batch of text embeddings (precomputed via a CLIP encoder, for instance).
        :return: Diffusion loss in the latent space.
        
        The target images are first encoded into the latent space. Then, the parent's forward
        method is called on these latents while passing along the text embedding.
        """
        # Encode the target images into latent space.
        target_latents = self.encode(target)
        if isinstance(target_latents, tuple):
            target_latents = target_latents[0]
        # Call the parent's forward method with the latent images.
        # The parent's forward expects a 'text_emb' keyword argument.
        return super().forward(target_latents, text_emb=text_emb)
    
    @torch.inference_mode()
    def sample(self, batch_size = 16, save_path_for_text=None, return_all_timesteps = False):
        """
        Generate samples conditioned on the provided text embeddings.
        
        :param text_emb: Batch of text embeddings.
        :param batch_size: Number of images to generate.
        :param return_all_timesteps: Whether to return intermediate samples.
        :return: Generated images (decoded from latent samples).
        
        The method uses the appropriate sampling routine (p_sample_loop or ddim_sample)
        on the latent space and then decodes the generated latents via the VAE.
        """
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        
        # Process text embedding according to conditioning mode.
        # if self.conditioning_mode == "concat":
        #     # Use a dummy timestep (e.g. the midpoint) to obtain a representative time embedding.
        #     dummy_t = torch.full((batch_size,), self.num_timesteps // 2, device=text_emb.device, dtype=torch.long)
        #     t_emb = self.model.time_mlp(dummy_t)
        #     text_features = self.text_proj(text_emb)
        #     combined = torch.cat([t_emb, text_features], dim=-1)
        #     conditioning = self.concat_proj(combined)
        # else:
        #     conditioning = text_emb

        # conditioning = text_emb     

        # Sample latent codes using the GaussianDiffusion sampling function.
        latent_samples = sample_fn((batch_size, channels, h, w), save_path_for_text, return_all_timesteps = return_all_timesteps)

        # Decode latent samples back to image space.
        return self.decode(latent_samples)