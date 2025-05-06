import math
import os
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from einops import repeat
from tqdm.auto import tqdm
from denoising_diffusion.utils import num_to_groups


class InceptionScoreEvaluation:
    def __init__(
        self,
        batch_size,
        sampler,
        channels=3,
        accelerator=None,
        stats_dir="./results",
        device="cuda",
        num_samples=50000,
    ):
        """
        Args:
            batch_size (int): Batch size for generating samples.
            sampler (object): Diffusion model sampler with a .sample(batch_size) method.
            channels (int): Number of channels in generated images.
            accelerator: Accelerator instance for logging (optional).
            stats_dir (str): Directory to store the Inception Score log file.
            device (str): Device to run inference on.
            num_samples (int): Total number of samples to generate for IS computation.
        """
        self.batch_size = batch_size
        self.n_samples = num_samples
        self.device = device
        self.channels = channels
        self.sampler = sampler
        self.stats_dir = stats_dir
        self.print_fn = print if accelerator is None else accelerator.print

        # Load a pretrained Inception v3 model (without auxiliary logits)
        self.inception_model = inception_v3(weights="IMAGENET1K_V1", aux_logits=True).to(device)
        self.inception_model.eval()
        for p in self.inception_model.parameters():
            p.requires_grad = False

        # Ensure the results directory exists and set up a log file for IS scores.
        os.makedirs(stats_dir, exist_ok=True)
        self.log_path = os.path.join(stats_dir, "inception_score_log.txt")


    @torch.inference_mode()
    def calculate_inception_score(self, fake_samples):
        """
        Compute the Inception Score (IS) given a batch (or set) of generated fake_samples.
        
        Args:
            fake_samples (torch.Tensor): Generated images (N, C, H, W)
            
        Returns:
            float: The Inception Score.
        """
        self.sampler.eval()
        preds_list = []
        # fake_samples is assumed to be a concatenation of multiple batches.
        self.print_fn(f"Calculating Inception Score on {fake_samples.shape[0]} generated samples.")

        # Process fake_samples in mini-batches.
        batches = num_to_groups(fake_samples.shape[0], self.batch_size)
        for batch in batches:
            batch_samples = fake_samples[:batch].to(self.device)
            fake_samples = fake_samples[batch:]  # Remove the batch we just processed

            # If grayscale, convert to 3-channel RGB.
            if self.channels == 1:
                batch_samples = repeat(batch_samples, "b 1 h w -> b 3 h w")

            # Convert range [-1, 1] to [0, 1] if necessary.
            if batch_samples.min() < 0:
                batch_samples = (batch_samples + 1) / 2.0

            # Resize to 299x299 for Inception v3.
            if batch_samples.shape[-2:] != (299, 299):
                batch_samples = F.interpolate(batch_samples, size=(299, 299), mode='bilinear', align_corners=False)

            # Normalize using ImageNet mean and std.
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            batch_samples = (batch_samples - mean) / std

            # Get logits and softmax probabilities.
            logits = self.inception_model(batch_samples)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1)
            preds_list.append(probs.cpu())

        preds = torch.cat(preds_list, dim=0)  # shape: (N, 1000)
        p_y = preds.mean(dim=0)

        eps = 1e-10
        kl_div = preds * (torch.log(preds + eps) - torch.log(p_y + eps))
        kl_div = kl_div.sum(dim=1)
        avg_kl_div = kl_div.mean().item()
        inception_score = math.exp(avg_kl_div)

        # Log the computed Inception Score to file.
        try:
            with open(self.log_path, "a") as f:
                f.write(f"{inception_score}\n")
        except Exception as e:
            self.print_fn("Warning: could not write Inception Score to log file:", e)

        # self.print_fn(f"Inception Score: {inception_score:.4f}")
        return inception_score
