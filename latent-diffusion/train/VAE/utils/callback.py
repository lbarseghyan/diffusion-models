import os
import torch
from torchvision import transforms
from pytorch_lightning.callbacks import Callback


class SampleCallback(Callback):
    """
    Saves a batch of latent representations and a few reconstructed images.
    Also, for each saved sample, concatenates the original input image with
    its reconstruction side-by-side and saves the concatenated image.
    This happens every `every_n_epochs` epochs.
    """
    def __init__(self, every_n_epochs: int, sample_dir="samples"):
        self.every_n_epochs = every_n_epochs
        self.sample_dir = sample_dir
         # Ensure sample_dir is a directory.
        if os.path.exists(self.sample_dir):
            if not os.path.isdir(self.sample_dir):
                os.remove(self.sample_dir)
        os.makedirs(self.sample_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs == 0:
            # Support both single DataLoader and list of dataloaders.
            val_loader = (trainer.val_dataloaders[0]
                          if isinstance(trainer.val_dataloaders, list)
                          else trainer.val_dataloaders)
            sample_batch = next(iter(val_loader))

            # Retrieve inputs from the model's get_input method.
            input_images = pl_module.get_input(sample_batch, pl_module.image_key).to(pl_module.device)

            # Set model to eval mode and run inference.
            pl_module.eval()
            with torch.no_grad():
                # For VQModel: encode returns (quant, emb_loss, info)
                quant, _, _ = pl_module.encode(input_images)
                reconstruction = pl_module.decode(quant)
            pl_module.train()

            latent_path = os.path.join(self.sample_dir, f"epoch_{epoch}_latent.pt")
            torch.save(quant.cpu(), latent_path)

            to_pil = transforms.ToPILImage()

            # Loop over the first few samples (up to 5) in the batch.
            num_samples = min(5, reconstruction.size(0))
            for i in range(num_samples):
                # Get the original and reconstructed image tensors.
                orig_img_tensor = input_images[i].cpu()
                rec_img_tensor = reconstruction[i].cpu()

                # Invert the normalization from [-1, 1] to [0, 1].
                orig_img_tensor = (orig_img_tensor * 0.5 + 0.5).clamp(0, 1)
                rec_img_tensor = (rec_img_tensor * 0.5 + 0.5).clamp(0, 1)

                # Concatenate along the width (dim=2, because tensor shape is [C, H, W]).
                concatenated = torch.cat([orig_img_tensor, rec_img_tensor], dim=2)

                # Convert concatenated tensor to a PIL image.
                concat_img = to_pil(concatenated)

                # Save the concatenated image.
                concat_path = os.path.join(self.sample_dir, f"epoch_{epoch}_concatenated_{i}.png")
                concat_img.save(concat_path)

            print(f"Epoch {epoch}: Saved latent tensor and concatenated input-reconstruction image samples.")
