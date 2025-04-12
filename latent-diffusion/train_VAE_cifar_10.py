import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np


######################################
# Custom Dataset for Image Folder
######################################
class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.images = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        image = Image.open(path).convert("RGB")
        if self.transform is None:
            np_img = np.array(image)  # shape [H, W, C]
            tensor_img = torch.tensor(np_img).float() / 255.0
            return {"image": tensor_img}
        else:
            img = self.transform(image)
            if img.dim() == 3:
                # Permute tensor from CHW to HWC if needed
                img = img.permute(1, 2, 0)
            return {"image": img}


######################################
# Data Transforms and DataLoader Setup
######################################

# Using CIFAR-10 style normalization: map [0,1] to roughly [-1,1]
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = ImageFolderDataset(folder='../data/cifar_small/train_images', transform=transform)
val_dataset = ImageFolderDataset(folder='../data/cifar_small/test_images', transform=transform)

batch_size = 128
num_workers = 4

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)


######################################
# Import and Instantiate Your VQModel
######################################
from ldm.models.autoencoder import VQModel

ddconfig = {
    "double_z": False,
    "z_channels": 3,
    "resolution": 32,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 64,
    "ch_mult": [1, 2, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [],
    "dropout": 0.0
}

lossconfig = {
    "target": "taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
    "params": {
        "disc_conditional": False,
        "disc_in_channels": 3,
        "disc_start": 0,
        "disc_weight": 0.75,
        "codebook_weight": 1.0,
    }
}

embed_dim = 3
n_embed = 8192
base_learning_rate = 4.5e-06

model = VQModel(
    ddconfig=ddconfig,
    lossconfig=lossconfig,
    n_embed=n_embed,
    embed_dim=embed_dim,
    monitor="val/rec_loss",  # monitor metric for checkpointing
)

model.learning_rate = base_learning_rate

print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters.")


##################################
# Custom Callback to Save Samples
##################################

from pytorch_lightning.callbacks import Callback, ModelCheckpoint

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
        if (epoch + 1) % self.every_n_epochs == 0:
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

            latent_path = os.path.join(self.sample_dir, f"epoch_{epoch+1}_latent.pt")
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
                concat_path = os.path.join(self.sample_dir, f"epoch_{epoch+1}_concatenated_{i}.png")
                concat_img.save(concat_path)

            print(f"Epoch {epoch+1}: Saved latent tensor and concatenated input-reconstruction image samples.")

#########################################
# Set Up Checkpoint and Sample Callbacks
#########################################

# Save every n epochs (for example, n = 5)
n = 5

checkpoint_callback = ModelCheckpoint(
    dirpath="results/VAE/checkpoints/",
    filename="model_epoch{epoch:02d}",
    every_n_epochs=n,
    save_top_k=-1,  # save all checkpoints created at these intervals
)

sample_callback = SampleCallback(
    every_n_epochs=n,
    sample_dir="results/VAE/samples"
)


##################################
# PyTorch Lightning Trainer Setup
##################################
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
devices = 1

trainer = pl.Trainer(
    max_epochs=1500000,
    accelerator=accelerator,
    devices=devices,
    callbacks=[checkpoint_callback, sample_callback],
)


########################
# Main Training Routine
########################
if __name__ == '__main__':
    pl.seed_everything(42)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
