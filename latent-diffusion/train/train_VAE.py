import os
import sys
import argparse
import yaml

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np

from utils.data import ImageFolderDataset
from utils.callback import SampleCallback

# ─── Load config ─────────────────────────────────────────────────────────

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml',
                    help='Path to YAML config file')
args = parser.parse_args()
cfg = load_config(args.config)


# ─── Data Transforms and DataLoader Setup ─────────────────────────────────────────────────────────

# Using CIFAR-10 style normalization: map [0,1] to roughly [-1,1]
size = (cfg['ddconfig']['resolution'], cfg['ddconfig']['resolution'])
transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = ImageFolderDataset(folder=cfg['train_images_path'], transform=transform)
val_dataset = ImageFolderDataset(folder=cfg['val_images_path'], transform=transform)

batch_size = cfg.get('batch_size', 128)
num_workers = cfg.get('num_workers', 4)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)



# ─── Import and Instantiate Your VQModel ─────────────────────────────────────────────────────────

HERE = os.path.dirname(__file__)                      # …/latent-diffusion/train
BASE = os.path.abspath(os.path.join(HERE, os.pardir))

sys.path.append(BASE)
from ldm.models.autoencoder import VQModel

model = VQModel(
    ddconfig=cfg['ddconfig'],
    lossconfig=cfg['lossconfig'],
    n_embed=cfg['n_embed'],
    embed_dim=cfg['embed_dim'],
    monitor="val/rec_loss",  # monitor metric for checkpointing
)

model.learning_rate = cfg['base_learning_rate']

print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters.")



# ─── Set Up Checkpoint and Sample Callbacks ─────────────────────────────────────────────────────────

from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath=cfg['checkpoints_path'],
    filename=cfg['filename'],
    every_n_epochs=cfg['every_n_epochs'],
    save_top_k=-1,  # save all checkpoints created at these intervals
)

sample_callback = SampleCallback(
    every_n_epochs=cfg['every_n_epochs'],
    sample_dir=cfg['samples_path']
)


# ─── PyTorch Lightning Trainer Setup ─────────────────────────────────────────────────────────

accelerator = "gpu" if torch.cuda.is_available() else "cpu"
devices = 1

trainer = pl.Trainer(
    max_epochs=cfg['max_epochs'],
    accelerator=accelerator,
    devices=devices,
    callbacks=[checkpoint_callback, sample_callback],
)



# ─── Main Training Routine ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    pl.seed_everything(42)
    trainer.fit(model, 
                train_dataloaders=train_loader, 
                val_dataloaders=val_loader,
                ckpt_path = cfg['ckpt_path'])