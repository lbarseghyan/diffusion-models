import sys
sys.path.append('denoising-diffusion-pytorch')

import os
import random
import pickle
import clip
from PIL import Image
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T
from denoising_diffusion_pytorch.utils import *


class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)



class ImageConditionalDataset(Dataset):
    """
    This dataset expects a folder structure:
        root/
            condition/  --> edge maps (e.g., "99_A.png")
            target/     --> corresponding shoe images (e.g., "99_B.png")
    The pairing is done based on the numeric prefix (e.g., "99").
    """
    def __init__(self, 
                 root, 
                 image_size,
                 augment_horizontal_flip=False,
                 convert_image_to=None):
        self.root = Path(root)
        self.image_size = image_size
        
        # Get list of condition images
        self.cond_folder = self.root / "condition"
        self.target_folder = self.root / "target"
        self.cond_paths = sorted(list(self.cond_folder.glob("*.*")))
        
        # Define transform:
        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.cond_paths)
    
    def __getitem__(self, index):
        # Get condition file path
        cond_path = self.cond_paths[index]
        # Extract numeric prefix to match with target image.
        # Assuming condition image names like "99_A.png" and target "99_B.png"
        prefix = cond_path.stem.split("_")[0]
        target_path = self.target_folder / f"{prefix}_B.jpg"
        
        # Load images
        cond_img = Image.open(cond_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")
        
        cond = self.transform(cond_img)
        target = self.transform(target_img)
        return target, cond


class TextConditionalDataset(Dataset):
    """
    This dataset expects a folder structure:
        root/
            condition/  --> captions (e.g., "12.txt")
            target/     --> corresponding images (e.g., "12.jpeg")
    The pairing is done based on the numeric prefix (e.g., "12").
    """
    def __init__(self, 
                 root, 
                 image_size,
                 augment_horizontal_flip=False,
                 convert_image_to=None,
                 embedding_file=None):
        
        self.root = Path(root)
        self.image_size = image_size
        
        # Get list of condition images
        self.cond_folder = self.root / "condition"
        self.target_folder = self.root / "target"
        # target_folder = self.root / "target"
        # self.cond_paths = sorted(list(self.cond_folder.glob("*.*")))
        self.target_paths = sorted(list(self.target_folder.glob("*.*")))
        
        # Define transform:
        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.model_clip, self.preprocess = clip.load("ViT-B/32", device=self.device) 

        if exists(embedding_file):
            self.embedding_file = embedding_file
        else:
            self.embedding_file = self.root / "text_embeddings.pkl"

        if os.path.exists(self.embedding_file):
            with open(self.embedding_file, "rb") as f:
                self.embeddings_dict = pickle.load(f)
        else:
            self.embeddings_dict = self.precompute_text_embeddings()

    def precompute_text_embeddings(self):
        """
        For each text file in condition_folder, compute the text embeddings for all captions
        and save them in a dictionary where keys are the base filenames (without extension)
        and values are dictionaries with:
          - 'captions': a list of captions (the raw text)
          - 'embeddings': a numpy array of shape (num_captions, embedding_dim)
        """
        # Load CLIP model
        clip_model, _ = clip.load("ViT-B/32", device=self.device)
        clip_model.eval()
        
        condition_folder = Path(self.cond_folder)
        embeddings_dict = {}
        
        # Process each text file (assume extension .txt)
        for txt_file in sorted(condition_folder.glob("*.txt")):
            base_name = txt_file.stem  # e.g., "000000123456"
            with open(txt_file, "r", encoding="utf-8") as f:
                captions = [line.strip() for line in f if line.strip()]
            if len(captions) == 0:
                continue  # Skip files with no valid captions

            # Tokenize all captions at once
            tokens = clip.tokenize(captions).to(self.device)  # shape: (num_captions, token_length)
            with torch.no_grad():
                text_embeddings = clip_model.encode_text(tokens)  # shape: (num_captions, embedding_dim)
            
            # Store both captions and embeddings in the dictionary
            embeddings_dict[base_name] = {
                "captions": captions,
                "embeddings": text_embeddings.cpu().numpy()
            }
        
        # Save the dictionary to file
        with open(self.embedding_file, "wb") as f:
            pickle.dump(embeddings_dict, f)
        print(f"Precomputed text embeddings saved to {self.embedding_file}")

        return embeddings_dict

    def __len__(self):
        return len(self.target_paths)
    
    def __getitem__(self, index):     ### improve ###
        # Load target image.
        target_path = self.target_paths[index]
        target_img = Image.open(target_path).convert("RGB")
        target = self.transform(target_img)
        
        # Get the corresponding base name and precomputed text embeddings.
        base_name = target_path.stem  # e.g., "000000123456"
        data = self.embeddings_dict.get(base_name, None)
        embeddings = data["embeddings"] if data else None
        
        # Randomly select one embedding if available.
        if embeddings is not None:
            # embeddings is a numpy array of shape (num_captions, embedding_dim)
            num_captions = embeddings.shape[0]
            chosen_index = random.randint(0, num_captions - 1)
            text_emb = torch.tensor(embeddings[chosen_index], dtype=torch.float)
        else:
            # Fallback: if no embeddings exist, use a zero vector (adjust dimension as needed, e.g., 512)
            text_emb = torch.zeros(512, dtype=torch.float)
        
        return target, text_emb