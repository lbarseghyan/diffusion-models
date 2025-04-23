import os
import torch
from PIL import Image
import numpy as np

# Custom Dataset for Image Folder
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