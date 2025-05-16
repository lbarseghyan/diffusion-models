# Unified Framework for Diffusion Models 

A **unified PyTorch framework** for training, evaluating, and comparing modern diffusion‑based generative models – including **DDPM**, **DDIM**, and **LDM** – with both unconditional and conditional variants.

> This repository accompanies my master’s capstone thesis “\**Unified Framework for Diffusion Models with Comparative Analysis and Evaluation**.” It consolidates research code, reproducible experiments, and utilities into a single, easy‑to‑extend codebase.

<br>

## Features

| Category         | Highlights                                                                                                                                                                           |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Model Zoo**    | • DDPM & DDIM (based on `lucidrains/denoising‑diffusion‑pytorch`)• Latent Diffusion (ported & simplified from `CompVis/latent‑diffusion`)• Plug‑and‑play U‑Net backbones (2D/latent) |
| **Conditioning** | • Unconditional generation• Image‑to‑Image (edge ➜ image, etc.)• Text‑to‑Image with  • *embedding concatenation*  • *multi‑scale cross‑attention*                                    |                                                                                                                                                                     
| **Evaluation**   | Built‑in FID & IS computation, sample grids, TensorBoard support.                                                                                                                    |
| **Performance**  | Mixed‑precision, gradient accumulation, multi‑GPU via 🤗 `accelerate`.                                                                                                               |

<br>


## Getting Started

### 1. Clone 

```bash
git clone https://github.com/lbarseghyan/diffusion-models.git
cd diffusion-models
```

### 2. Create Environment

```bash
# Linux / CUDA‑enabled
conda env create -f environment.yml
conda activate diff

# macOS (CPU) 
conda env create -f environment_macos.yml
conda activate diff
```

<br>

## Data Preparation

Before training or evaluating diffusion models, download and preprocess the required datasets. Three helper scripts are provided under `data/scripts/`:

### Usage

1. **CIFAR‑10**

   ```bash
   cd data/scripts
   python3 download_and_preprocess_cifar10.py
   ```

   * Raw data will be placed under `data/raw/cifar-10-batches-py/` 
   * Processed data in `data/cifar-10/`

2. **edges2shoes**

   ```bash
   cd data/scripts
   python3 download_edges2shoes.py
   ```

   * Raw archive and extracted files under `data/raw/edges2shoes/`
   * Processed data in `data/edges2shoes/`

3. **COCO (minitrain, val, test)**

   ```bash
   cd data/scripts
   python3 download_and_preprocess_coco.py
   ```

   * COCO-minitrain via Kaggle CLI into `data/raw/coco/coco_minitrain/`
   * COCO 2017 val/test via direct URLs into `data/raw/coco/{val2017,test2017}/`
   * Processed data in `data/coco/`

After running each script, your `data/` folder will contain both the raw downloads (preserved) and the ready-to-use data for training and evaluation.

> **Note:** Once you have your processed data confirmed and no longer need to re-run preprocessing, you can safely delete the corresponding subfolders under `data/raw/` to free up disk space.

<br>

## Quick Examples

### Train DDPM on CIFAR‑10

```bash
python denoising-diffusion-pytorch/train/train_ddpm.py \
    --config=denoising-diffusion-pytorch/train/configs/ddpm_cifar.yaml
```


### Train LDM (edge ➜ shoe)

```bash
python latent-diffusion/train/train_ldm_image_conditional.py \
    --config=latent-diffusion/train/configs/ddpm_image_conditional_edges2shoes.yaml
```
<br>

## Contributing
Pull requests are welcome! Please open an issue first to discuss major changes.


## Citation
If you use this codebase, please cite the thesis:

```bibtex
@mastersthesis{Barseghyan2025Diffusion,
  author       = {Barseghyan, Laura},
  title        = {Unified Framework for Diffusion Models with Comparative Analysis and Evaluation},
  school       = {American University of Armenia},
  year         = {2025},
  url          = {https://github.com/lbarseghyan/diffusion-models}
}
```

<br>

This repository also incorporates code from:

* `lucidrains/denoising-diffusion-pytorch`&#x20;
* `CompVis/latent-diffusion`&#x20;


## Acknowledgements
Huge thanks to Phil Wang (`lucidrains`) and the CompVis team for open‑sourcing their pioneering diffusion model implementations.