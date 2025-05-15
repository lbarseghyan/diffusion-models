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
