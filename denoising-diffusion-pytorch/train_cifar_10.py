from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    dropout = 0.1,
)


diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    timesteps = 1000,           # number of steps
)


trainer = Trainer(
    diffusion,
    '../data/cifar-10/train_images',
    train_batch_size = 64,
    train_lr = 2e-4,
    train_num_steps = 800000,           
    calculate_fid = True,              
    save_and_sample_every = 5000,
    num_fid_samples = 1000             
)


if __name__ == '__main__':
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print()
    trainer.train()  