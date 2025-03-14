from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 128,
    dim_mults = (1, 2, 2, 2),
    dropout = 0.1,
    # flash_attn = True        default=False
)


diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    timesteps = 1000,           # number of steps
    hybrid_loss = True
)


trainer = Trainer(
    diffusion,
    '../data/cifar-10/train_images',
    train_batch_size = 64,
    train_lr = 2e-4,
    train_num_steps = 800000,         # total training steps,
    # gradient_accumulate_every = 2,    # gradient accumulation steps,  default = 1
    # ema_decay = 0.995,                # exponential moving average decay
    # amp = True,                       # turn on mixed precision,   default = False
    calculate_fid = False              # whether to calculate fid during training
)


if __name__ == '__main__':
    # Place all code that starts new processes or spawns workers here
    trainer.train()  