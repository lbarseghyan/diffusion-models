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
    '../cifar-10/all_images',
    train_batch_size = 64,
    train_lr = 2e-4,
    train_num_steps = 800000,           
    calculate_fid = False,              
    save_and_sample_every = 5000,
    # num_fid_samples = 5000              # CHANGE  
    results_folder = 'results/800k_steps_fid_samples_false_07_03_25' 
)


if __name__ == '__main__':
    # Place all code that starts new processes or spawns workers here
    trainer.train()  