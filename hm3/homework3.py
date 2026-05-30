import os
import sys
import glob
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

# Colab Setup: Install required package and clone repo for IPPy
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("Configuring Google Colab environment...")
    
    # Mount Google Drive to access the dataset
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Unzip the dataset from Drive to Colab's fast local storage
    if not os.path.exists('/content/Mayo'):
        print("Extracting dataset from Google Drive...")
        os.system('unzip -q /content/drive/MyDrive/homeworks_CI/Mayo.zip -d /content/')
    
    # Install dependencies and clone repo
    os.system('pip install astra-toolbox')
    if not os.path.exists('CI_homeworks'):
        os.system('git clone -b hm3 https://github.com/alessandrocampedelli/CI_homeworks.git')
    sys.path.append('CI_homeworks')
    
    # Path to the dataset in fast local storage
    mayo_dataset_path = '/content/Mayo'
    book_root = Path('CI_homeworks').resolve()
    weights_dir = Path('/content/drive/MyDrive/homeworks_CI/weights2')
else:
    # Local setup
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    book_root = Path(__file__).resolve().parent.parent
    mayo_dataset_path = str(book_root / 'hm2' / 'Mayo')
    weights_dir = book_root / 'weights2'
    
weights_dir.mkdir(parents=True, exist_ok=True)

from IPPy import operators, utilities
from IPPy.utilities import gaussian_noise, get_device
from IPPy.utilities import metrics
from IPPy.nn.diffusion import DiffusionUNet, cosine_beta_schedule, extract, denormalize_to_01
from IPPy.nn.vae import ConvVAE

device = get_device()
torch.manual_seed(0)

print('Working device:', device)
print('Weights directory:', weights_dir)

# ==========================================
# Part 1: Data Pipeline
# ==========================================
class MayoDataset(Dataset):
    def __init__(self, data_path, data_shape=64):
        super().__init__()
        self.fname_list = sorted(glob.glob(f'{data_path}/**/*.png', recursive=True))
        
        self.transform = transforms.Compose([
            transforms.Resize((data_shape, data_shape), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)), # Normalize to [-1, 1] for VAE/Diffusion
        ])

    def __len__(self):
        return len(self.fname_list)

    def __getitem__(self, idx):
        img_path = self.fname_list[idx]
        img = Image.open(img_path).convert('L')
        img_tensor = self.transform(img)
        return img_tensor

data_shape = 64
mayo_test_path = book_root / 'hm2' / 'Mayo' / 'test'
test_dataset = MayoDataset(mayo_test_path, data_shape=data_shape)

if len(test_dataset) > 0:
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
else:
    print("Warning: Test dataset is empty or not found.")
    test_loader = []

# Forward Operator K: Motion Blur
K = operators.Blurring(
    img_shape=(data_shape, data_shape), 
    kernel_type='motion', 
    kernel_size=15, 
    motion_angle=45.0
)
noise_level = 0.05

# Load models
vae_model = ConvVAE(latent_dim=128).to(device)
vae_weights_path = weights_dir / 'VAE.pth'
if vae_weights_path.exists():
    vae_model.load_state_dict(torch.load(vae_weights_path, map_location=device))
    print("Loaded VAE weights successfully.")
else:
    print(f"Warning: VAE weights not found at {vae_weights_path}. Prior will not work optimally.")
vae_model.eval()

diffusion_model = DiffusionUNet(
    in_ch=1,
    base_ch=64,
    channel_mults=(1, 2, 4),
    time_dim=256,
    dropout=0.05,
    attn_levels=(1, 2),
).to(device)
diff_weights_path = weights_dir / 'DDPMDenoiser.pth'
if diff_weights_path.exists():
    # If the user saved a state dict
    diffusion_model.load_state_dict(torch.load(diff_weights_path, map_location=device))
    print("Loaded Diffusion weights successfully.")
else:
    print(f"Warning: Diffusion weights not found at {diff_weights_path}. Prior will not work optimally.")
diffusion_model.eval()

# ==========================================
# Part 2: Latent Reconstruction with VAE
# ==========================================
def latent_objective(z, generator, y_delta, K, lam=1e-3):
    # Decode z
    x_gen = generator(z)
    
    # Forward pass
    y_gen = K(x_gen)
    
    # Data fidelity
    data_fidelity = torch.nn.functional.mse_loss(y_gen, y_delta)
    
    # Latent regularization (L2)
    latent_reg = torch.sum(z ** 2) * lam
    
    return data_fidelity + latent_reg

def reconstruct_with_vae_prior(generator, y_delta, K, latent_dim, num_steps=500, lr=1e-2, lam=1e-3):
    z = torch.randn(1, latent_dim, device=device, requires_grad=True)
    
    optimizer = torch.optim.Adam([z], lr=lr)
    
    pbar = tqdm(range(num_steps), desc="VAE Optimization")
    for step in pbar:
        optimizer.zero_grad()
        loss = latent_objective(z, generator, y_delta, K, lam=lam)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({'loss': loss.item()})
        
    with torch.no_grad():
        x_final = generator(z)
    return x_final

# ==========================================
# Part 3: Reconstruction with Diffusion (DPS)
# ==========================================
# Precompute diffusion schedule
num_diffusion_timesteps = 1000
betas = cosine_beta_schedule(num_diffusion_timesteps).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

def predict_x0_from_eps(x_t, eps_pred, t, alpha_bars):
    alpha_bar_t = extract(alpha_bars, t, x_t.shape)
    x0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
    return x0_pred

def reconstruct_with_diffusion_prior(model, y_delta, K, alpha_bars, num_steps=1000, guidance_scale=0.1):
    # We will run a DDPM or DDIM-like loop. Let's do standard DDPM with DPS guidance
    x = torch.randn_like(y_delta).to(device) # start from pure noise
    
    # For a quicker DPS, you could use DDIM, but we stick to the course's DDPM for simplicity
    # if `num_steps` < 1000, it would require a strided schedule
    
    pbar = tqdm(reversed(range(0, num_steps)), desc="DPS Diffusion Loop", total=num_steps)
    for i in pbar:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        
        # 1. Require grad on x_t for DPS
        x_t = x.detach().requires_grad_(True)
        
        # 2. Predict noise
        eps_pred = model(x_t, t)
        
        # 3. Predict x0
        x0_pred = predict_x0_from_eps(x_t, eps_pred, t, alpha_bars)
        
        # 4. Data consistency guidance
        y_pred = K(x0_pred)
        data_loss = torch.nn.functional.mse_loss(y_pred, y_delta)
        
        grad_x_t = torch.autograd.grad(outputs=data_loss, inputs=x_t)[0]
        
        # 5. DDPM update (standard reverse step)
        alpha_t = extract(alphas, t, x_t.shape)
        alpha_bar_t = extract(alpha_bars, t, x_t.shape)
        
        if i > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)
            
        # Simplified DDPM update formula
        beta_t = extract(betas, t, x_t.shape)
        model_mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred)
        x_prev = model_mean + torch.sqrt(beta_t) * noise
        
        # 6. Apply DPS guidance
        # Scale guidance depending on the noise level or just fixed
        x = x_prev - guidance_scale * grad_x_t
        x = x.detach()

    return x

# ==========================================
# Execution and Evaluation
# ==========================================
if __name__ == '__main__':
    if len(test_loader) > 0:
        clean_x = next(iter(test_loader)).to(device)
        
        # Generate measurements
        y_noiseless = K(clean_x)
        noise = gaussian_noise(y_noiseless, noise_level=noise_level)
        y_delta = y_noiseless + noise
        
        # 1. Latent Prior Reconstruction
        print("\n--- Running VAE Latent Prior Reconstruction ---")
        x_vae = reconstruct_with_vae_prior(vae_model.decode, y_delta, K, latent_dim=128, num_steps=500, lam=1e-3)
        
        # 2. Diffusion Prior Reconstruction
        print("\n--- Running Diffusion DPS Reconstruction ---")
        # We can use a subset of steps for speed if we define a strided schedule, but for accuracy we use full 1000
        x_diff = reconstruct_with_diffusion_prior(diffusion_model, y_delta, K, alphas_cumprod, num_steps=1000, guidance_scale=0.5)
        
        # Normalize back to 0-1 for visualization and metrics
        clean_x_01 = denormalize_to_01(clean_x)
        y_delta_01 = denormalize_to_01(y_delta)
        x_vae_01 = denormalize_to_01(x_vae)
        x_diff_01 = denormalize_to_01(x_diff)
        
        # Compare
        print("\n--- Quantitative Comparison ---")
        print(f"{'Method':<15} | {'MSE':<10} | {'PSNR (dB)':<10} | {'SSIM':<10}")
        print("-" * 55)
        
        def compute_metrics(name, pred, target):
            mse = torch.nn.functional.mse_loss(pred, target).item()
            psnr = metrics.PSNR(pred.cpu(), target.cpu())
            ssim = metrics.SSIM(pred.cpu(), target.cpu())
            print(f"{name:<15} | {mse:<10.6f} | {psnr:<10.2f} | {ssim:<10.4f}")
            
        compute_metrics('Corrupted', y_delta_01, clean_x_01)
        compute_metrics('VAE Prior', x_vae_01, clean_x_01)
        compute_metrics('Diffusion Prior', x_diff_01, clean_x_01)
        
        # Visualize
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(clean_x_01.squeeze().cpu().numpy(), cmap='gray')
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        
        axes[1].imshow(y_delta_01.squeeze().cpu().numpy(), cmap='gray')
        axes[1].set_title('Corrupted (Blur + Noise)')
        axes[1].axis('off')
        
        axes[2].imshow(x_vae_01.squeeze().cpu().numpy(), cmap='gray')
        axes[2].set_title('VAE Prior Recon')
        axes[2].axis('off')
        
        axes[3].imshow(x_diff_01.squeeze().cpu().numpy(), cmap='gray')
        axes[3].set_title('Diffusion Prior Recon')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(book_root / 'hm3' / 'homework3_reconstruction.png')
        print(f"Saved reconstruction figure to {book_root / 'hm3' / 'homework3_reconstruction.png'}")
