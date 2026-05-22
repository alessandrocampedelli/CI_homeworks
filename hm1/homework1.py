# %% [markdown]
# # Homework 1 - Model Based Reconstructions
# 
# In this notebook, we evaluate 4 different image reconstruction tasks using model-based regularization techniques (Tikhonov, TV, and TpV).
# We use the `IPPy` library for defining operators and solvers.
# The 4 tasks are:
# 1. Denoise
# 2. Deblur
# 3. Super Resolution
# 4. CT Image Reconstruction

# %%
import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from IPPy import operators, solvers, utilities

# Use CPU for simplicity, or CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Disable gradient tracking globally since we use iterative solvers, not backprop.
# This prevents IPPy from crashing when calling .numpy() on tensors.
torch.set_grad_enabled(False)

def load_image(path, size=None):
    img = Image.open(path).convert('L') # Grayscale
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    x = T.ToTensor()(img).unsqueeze(0).to(device) # Shape: (1, 1, H, W)
    return x

def show_results(x_true, y_noisy, recons, titles, figsize=(20, 5)):
    n = len(recons) + 2
    fig, axes = plt.subplots(1, n, figsize=figsize)
    axes[0].imshow(x_true.squeeze().cpu().numpy(), cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(y_noisy.squeeze().cpu().numpy(), cmap='gray')
    axes[1].set_title('Degraded & Noisy')
    axes[1].axis('off')
    
    for i, (recon, title) in enumerate(zip(recons, titles)):
        axes[i+2].imshow(recon.squeeze().cpu().numpy(), cmap='gray')
        axes[i+2].set_title(title)
        axes[i+2].axis('off')
    plt.tight_layout()
    plt.show()

def plot_metrics_history(info_dicts, titles, metric='PSNR'):
    plt.figure(figsize=(8, 5))
    for info, title in zip(info_dicts, titles):
        if metric in info:
            y_vals = info[metric]
            if torch.is_tensor(y_vals):
                y_vals = y_vals.cpu().numpy().flatten()
            plt.plot(range(len(y_vals)), y_vals, label=title)
    plt.title(f'{metric} over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.show()

def print_metrics_table(task_name, metrics_dict):
    print(f"\n--- Final Metrics for {task_name} ---")
    print(f"{'Method':<15} | {'PSNR (dB)':<10} | {'SSIM':<10} | {'Rel. Error':<10}")
    print("-" * 55)
    for method, info in metrics_dict.items():
        psnr_val = float(info['PSNR'][-1].item()) if torch.is_tensor(info['PSNR'][-1]) else info['PSNR'][-1]
        ssim_val = float(info['SSIM'][-1].item()) if torch.is_tensor(info['SSIM'][-1]) else info['SSIM'][-1]
        re_val = float(info['RE'][-1].item()) if torch.is_tensor(info['RE'][-1]) else info['RE'][-1]
        print(f"{method:<15} | {psnr_val:<10.2f} | {ssim_val:<10.4f} | {re_val:<10.4f}")

# Grid search for best lambda w.r.t PSNR
def grid_search_lambda(solver, y, x_true, lambdas, p=None, maxiter=50):
    best_psnr = -1
    best_lam = None
    best_x = None
    best_info = None
    
    # Starting point for solver (same shape as x_true)
    x0 = torch.zeros_like(x_true)
    
    for lam in lambdas:
        if p is not None:
            # TpV
            x_est, info = solver(y, x_true, x0, lmbda=lam, p=p, maxiter=maxiter)
        elif isinstance(solver, solvers.CGLS):
            x_est, info = solver(y, x_true, x0, lam=lam, maxiter=maxiter)
        else:
            # TV
            x_est, info = solver(y, x_true, x0, lmbda=lam, maxiter=maxiter)
        
        psnr = info['PSNR'][-1]
        if torch.is_tensor(psnr):
            psnr = float(psnr.item())
        
        if psnr > best_psnr:
            best_psnr = psnr
            best_lam = lam
            best_x = x_est
            best_info = info
    
    print(f"Best lambda found: {best_lam} (PSNR: {best_psnr:.2f})")
    return best_x, best_info

# %% [markdown]
# ## 1. Denoising
# 
# We use an Identity operator and add Gaussian noise.

# %%
x1 = load_image('assets/img1.jpeg', size=256)
op_id = operators.Identity(x1.shape[2:])

y1_clean = op_id(x1)
noise1 = utilities.gaussian_noise(y1_clean, noise_level=0.1)
y1 = y1_clean + noise1

print("Running Denoising solvers...")
lambdas = [0.01, 0.05, 0.1, 0.5, 1.0]

# Tikhonov
print("CGLS (Tikhonov):")
solver_cgls_1 = solvers.CGLS(op_id)
x1_tikhonov, info1_tikhonov = grid_search_lambda(solver_cgls_1, y1, x1, lambdas)

# TV
print("SGP (TV):")
solver_sgp_1 = solvers.SGP(op_id)
x1_tv, info1_tv = grid_search_lambda(solver_sgp_1, y1, x1, lambdas)

# TpV (p=0.5)
print("Chambolle-Pock (TpV p=0.5):")
solver_tpv_1 = solvers.ChambollePockTpVUnconstrained(op_id)
x1_tpv, info1_tpv = grid_search_lambda(solver_tpv_1, y1, x1, lambdas, p=0.5)

metrics_denoise = {
    "Tikhonov": info1_tikhonov,
    "TV": info1_tv,
    "TpV (p=0.5)": info1_tpv
}

print_metrics_table("Denoising", metrics_denoise)
plot_metrics_history([info1_tikhonov, info1_tv, info1_tpv], ["Tikhonov", "TV", "TpV"], metric='PSNR')
show_results(x1, y1, [x1_tikhonov, x1_tv, x1_tpv], ["Tikhonov", "TV", "TpV (p=0.5)"])

# %% [markdown]
# ## 2. Deblurring
# 
# We use a Gaussian Blurring operator and add a smaller amount of noise.

# %%
x2 = load_image('assets/img2.jpeg', size=256)
op_blur = operators.Blurring(x2.shape[2:], kernel_type='gaussian', kernel_size=7, kernel_variance=2.0)

y2_clean = op_blur(x2)
noise2 = utilities.gaussian_noise(y2_clean, noise_level=0.02)
y2 = y2_clean + noise2

print("Running Deblurring solvers...")
lambdas = [0.001, 0.005, 0.01, 0.05]

print("CGLS (Tikhonov):")
solver_cgls_2 = solvers.CGLS(op_blur)
x2_tikhonov, info2_tikhonov = grid_search_lambda(solver_cgls_2, y2, x2, lambdas)

print("SGP (TV):")
solver_sgp_2 = solvers.SGP(op_blur)
x2_tv, info2_tv = grid_search_lambda(solver_sgp_2, y2, x2, lambdas)

print("Chambolle-Pock (TpV p=0.8):")
solver_tpv_2 = solvers.ChambollePockTpVUnconstrained(op_blur)
x2_tpv, info2_tpv = grid_search_lambda(solver_tpv_2, y2, x2, lambdas, p=0.8)

metrics_deblur = {
    "Tikhonov": info2_tikhonov,
    "TV": info2_tv,
    "TpV (p=0.8)": info2_tpv
}

print_metrics_table("Deblurring", metrics_deblur)
plot_metrics_history([info2_tikhonov, info2_tv, info2_tpv], ["Tikhonov", "TV", "TpV"], metric='PSNR')
show_results(x2, y2, [x2_tikhonov, x2_tv, x2_tpv], ["Tikhonov", "TV", "TpV (p=0.8)"])

# %% [markdown]
# ## 3. Super Resolution
# 
# We use an averaging DownScaling operator by a factor of 2.

# %%
x3 = load_image('assets/img3.jpeg', size=256)
op_down = operators.DownScaling(x3.shape[2:], downscale_factor=2, mode='avg')

y3_clean = op_down(x3)
noise3 = utilities.gaussian_noise(y3_clean, noise_level=0.01)
y3 = y3_clean + noise3

print("Running Super Resolution solvers...")
lambdas = [0.001, 0.005, 0.01, 0.05]

print("CGLS (Tikhonov):")
solver_cgls_3 = solvers.CGLS(op_down)
x3_tikhonov, info3_tikhonov = grid_search_lambda(solver_cgls_3, y3, x3, lambdas)

print("SGP (TV):")
solver_sgp_3 = solvers.SGP(op_down)
x3_tv, info3_tv = grid_search_lambda(solver_sgp_3, y3, x3, lambdas)

print("Chambolle-Pock (TpV p=0.5):")
solver_tpv_3 = solvers.ChambollePockTpVUnconstrained(op_down)
x3_tpv, info3_tpv = grid_search_lambda(solver_tpv_3, y3, x3, lambdas, p=0.5)

metrics_sr = {
    "Tikhonov": info3_tikhonov,
    "TV": info3_tv,
    "TpV (p=0.5)": info3_tpv
}

print_metrics_table("Super Resolution", metrics_sr)
plot_metrics_history([info3_tikhonov, info3_tv, info3_tpv], ["Tikhonov", "TV", "TpV"], metric='PSNR')
y3_display = torch.nn.functional.interpolate(y3, scale_factor=2, mode='nearest')
show_results(x3, y3_display, [x3_tikhonov, x3_tv, x3_tpv], ["Tikhonov", "TV", "TpV (p=0.5)"])

# %% [markdown]
# ## 4. CT Image Reconstruction
# 
# We simulate parallel-beam projections (Radon transform).

# %%
x4 = load_image('assets/img4.jpeg', size=128) # Smaller size for CT to speed up
angles = np.linspace(0, np.pi, 90, endpoint=False) # 90 views

try:
    op_ct = operators.CTProjector(x4.shape[2:], angles=angles, geometry='parallel')
    has_ct = True
except Exception as e:
    print(f"CT Projector failed: {e}")
    has_ct = False

if has_ct:
    y4_clean = op_ct(x4)
    noise4 = utilities.gaussian_noise(y4_clean, noise_level=0.05)
    y4 = y4_clean + noise4
    
    print("Running CT Reconstruction solvers...")
    lambdas = [0.005, 0.01, 0.05, 0.1]
    
    print("CGLS (Tikhonov):")
    solver_cgls_4 = solvers.CGLS(op_ct)
    x4_tikhonov, info4_tikhonov = grid_search_lambda(solver_cgls_4, y4, x4, lambdas)
    
    print("SGP (TV):")
    solver_sgp_4 = solvers.SGP(op_ct)
    x4_tv, info4_tv = grid_search_lambda(solver_sgp_4, y4, x4, lambdas)
    
    print("Chambolle-Pock (TpV p=0.5):")
    solver_tpv_4 = solvers.ChambollePockTpVUnconstrained(op_ct)
    x4_tpv, info4_tpv = grid_search_lambda(solver_tpv_4, y4, x4, lambdas, p=0.5)
    
    metrics_ct = {
        "Tikhonov": info4_tikhonov,
        "TV": info4_tv,
        "TpV (p=0.5)": info4_tpv
    }
    
    print_metrics_table("CT Reconstruction", metrics_ct)
    plot_metrics_history([info4_tikhonov, info4_tv, info4_tpv], ["Tikhonov", "TV", "TpV"], metric='PSNR')
    
    # y4 is a sinogram, we shouldn't plot it in the same space as the image. 
    # Let's show FBP instead of y4 if FBP is available.
    solver_fbp = solvers.FBP(op_ct)
    x4_fbp, _ = solver_fbp(y4, x4, torch.zeros_like(x4))
    
    show_results(x4, x4_fbp, [x4_tikhonov, x4_tv, x4_tpv], ["Tikhonov", "TV", "TpV (p=0.5)"])

# %% [markdown]
# ## Discussion and Conclusions
# 
# - **Denoising**: Tikhonov often smooths out textures, while TV perfectly preserves edges and piecewise constant areas but may introduce a "staircase" effect. TpV with $p<1$ enhances edge preservation even more strongly, promoting sparser gradients.
# - **Deblurring**: TV and TpV generally reconstruct sharper edges than Tikhonov, which tends to leave some residual blur or ringing artifacts depending on the $\lambda$.
# - **Super Resolution**: Tikhonov (L2 penalty) can act like a bicubic interpolation, while TV explicitly searches for piecewise flat solutions, producing sharper boundaries.
# - **CT Reconstruction**: Sparse views heavily penalize simple least squares (Tikhonov). TV shines in missing data problems (like few-view CT) by minimizing total variation, successfully recovering piece-wise constant anatomies.
