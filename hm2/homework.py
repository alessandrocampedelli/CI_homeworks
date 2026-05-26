# %% [markdown]
# # Homework 1: End-to-End Reconstruction Before Generative Models
# 
# **Setup for Google Colab:**
# Run this cell to install dependencies and download the IPPy library.
# The dataset is assumed to be in the same folder as the notebook (`./Mayo`).

# %%
import os
import sys
import glob

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
        os.system('git clone https://github.com/alessandrocampedelli/CI_homeworks.git')
    sys.path.append('CI_homeworks')
    
    # Path to the dataset in fast local storage
    mayo_dataset_path = '/content/Mayo'
else:
    # Local setup
    sys.path.append('..')
    mayo_dataset_path = './Mayo'

# %%
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

# Import IPPy utilities and operators
from IPPy import operators, utilities
from IPPy.utilities import metrics

book_root = Path('.').resolve()

# Create directory for saving weights if it doesn't exist
if IN_COLAB:
    # Save weights directly to your Google Drive so you never lose them
    weights_dir = Path('/content/drive/MyDrive/homeworks_CI/weights')
else:
    weights_dir = book_root / 'weights'
weights_dir.mkdir(parents=True, exist_ok=True)

device = utilities.get_device()
torch.manual_seed(0)

print('Working device:', device)
print('Weights directory:', weights_dir)

# %% [markdown]
# ## Part 1: Data Pipeline and Synthetic Measurements

# %%
class MayoDataset(Dataset):
    def __init__(self, data_path, data_shape=256):
        super().__init__()
        # Search for all PNG files recursively in the given path
        self.fname_list = sorted(glob.glob(f'{data_path}/**/*.png', recursive=True))
        
        if len(self.fname_list) == 0:
            print(f"Warning: No PNG images found in {data_path}. Please check the path.")

        self.transform = transforms.Compose([
            transforms.ToTensor(), # Converts to [0.0, 1.0] and shape (C, H, W)
            transforms.Resize((data_shape, data_shape), antialias=True),
        ])

    def __len__(self):
        return len(self.fname_list)

    def __getitem__(self, idx):
        img_path = self.fname_list[idx]
        # Open image as grayscale
        img = Image.open(img_path).convert('L')
        img_tensor = self.transform(img)
        return img_tensor

# Build the training and test datasets, create the dataloaders
# The dataset has explicit 'train' and 'test' subfolders.
data_shape = 256
batch_size = 4

train_path = os.path.join(mayo_dataset_path, 'train')
test_path = os.path.join(mayo_dataset_path, 'test')

train_dataset = MayoDataset(train_path, data_shape=data_shape)
test_dataset = MayoDataset(test_path, data_shape=data_shape)

if len(train_dataset) > 0 and len(test_dataset) > 0:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
else:
    # Dummy loaders for demonstration if dataset path is invalid during local test
    train_loader = []
    test_loader = []

# Define the forward operator K: Motion Blur
K = operators.Blurring(
    img_shape=(data_shape, data_shape), 
    kernel_type='motion', 
    kernel_size=15, 
    motion_angle=45.0
)

# Visualize a clean / corrupted pair (if dataset is loaded)
if len(test_dataset) > 0:
    sample_clean = next(iter(test_loader)) # Shape: (1, 1, 256, 256)
    sample_corrupted_noiseless = K(sample_clean)
    # Add noise (using IPPy utility)
    noise_level = 0.05
    noise = utilities.gaussian_noise(sample_corrupted_noiseless, noise_level=noise_level)
    sample_corrupted = sample_corrupted_noiseless + noise

    utilities.show([sample_clean, sample_corrupted], title=["Clean Image", "Corrupted (Motion Blur + Noise)"])


# %% [markdown]
# ## Part 2: Reconstruction Networks

# %%
class SimpleCNN(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, n_filters=32, kernel_size=3):
        super().__init__()
        # Standard CNN processing directly mapping y -> x_hat
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, n_filters, kernel_size, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(n_filters, n_filters * 2, kernel_size, padding=1),
            nn.BatchNorm2d(n_filters * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(n_filters * 2, n_filters, kernel_size, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
            
            # Final mapping to output channels, no activation (or ReLU as discussed in lecture)
            nn.Conv2d(n_filters, out_ch, kernel_size=1) 
        )

    def forward(self, x):
        return self.net(x)

class ResCNN(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, n_filters=32, kernel_size=3):
        super().__init__()
        # Network learns the residual R. Final output is x_hat = y + R
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, n_filters, kernel_size, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(n_filters, n_filters * 2, kernel_size, padding=1),
            nn.BatchNorm2d(n_filters * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(n_filters * 2, n_filters, kernel_size, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
            
            # Predict residual. Tanh is common for residuals as they can be negative.
            nn.Conv2d(n_filters, out_ch, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, y):
        # Global residual connection: output = input + residual
        R = self.net(y)
        return y + R

class UNet(nn.Module):
    """Optional Extension: A simple baseline U-Net from scratch"""
    def __init__(self, in_ch=1, out_ch=1, features=[32, 64, 128]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        in_c = in_ch
        for feature in features:
            self.downs.append(self._block(in_c, feature))
            in_c = feature
            
        # Bottleneck
        self.bottleneck = self._block(features[-1], features[-1]*2)
        
        # Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(self._block(feature*2, feature))
            
        self.final_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)
        
    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        skip_connections = []
        
        # Down
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # Reverse for decoding
        
        # Up
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x) # Upconv
            skip_connection = skip_connections[i//2]
            
            # Handle rounding issues in spatial dims if necessary
            if x.shape != skip_connection.shape:
                import torch.nn.functional as F
                x = F.interpolate(x, size=skip_connection.shape[2:])
                
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](x) # Block
            
        return self.final_conv(x)


# Instantiate models
model_simple = SimpleCNN().to(device)
model_res = ResCNN().to(device)
model_unet = UNet().to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"SimpleCNN parameters: {count_parameters(model_simple)}")
print(f"ResCNN parameters: {count_parameters(model_res)}")
print(f"UNet parameters: {count_parameters(model_unet)}")

# Define paths for saving/loading weights
simple_weights_path = weights_dir / 'SimpleCNN.pth'
res_weights_path = weights_dir / 'ResCNN.pth'
unet_weights_path = weights_dir / 'UNet.pth'

# %% [markdown]
# ## Part 3: Training, Saving, and Evaluating the Models

# %%
def train_model(model, train_loader, K, weights_path, num_epochs=20, noise_level=0.05, lr=1e-3, resume=True):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []
    
    start_epoch = 0
    checkpoint_path = Path(str(weights_path) + ".checkpoint.pth")
    
    if resume and checkpoint_path.exists():
        print(f"Resuming {model.__class__.__name__} from checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint.get('loss_history', [])
        print(f"Resumed at epoch {start_epoch}/{num_epochs}")
    
    # If there's no data (e.g. invalid path), skip training
    if not train_loader:
        print("No training data found. Skipping training.")
        return history

    model.train()
    
    # Iterate over epochs
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        
        # Use tqdm on batches so progress is visible
        batch_iter = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] {model.__class__.__name__}")
        for batch_idx, clean_x in enumerate(batch_iter):
            clean_x = clean_x.to(device)
            
            # Online corruption
            corrupted_noiseless = K(clean_x)
            noise = utilities.gaussian_noise(corrupted_noiseless, noise_level=noise_level)
            corrupted_y = corrupted_noiseless + noise
            
            optimizer.zero_grad()
            
            # Forward pass
            x_pred = model(corrupted_y)
            
            # Loss against ground truth (clean)
            loss = loss_fn(x_pred, clean_x)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        history.append(avg_loss)
        
        # Save checkpoint at the end of each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': history
        }, checkpoint_path)
        
        # Also update the standard weights file
        torch.save(model.state_dict(), weights_path)
        
    return history

# Note: Training may take time on CPU. You might want to reduce num_epochs if strictly testing.
num_epochs = 20

# Train SimpleCNN
history_simple = train_model(model_simple, train_loader, K, simple_weights_path, num_epochs=num_epochs)

# Train ResCNN
history_res = train_model(model_res, train_loader, K, res_weights_path, num_epochs=num_epochs)

# Train UNet
history_unet = train_model(model_unet, train_loader, K, unet_weights_path, num_epochs=num_epochs)

# Plot training curves
if history_simple and history_res and history_unet:
    plt.figure(figsize=(10, 5))
    plt.plot(history_simple, label='SimpleCNN')
    plt.plot(history_res, label='ResCNN')
    plt.plot(history_unet, label='UNet')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

# %% [markdown]
# ## Part 4: Visual and Quantitative Comparison

# %%
# Reload weights before evaluation to ensure we evaluate the saved model
if simple_weights_path.exists(): model_simple.load_state_dict(torch.load(simple_weights_path, map_location=device))
if res_weights_path.exists(): model_res.load_state_dict(torch.load(res_weights_path, map_location=device))
if unet_weights_path.exists(): model_unet.load_state_dict(torch.load(unet_weights_path, map_location=device))

model_simple.eval()
model_res.eval()
model_unet.eval()

if len(test_loader) > 0:
    # Take one batch for visual comparison
    test_clean = next(iter(test_loader)).to(device)
    
    # Generate corrupted measurement
    test_corrupted_noiseless = K(test_clean)
    test_noise = utilities.gaussian_noise(test_corrupted_noiseless, noise_level=0.05)
    test_corrupted = test_corrupted_noiseless + test_noise
    
    with torch.no_grad():
        pred_simple = model_simple(test_corrupted)
        pred_res = model_res(test_corrupted)
        pred_unet = model_unet(test_corrupted)
        
    # Visual comparison using IPPy show function (ensure tensors are on CPU)
    images_to_show = [img.cpu().detach() for img in [test_clean, test_corrupted, pred_simple, pred_res, pred_unet]]
    titles = ["Ground Truth", "Corrupted Input", "SimpleCNN", "ResCNN", "UNet"]
    utilities.show(images_to_show, title=titles)
    
    # Quantitative comparison over the whole test set
    metrics_dict = {
        'SimpleCNN': {'MSE': 0.0, 'PSNR': 0.0, 'SSIM': 0.0},
        'ResCNN': {'MSE': 0.0, 'PSNR': 0.0, 'SSIM': 0.0},
        'UNet': {'MSE': 0.0, 'PSNR': 0.0, 'SSIM': 0.0}
    }
    
    loss_fn = nn.MSELoss()
    
    test_iter = tqdm(test_loader, desc="Evaluating on Test Set")
    for clean_x in test_iter:
        clean_x = clean_x.to(device)
        corr_noiseless = K(clean_x)
        noise = utilities.gaussian_noise(corr_noiseless, noise_level=0.05)
        corr_y = corr_noiseless + noise
        
        with torch.no_grad():
            out_s = model_simple(corr_y)
            out_r = model_res(corr_y)
            out_u = model_unet(corr_y)
            
            # Update metrics
            metrics_dict['SimpleCNN']['MSE'] += loss_fn(out_s, clean_x).item()
            metrics_dict['SimpleCNN']['PSNR'] += metrics.PSNR(out_s.cpu(), clean_x.cpu())
            metrics_dict['SimpleCNN']['SSIM'] += metrics.SSIM(out_s.cpu(), clean_x.cpu())
            
            metrics_dict['ResCNN']['MSE'] += loss_fn(out_r, clean_x).item()
            metrics_dict['ResCNN']['PSNR'] += metrics.PSNR(out_r.cpu(), clean_x.cpu())
            metrics_dict['ResCNN']['SSIM'] += metrics.SSIM(out_r.cpu(), clean_x.cpu())
            
            metrics_dict['UNet']['MSE'] += loss_fn(out_u, clean_x).item()
            metrics_dict['UNet']['PSNR'] += metrics.PSNR(out_u.cpu(), clean_x.cpu())
            metrics_dict['UNet']['SSIM'] += metrics.SSIM(out_u.cpu(), clean_x.cpu())
            
    n_test = len(test_loader)
    print("\n--- Quantitative Comparison ---")
    print(f"{'Model':<15} | {'MSE':<10} | {'PSNR (dB)':<10} | {'SSIM':<10}")
    print("-" * 55)
    for name, m in metrics_dict.items():
        print(f"{name:<15} | {m['MSE']/n_test:<10.6f} | {m['PSNR']/n_test:<10.2f} | {m['SSIM']/n_test:<10.4f}")


# %% [markdown]
# ## Deliverables and Discussion
# 
# **1. Which model performed better in your experiments, and why do you think that happened?**
# > UNet generally performs best. The motion blur operator creates non-local artifacts (blur spans across multiple pixels). A simple CNN struggles to resolve these because its receptive field is small. The UNet, through downsampling, expands its receptive field significantly, allowing it to capture and reverse the large-scale blur, while skip connections recover the high-frequency spatial details lost during pooling.
# 
# **2. Did the residual architecture help? If yes, in what sense?**
# > Yes, the ResCNN typically outperforms the SimpleCNN and is easier to optimize. Instead of forcing the network to memorize and output the entire clean image from scratch, the residual connection `x_hat = y + R` allows the network to focus solely on predicting the correction (the artifact/noise `R`). This simplifies the learning task, leading to faster convergence and better preservation of structures already present in the input.
# 
# **3. How did the noise level affect training and reconstruction quality?**
# > A higher noise level forces the network to learn a more robust, generalized mapping rather than memorizing the clean images (avoiding the "inverse crime"). However, if the noise level is too high, the network tends to output overly smooth/blurry reconstructions (since it is trained with MSE, which penalizes variance and thus favors the mean).
# 
# **4. Why is it important to generate the corruption through a known operator K instead of treating the problem as a generic image-to-image task?**
# > Generating the corruption online dynamically incorporates the physics of the measurement process (the forward model $K$). This ensures the network learns the specific inverse mapping $K^{-1}$ rather than just a generic stylistic mapping. It also provides infinite variations of noise realizations, preventing overfitting and ensuring the model generalizes well to real-world measurements that follow the same physical process.
# 
# **5. Why should one be cautious when evaluating pure end-to-end methods only through visual quality?**
# > End-to-end models, especially those trained with metrics like SSIM or GANs, can hallucinate realistic-looking structures (e.g., textures or anatomical features) that were not present in the original object. In medical imaging (like Mayo CTs), a visually pleasing but hallucinated reconstruction can lead to false diagnoses. This is why quantitative metrics, task-based evaluation, and expert visual inspection are crucial.
# 
# **6. If you implemented the optional UNet extension, how did it compare with the simpler CNN-based models?**
# > The UNet provided significantly sharper reconstructions and lower MSE than both SimpleCNN and ResCNN. Its encoder-decoder structure allows it to tackle the blur at a global scale (deep layers) while refining edges locally (skip connections). SimpleCNN and ResCNN, lacking this multi-scale processing, leave residual blur because they cannot gather enough spatial context to invert large kernels.
