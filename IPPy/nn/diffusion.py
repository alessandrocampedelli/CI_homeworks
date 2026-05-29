import math
import torch
from torch import nn
import torch.nn.functional as F

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def denormalize_to_01(x):
    return (x + 1) * 0.5

class EMA:
    def __init__(self, model, decay=0.9995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

# Standard building blocks for Diffusion UNet
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch * 2)) if time_emb_dim else None
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.conv1(F.silu(self.norm1(x)))
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb[(..., ) + (None, ) * 2]
            scale, shift = time_emb.chunk(2, dim=1)
            h = h * (scale + 1) + shift
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, h * w), qkv)
        q = q * self.scale
        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h d j -> b h d i', attn, v)
        out = out.reshape(b, -1, h, w)
        return self.to_out(out) + x

class DiffusionUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, channel_mults=(1, 2, 4), time_dim=256, dropout=0.05, attn_levels=(1, 2)):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        ch = base_ch
        in_ch_list = [base_ch]
        
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        
        # Downs
        for level, mult in enumerate(channel_mults):
            out_ch = base_ch * mult
            is_attn = level in attn_levels
            self.downs.append(nn.ModuleList([
                Block(ch, out_ch, time_dim, dropout),
                Block(out_ch, out_ch, time_dim, dropout),
                Attention(out_ch) if is_attn else nn.Identity(),
                nn.Conv2d(out_ch, out_ch, 4, 2, 1) if level != len(channel_mults) - 1 else nn.Identity()
            ]))
            ch = out_ch
            in_ch_list.append(ch)
            
        self.mid_block1 = Block(ch, ch, time_dim, dropout)
        self.mid_attn = Attention(ch)
        self.mid_block2 = Block(ch, ch, time_dim, dropout)
        
        # Ups
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_ch * mult
            is_attn = level in attn_levels
            self.ups.append(nn.ModuleList([
                Block(ch + in_ch_list.pop(), out_ch, time_dim, dropout),
                Block(out_ch, out_ch, time_dim, dropout),
                Attention(out_ch) if is_attn else nn.Identity(),
                nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1) if level != 0 else nn.Identity()
            ]))
            ch = out_ch
            
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_ch, 3, padding=1)
        )
        
    def forward(self, x, time):
        t = self.time_mlp(time)
        x = self.init_conv(x)
        h = []
        
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
            
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
            
        return self.final_conv(x)
