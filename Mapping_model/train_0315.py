# -*- coding: utf-8 -*-
"""
Three-stage training (Stage1/2/3) with:
- VAE decode/encode bridge in pixel domain
- Optional differentiable channel attacks
- NAFNet denoiser inserted before RevealNet (Stage3 fine-tuning)
- Three-stage freezing strategy
- Stage3: learnable orthogonal matrix Q

【修改说明】
- Stage1 / Stage2 完全保持不变（包括输出格式）
- Stage3 中：使用“多专家 + 软加权”去噪网络（噪声分类器 + 5 个 NAFNet 专家）
- Stage3 中联合微调：RevealNet + Q + 多专家去噪网络（分类器 + 专家）
- 损失函数修改：移除 LPIPS 和 L1，改为计算像素域转回 Latent 域分布的 KL 散度
- 清理冗余：移除单一 jpegdiff 信道模式及相关质量调度参数
"""

import os
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import swanlab  # 替换wandb为swanlab

from diffusers import AutoencoderKL
from Denoising_net import build_default_nafnet_4x64, NAFNet
from DiffAttack import DifferentiableGaussianBlur, DifferentiableResizeRecover, DifferentiableAWGN, DifferentiableJPEG, MixedAttack1_GBlur_Resize_AWGN, MixedAttack2_GBlur_Resize_JPEG, MixedAttack3_GBlur_AWGN_JPEG, MixedAttack4_Resize_AWGN_JPEG

import torch.optim.lr_scheduler as lr_scheduler
from noise_classifier import ResNet34_Noise_Classifier  # 多专家分类器

# ===================== Basic Config =====================
SWANLAB_PROJECT = "stega-vae_diffusion-denoising-SingleAttack"
SWANLAB_WORKSPACE = None  

MODEL_ID = "/home/ygf/stable-diffusion-2-1-base"
COVER_FOLDER = "/home/ygf/ICME2026_new/output_1205/cover/pt/mean"

STAGE1_MODEL_PATH = "best_stega_unet_stage1_denoising_random_attack_ori.pt"
STAGE2_MODEL_PATH = "best_stega_unet_stage2_denoising_random_attack_ori_0105.pt"
STAGE3_MODEL_PATH = "best_stega_unet_stage3_denoising_random_attack_ori_withQ_fine_0123.pt"

# ===================== MoE 多专家去噪配置 =====================
CLASSIFIER_CKPT_PATH = "/home/ygf/First_paper/Denoising_model/checkpoints/noise_cls_resnet34_best.pt"  # 噪声分类器权重
DENOISER_CKPT_DIR = "/home/ygf/First_paper/Denoising_model/checkpoints"                                # 5 个专家 NAFNet 权重目录
DENOISER_CKPT_PREFIX = "nafnet_4x64_"                                                                  # nafnet_4x64_AWGN.pt 等
NOISE_LIST = ["AWGN", "GBLUR", "JPEG", "MBLUR", "RESIZE"]

# ===================== Device =====================
device_global = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
set_seed(42)

# ===================== Load VAE & Losses =====================
vae = AutoencoderKL.from_pretrained(
    MODEL_ID,
    subfolder="vae",
    torch_dtype=torch.float32
).to(device_global)
vae.eval()
for p in vae.parameters():
    p.requires_grad = False
print("VAE loaded")


# ===================== Initialize Differentiable Attacks =====================
jpegdiff = DifferentiableJPEG(quality=70).to(device_global)
gblur = DifferentiableGaussianBlur(kernel_size=3, sigma=1.0).to(device_global)
resize_recover = DifferentiableResizeRecover(scale_factor=1.0, mode='bilinear').to(device_global)
awgn = DifferentiableAWGN(sigma=0.05, clip_range=(0.0, 1.0)).to(device_global)

ATTACKS = {
    "jpegdiff": lambda rgb: jpegdiff(rgb),
    "gblur": lambda rgb: gblur(rgb),
    "resize_recover": lambda rgb: resize_recover(rgb),
    "awgn": lambda rgb: awgn(rgb),
}
ATTACK_NAMES = list(ATTACKS.keys())

for attack in [jpegdiff, gblur, resize_recover, awgn]:
    attack.eval()

print(f"Attacks initialized: {ATTACK_NAMES}")


def randomize_attack_params():
    """Per-batch attack parameter randomization"""
    # JPEG quality
    jpeg_q = random.choice([90, 70, 50])
    jpegdiff.quality = int(jpeg_q)

    # Gaussian blur kernel/sigma linkage
    gblur_kernel = random.choice([3, 5, 7])
    gblur_sigma = {3: 1.0, 5: 1.5, 7: 2.0}[gblur_kernel]
    gblur.update_params(kernel_size=gblur_kernel, sigma=gblur_sigma)

    # Resize factors
    resize_factor = random.choice([1.5, 1.25, 0.75, 0.5])
    resize_recover.update_scale(scale_factor=resize_factor)

    # AWGN sigma
    awgn_sigma = random.choice([0.01, 0.05, 0.1])
    awgn.update_sigma(sigma=awgn_sigma)

    return {
        "jpeg_quality": float(jpeg_q),
        "gblur_kernel": int(gblur_kernel),
        "gblur_sigma": float(gblur_sigma),
        "resize_factor": float(resize_factor),
        "awgn_sigma": float(awgn_sigma),
    }


# ===================== Load NAFNet Denoiser =====================
def load_nafnet_model(ckpt_path: str, width: int = 32, device: str = "cpu") -> NAFNet:
    model = build_default_nafnet_4x64(width=width)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded NAFNet from {ckpt_path}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"NAFNet params: {total_params/1e6:.3f} M")
    return model


def infer_nafnet(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    is_training: bool = True
) -> torch.Tensor:
    """直接推理，支持训练时的梯度传播"""
    model.train() if is_training else model.eval()
    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        output = model(input_tensor)
    return output


# ===================== 多专家软加权 MoE 去噪模块 =====================
class MultiExpertDenoiser(nn.Module):
    """
    多专家去噪模块（MoE，软加权）
    - 包含：
        1) 噪声分类器 ResNet34_Noise_Classifier
        2) 5 个 NAFNet 专家（AWGN / GBLUR / JPEG / MBLUR / RESIZE）
    - 前向：
        confs = classifier(x)  # (B,5)
        outs_i = expert_i(x)   # (B,4,64,64)
        fused = Σ_i confs[:,i] * outs_i
    """
    def __init__(
        self,
        classifier_ckpt: str,
        denoiser_ckpt_dir: str,
        denoiser_prefix: str,
        noise_list: list[str],
        device: str = "cuda",
    ):
        super().__init__()
        self.noise_list = noise_list

        # 1) 分类器
        self.classifier = ResNet34_Noise_Classifier(
            num_classes=len(noise_list),
            in_channels=4
        )
        assert os.path.exists(classifier_ckpt), f"❌ 分类器权重不存在: {classifier_ckpt}"
        ckpt_cls = torch.load(classifier_ckpt, map_location=device)
        self.classifier.load_state_dict(ckpt_cls["model_state_dict"], strict=True)

        # 2) 专家 NAFNet
        experts = {}
        for nt in noise_list:
            ckpt_path = os.path.join(denoiser_ckpt_dir, f"{denoiser_prefix}{nt}.pt")
            assert os.path.exists(ckpt_path), f"❌ 专家 {nt} 权重不存在: {ckpt_path}"

            model = build_default_nafnet_4x64(width=32)
            ckpt_naf = torch.load(ckpt_path, map_location=device)
            state_dict = ckpt_naf["model"] if "model" in ckpt_naf else ckpt_naf
            model.load_state_dict(state_dict, strict=True)
            experts[nt] = model

        self.experts = nn.ModuleDict(experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,4,64,64) 或 (4,64,64)
        返回同形状的去噪结果，支持梯度回传（分类器 + 专家）
        """
        is_single = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            is_single = True

        B = x.shape[0]
        device = next(self.parameters()).device
        x = x.to(device)

        # 分类器 softmax 权重 (B,5)
        confs, _ = self.classifier(x)  # confs 已经是 softmax
        # 专家输出 (B,5,4,64,64)
        expert_outs = []
        for nt in self.noise_list:
            out_nt = self.experts[nt](x)
            expert_outs.append(out_nt)
        expert_outs = torch.stack(expert_outs, dim=1)

        # 软加权融合
        fused = torch.zeros_like(x)
        for b in range(B):
            w = confs[b].view(-1, 1, 1, 1)  # (5,1,1,1)
            fused[b] = torch.sum(expert_outs[b] * w, dim=0)

        if is_single:
            fused = fused.squeeze(0)
        return fused


# 全局 MoE 句柄（仅 Stage3 使用）
MoE_DENOISER: MultiExpertDenoiser | None = None


# ===================== Orthogonal Matrix Generation =====================
def generate_orthogonal_matrix(n: int = 64, seed: int | None = None, device: str = "cuda", dtype=torch.float32) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    A = torch.randn((n, n), dtype=dtype, device=device)
    Q, R = torch.linalg.qr(A)
    d = torch.sign(torch.diag(R))
    d[d == 0] = 1.0
    Q = Q * d.unsqueeze(0)
    return Q


# ===================== Diffusion-style UNet blocks =====================
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0, num_groups=32):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, ch, num_heads=4, num_groups=32):
        super().__init__()
        assert ch % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = ch // num_heads

        self.norm = nn.GroupNorm(num_groups, ch)
        self.q = nn.Conv2d(ch, ch, kernel_size=1)
        self.k = nn.Conv2d(ch, ch, kernel_size=1)
        self.v = nn.Conv2d(ch, ch, kernel_size=1)
        self.proj = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        h_in = self.norm(x)

        q = self.q(h_in).view(b, self.num_heads, self.head_dim, h * w)
        k = self.k(h_in).view(b, self.num_heads, self.head_dim, h * w)
        v = self.v(h_in).view(b, self.num_heads, self.head_dim, h * w)

        scale = self.head_dim ** -0.5
        attn = torch.softmax(torch.einsum("bhdn,bhdm->bhnm", q * scale, k), dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v).reshape(b, c, h, w)

        out = self.proj(out)
        return x + out


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class DiffusionUNet(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        base_ch=96,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=(16, 32),
        num_groups=32,
        dropout=0.0,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.base_ch = base_ch
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = set(attn_resolutions)
        self.num_groups = num_groups
        self.input_resolution = 64

        self.in_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_ch
        curr_res = self.input_resolution
        self._feature_channels = []

        for i, mult in enumerate(channel_mults):
            out_ch_level = base_ch * mult
            for _ in range(num_res_blocks):
                res_block = ResBlock(ch, out_ch_level, dropout=dropout, num_groups=num_groups)
                attn_block = AttentionBlock(out_ch_level, num_heads=4, num_groups=num_groups) \
                    if curr_res in self.attn_resolutions else nn.Identity()
                self.down_blocks.append(nn.ModuleList([res_block, attn_block]))
                ch = out_ch_level
                self._feature_channels.append(ch)

            if i != len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
                curr_res //= 2
            else:
                self.downsamples.append(nn.Identity())

        self.mid_block1 = ResBlock(ch, ch, dropout=dropout, num_groups=num_groups)
        self.mid_attn = AttentionBlock(ch, num_heads=4, num_groups=num_groups)
        self.mid_block2 = ResBlock(ch, ch, dropout=dropout, num_groups=num_groups)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch_level = base_ch * mult
            for _ in range(num_res_blocks):
                skip_ch = self._feature_channels.pop()
                res_block = ResBlock(ch + skip_ch, out_ch_level, dropout=dropout, num_groups=num_groups)
                attn_block = AttentionBlock(out_ch_level, num_heads=4, num_groups=num_groups) \
                    if curr_res in self.attn_resolutions else nn.Identity()
                self.up_blocks.append(nn.ModuleList([res_block, attn_block]))
                ch = out_ch_level

            if i != 0:
                self.upsamples.append(Upsample(ch))
                curr_res *= 2
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = nn.GroupNorm(num_groups, ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        hs = []
        h = self.in_conv(x)
        hs.append(h)

        down_block_idx = 0
        for i, downsample in enumerate(self.downsamples):
            for _ in range(self.num_res_blocks):
                res_block, attn_block = self.down_blocks[down_block_idx]
                h = attn_block(res_block(h))
                hs.append(h)
                down_block_idx += 1
            h = downsample(h)

        h = self.mid_block2(self.mid_attn(self.mid_block1(h)))

        up_block_idx = 0
        for i, upsample in enumerate(self.upsamples):
            for _ in range(self.num_res_blocks):
                skip = hs.pop()
                res_block, attn_block = self.up_blocks[up_block_idx]
                h = attn_block(res_block(torch.cat([h, skip], dim=1)))
                up_block_idx += 1
            h = upsample(h)

        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h


# ===================== Hide / Reveal =====================
class HideNet(nn.Module):
    def __init__(self, in_ch=8, out_ch=4, base_ch=64):
        super().__init__()
        self.unet = DiffusionUNet(
            in_ch=in_ch,
            out_ch=out_ch,
            base_ch=base_ch,
            channel_mults=(1, 2, 4, 8),
            num_res_blocks=2,
            attn_resolutions=(16, 32),
            num_groups=32,
            dropout=0.0,
        )

    def forward(self, tensor1, M):
        x = torch.cat([tensor1, M], dim=1)
        delta = self.unet(x)
        tensor2 = tensor1 + delta
        return tensor2, delta


class RevealNet(nn.Module):
    def __init__(self, in_ch=12, out_ch=4, base_ch=64):
        super().__init__()
        self.unet = DiffusionUNet(
            in_ch=in_ch,
            out_ch=out_ch,
            base_ch=base_ch,
            channel_mults=(1, 2, 4, 8),
            num_res_blocks=2,
            attn_resolutions=(16, 32),
            num_groups=32,
            dropout=0.0,
        )

    def forward(self, tensor1, tensor2):
        diff = tensor2 - tensor1
        x = torch.cat([tensor1, tensor2, diff], dim=1)
        out = self.unet(x)
        return out


# ===================== Stega Model =====================
class StegaModel(nn.Module):
    def __init__(
        self,
        lambda_rec: float = 1.0,
        lambda_vae_lpips: float = 1.0,
        base_ch: int = 64,
    ):
        super().__init__()
        self.hide_net = HideNet(in_ch=8, out_ch=4, base_ch=base_ch)
        self.reveal_net = RevealNet(in_ch=12, out_ch=4, base_ch=base_ch)

        self.lambda_rec = float(lambda_rec)
        self.lambda_vae_lpips = float(lambda_vae_lpips)

        # Learnable Q (used in Stage3; Stage1/2 使用随机 Q)
        init_Q = generate_orthogonal_matrix(
            n=64,
            seed=42,
            device="cpu",
            dtype=torch.float32
        )
        self.Q = nn.Parameter(init_Q, requires_grad=False)

    def hide(self, tensor1, M):
        return self.hide_net(tensor1, M)

    def reveal(self, tensor1, tensor2):
        return self.reveal_net(tensor1, tensor2)

    def compute_losses(self, tensor1, tensor2_clean, M, M_hat, delta=None):
        M_hat_prob = torch.sigmoid(M_hat)
        loss_rec = F.binary_cross_entropy(M_hat_prob, M)

        if self.lambda_vae_lpips > 0.0:
            # 1. 解码到像素域
            z1 = tensor1.to(device_global, dtype=torch.float32) / vae.config.scaling_factor
            decoded1 = vae.decode(z1).sample.to(device_global, dtype=torch.float32)

            z2 = tensor2_clean.to(device_global, dtype=torch.float32) / vae.config.scaling_factor
            decoded2 = vae.decode(z2).sample.to(device_global, dtype=torch.float32)

            # 2. 重新编码回 Latent 域，获取高斯分布
            dist1 = vae.encode(decoded1).latent_dist
            dist2 = vae.encode(decoded2).latent_dist

            mu1, logvar1 = dist1.mean, dist1.logvar
            mu2, logvar2 = dist2.mean, dist2.logvar

            # 3. 计算 KL 散度 KL(Stego || Cover)
            loss_kl = 0.5 * torch.sum(logvar1 - logvar2 + (torch.exp(logvar2) + (mu2 - mu1).pow(2)) / torch.exp(logvar1) - 1.0, dim=[1, 2, 3])
            loss_vae = loss_kl.mean()
        else:
            loss_vae = loss_rec.new_tensor(0.0)

        total_loss = self.lambda_rec * loss_rec + self.lambda_vae_lpips * loss_vae

        # 伪装变量名以适应原代码逻辑
        log = {
            "loss_rec": float(loss_rec.detach().item()),
            "loss_vae_lpips": float(loss_vae.detach().item()) if self.lambda_vae_lpips > 0.0 else 0.0,
            "loss_l1": 0.0,
            "total_loss": float(total_loss.detach().item()),
        }
        return total_loss, log


# ===================== Dataset =====================
class TensorDataset(Dataset):
    def __init__(self, cover_folder, max_samples=None):
        self.cover_folder = cover_folder
        cover_files = sorted([
            f for f in os.listdir(cover_folder)
            if f.startswith("local_means_step_") and f.endswith(".pt")
        ])
        if len(cover_files) == 0:
            raise RuntimeError(f"No local_means_step_*.pt found in {cover_folder}")

        if max_samples is not None:
            random.shuffle(cover_files)
            cover_files = cover_files[:max_samples]

        self.cover_files = cover_files

    def __len__(self):
        return len(self.cover_files)

    def _load_tensor(self, path):
        t = torch.load(path, map_location="cpu")
        if t.dim() == 4 and t.shape[0] == 1:
            t = t.squeeze(0)
        elif t.dim() != 3 or t.shape[0] != 4:
            raise ValueError(f"Invalid tensor shape: {t.shape}, expected [4,64,64]")
        return t.float()

    def __getitem__(self, idx):
        cover_name = self.cover_files[idx]
        cover_path = os.path.join(self.cover_folder, cover_name)
        tensor1 = self._load_tensor(cover_path)
        M = (torch.rand_like(tensor1) > 0.5).float()
        return tensor1, M


# ===================== Train / Val step =====================
def apply_channel(tensor2_clean, channel_mode: str, epoch_jpeg_schedule_active: bool = False):
    """Apply channel attacks to latent tensor"""
    if channel_mode == "none":
        return tensor2_clean, "none", {}

    if channel_mode == "random_attack":
        attack_params = randomize_attack_params()
        t2, attack_name = channel_attack_latent_random(tensor2_clean, vae)
        return t2, attack_name, attack_params

    raise ValueError(f"Unknown channel_mode: {channel_mode}")


def channel_attack_latent_random(latent_clean, vae_model):
    """Latent -> RGB -> Random attack -> Latent"""
    rgb_01 = latent_to_rgb_01(latent_clean, vae_model)
    attack_name = random.choice(ATTACK_NAMES)
    rgb_attacked = ATTACKS[attack_name](rgb_01)
    latent_attacked = rgb_01_to_latent(rgb_attacked, vae_model)
    return latent_attacked, attack_name


def latent_to_rgb_01(latent, vae_model):
    """Convert latent to RGB [0,1]"""
    z = latent / vae_model.config.scaling_factor
    rgb = vae_model.decode(z).sample
    rgb_01 = (rgb * 0.5 + 0.5).clamp(0.0, 1.0)
    return rgb_01


def rgb_01_to_latent(rgb_01, vae_model):
    """Convert RGB [0,1] to latent"""
    rgb = (rgb_01 * 2.0 - 1.0).clamp(-1.0, 1.0)
    posterior = vae_model.encode(rgb)
    latent = posterior.latent_dist.mode() * vae_model.config.scaling_factor
    return latent.to(dtype=torch.float32)


def train_step(model, optimizer, batch, device, channel_mode: str, is_stage3: bool = False):
    """
    训练步骤
    - Stage1/2: 使用随机正交矩阵 Q
    - Stage3: 使用可学习的 model.Q，并在通道后插入多专家软加权去噪
    """
    model.train()
    tensor1, M = batch
    tensor1 = tensor1.to(device)
    M = M.to(device)

    optimizer.zero_grad()

    # 正交变换
    B, C, H, W = M.shape

    if is_stage3:
        # Stage3: 使用 learnable Q
        Q = model.Q.to(device=tensor1.device, dtype=tensor1.dtype)
    else:
        # Stage1/2: 使用随机正交矩阵
        Q = generate_orthogonal_matrix(
            n=H,
            seed=None,
            device=tensor1.device,
            dtype=tensor1.dtype
        )

    M_flat = M.view(-1, H, W)        # (B*C, H, W)
    MQ_flat = torch.matmul(M_flat, Q)  # (B*C, H, H)
    MQ = MQ_flat.view(B, C, H, W)

    # 隐藏信息
    tensor2_clean, delta = model.hide(tensor1, MQ)

    # 通道攻击
    tensor2, attack_name, attack_params = apply_channel(tensor2_clean, channel_mode)

    # Stage3 中插入「多专家软加权去噪」
    if is_stage3 and MoE_DENOISER is not None:
        tensor2 = MoE_DENOISER(tensor2)

    # 提取信息
    MQ_hat = model.reveal(tensor1, tensor2)
    B2, C2, H2, W2 = MQ_hat.shape
    MQ_hat_flat = MQ_hat.view(-1, H2, W2)

    Q_inv = torch.linalg.inv(Q)
    M_hat_flat = torch.matmul(MQ_hat_flat, Q_inv)
    M_hat = M_hat_flat.view(B2, C2, H2, W2)

    # 计算损失
    total_loss, log = model.compute_losses(tensor1, tensor2_clean, M, M_hat, delta)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    if is_stage3 and MoE_DENOISER is not None:
        torch.nn.utils.clip_grad_norm_(MoE_DENOISER.parameters(), max_norm=5.0)
    optimizer.step()

    log["attack_name"] = attack_name
    log.update(attack_params)
    return log


def val_step(model, batch, device, channel_mode: str, is_stage3: bool = False):
    """
    验证步骤
    - Stage1/2: 使用随机正交矩阵 Q
    - Stage3: 使用 learnable Q，并在通道后插入多专家软加权去噪（eval）
    """
    model.eval()
    with torch.no_grad():
        tensor1, M = batch
        tensor1 = tensor1.to(device)
        M = M.to(device)

        # 正交变换
        B, C, H, W = M.shape

        if is_stage3:
            Q = model.Q.to(device=tensor1.device, dtype=tensor1.dtype)
        else:
            Q = generate_orthogonal_matrix(
                n=H,
                seed=None,
                device=tensor1.device,
                dtype=tensor1.dtype
            )

        M_flat = M.view(-1, H, W)
        MQ_flat = torch.matmul(M_flat, Q)
        MQ = MQ_flat.view(B, C, H, W)

        # 隐藏信息
        tensor2_clean, delta = model.hide(tensor1, MQ)

        # 通道攻击
        tensor2, attack_name, attack_params = apply_channel(tensor2_clean, channel_mode)

        # Stage3 验证时也用 MoE 去噪
        if is_stage3 and MoE_DENOISER is not None:
            MoE_DENOISER.eval()
            tensor2 = MoE_DENOISER(tensor2)

        # 提取信息
        MQ_hat = model.reveal(tensor1, tensor2)
        B2, C2, H2, W2 = MQ_hat.shape
        MQ_hat_flat = MQ_hat.view(-1, H2, W2)

        Q_inv = torch.linalg.inv(Q)
        M_hat_flat = torch.matmul(MQ_hat_flat, Q_inv)
        M_hat = M_hat_flat.view(B2, C2, H2, W2)

        # 计算损失和准确率
        total_loss, log = model.compute_losses(tensor1, tensor2_clean, M, M_hat, delta)
        M_pred = (M_hat > 0.5).float()
        acc = (M_pred == M).float().mean().item()

        log["acc"] = float(acc)
        log["attack_name"] = attack_name
        log.update(attack_params)
        return log


# ===================== Shared training loop helpers =====================
def build_loaders(batch_size: int, max_samples: int = 250):
    dataset = TensorDataset(COVER_FOLDER, max_samples=max_samples)
    total_len = len(dataset)
    train_len = int(total_len * 0.8)
    val_len = total_len - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return train_loader, val_loader


def epoch_attack_stats_init():
    return {name: 0 for name in (["none"] + ATTACK_NAMES)}


def update_attack_stats(stats: dict, attack_name: str):
    if attack_name not in stats:
        stats[attack_name] = 0
    stats[attack_name] += 1


# ===================== Stage 1 =====================
def main_stage1(args):
    config = dict(
        batch_size=2,
        epochs=100,
        lr=5e-5,
        base_ch=64,
        lambda_rec=10.0,
        lambda_vae_lpips=1.0,
        model_id=MODEL_ID,
        channel=args.channel,
    )

    swanlab.init(
        project=SWANLAB_PROJECT,
        workspace=SWANLAB_WORKSPACE,
        experiment_name=f"stega_stage1_{args.channel}",
        config=config
    )
    cfg = swanlab.config

    device = device_global
    print(f"[Stage1] device={device} bs={cfg.batch_size} channel={cfg.channel}")

    train_loader, val_loader = build_loaders(cfg.batch_size, max_samples=250)

    model = StegaModel(
        lambda_rec=cfg.lambda_rec,
        lambda_vae_lpips=cfg.lambda_vae_lpips,
        base_ch=cfg.base_ch
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=cfg.epochs, 
        eta_min=cfg.lr / 100
    )

    best_val_loss = float("inf")

    for epoch in tqdm(range(1, cfg.epochs + 1), desc="[Stage1] Epoch"):
        print(f"\n===== [Stage1] Epoch {epoch}/{cfg.epochs} | channel={args.channel} =====")

        train_logs = []
        train_stats = epoch_attack_stats_init()
        train_pbar = tqdm(train_loader, desc=f"Train {epoch}", leave=False)
        for batch in train_pbar:
            log = train_step(model, optimizer, batch, device, channel_mode=args.channel, is_stage3=False)
            train_logs.append(log)
            update_attack_stats(train_stats, log.get("attack_name", "none"))

            avg_loss = sum(x["total_loss"] for x in train_logs) / len(train_logs)
            train_pbar.set_postfix({"loss": f"{avg_loss:.6f}", "attack": log.get("attack_name", "-")})

        val_logs = []
        val_stats = epoch_attack_stats_init()
        val_pbar = tqdm(val_loader, desc=f"Val   {epoch}", leave=False)
        for batch in val_pbar:
            log = val_step(model, batch, device, channel_mode=args.channel, is_stage3=False)
            val_logs.append(log)
            update_attack_stats(val_stats, log.get("attack_name", "none"))

            avg_acc = sum(x["acc"] for x in val_logs) / len(val_logs)
            val_pbar.set_postfix({"acc": f"{avg_acc:.4f}", "attack": log.get("attack_name", "-")})

        # 计算平均指标
        train_loss = sum(x["total_loss"] for x in train_logs) / len(train_logs)
        train_lpips = sum(x["loss_vae_lpips"] for x in train_logs) / len(train_logs)
        train_rec = sum(x["loss_rec"] for x in train_logs) / len(train_logs)
        train_l1 = sum(x["loss_l1"] for x in train_logs) / len(train_logs)

        val_loss = sum(x["total_loss"] for x in val_logs) / len(val_logs)
        val_acc = sum(x["acc"] for x in val_logs) / len(val_logs)
        val_lpips = sum(x["loss_vae_lpips"] for x in val_logs) / len(val_logs)
        val_rec = sum(x["loss_rec"] for x in val_logs) / len(val_logs)
        val_l1 = sum(x["loss_l1"] for x in val_logs) / len(val_logs)

        # 打印日志
        print(f"[Stage1] train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.4f}")
        print(f"[Stage1] train_lpips={train_lpips:.8f} val_lpips={val_lpips:.8f}")
        print(f"[Stage1] train_attack_stats={train_stats}")
        print(f"[Stage1] val_attack_stats={val_stats}")

        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 记录指标
        swanlab.log({
            "train/total_loss": train_loss,
            "train/vae_lpips": train_lpips,
            "train/rec_loss": train_rec,
            "train/l1_loss": train_l1,
            "val/total_loss": val_loss,
            "val/vae_lpips": val_lpips,
            "val/rec_loss": val_rec,
            "val/l1_loss": val_l1,
            "val/accuracy": val_acc,
            "lr/current_lr": current_lr,
        }, step=epoch)

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), STAGE1_MODEL_PATH)
            print(f"[Stage1] Saved best to {STAGE1_MODEL_PATH}")

    print("[Stage1] Done.")
    swanlab.finish()


# ===================== Stage 2 =====================
def main_stage2(args):
    config = dict(
        batch_size=2,
        epochs=50,
        lr=3e-5,
        base_ch=64,
        lambda_rec=1.0,
        lambda_vae_lpips=10.0,
        model_id=MODEL_ID,
        channel=args.channel,
    )

    swanlab.init(
        project=SWANLAB_PROJECT,
        workspace=SWANLAB_WORKSPACE,
        experiment_name=f"stega_stage2_{args.channel}",
        config=config
    )
    cfg = swanlab.config

    device = device_global
    print(f"[Stage2] device={device} bs={cfg.batch_size} channel={cfg.channel}")

    train_loader, val_loader = build_loaders(cfg.batch_size, max_samples=250)

    model = StegaModel(
        lambda_rec=cfg.lambda_rec,
        lambda_vae_lpips=cfg.lambda_vae_lpips,
        base_ch=cfg.base_ch
    ).to(device)

    # 加载Stage1模型
    if not os.path.exists(STAGE1_MODEL_PATH):
        raise FileNotFoundError(f"Stage1 model not found: {STAGE1_MODEL_PATH}")
    state_dict = torch.load(STAGE1_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict,strict=False)
    print(f"[Stage2] Loaded Stage1 weights: {STAGE1_MODEL_PATH}")

    # 冻结RevealNet
    for p in model.reveal_net.parameters():
        p.requires_grad = False

    # 优化器仅训练HideNet（以及 Q 仍然 requires_grad=False，不被训练）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=cfg.lr)
    scheduler = lr_scheduler.StepLR(
        optimizer, 
        step_size=20, 
        gamma=0.5
    )

    best_val_loss = float("inf")

    for epoch in tqdm(range(1, cfg.epochs + 1), desc="[Stage2] Epoch"):
        print(f"\n===== [Stage2] Epoch {epoch}/{cfg.epochs} | channel={args.channel} =====")

        train_logs = []
        train_stats = epoch_attack_stats_init()
        train_pbar = tqdm(train_loader, desc=f"Train {epoch}", leave=False)
        for batch in train_pbar:
            log = train_step(model, optimizer, batch, device, channel_mode=args.channel, is_stage3=False)
            train_logs.append(log)
            update_attack_stats(train_stats, log.get("attack_name", "none"))

            avg_loss = sum(x["total_loss"] for x in train_logs) / len(train_logs)
            train_pbar.set_postfix({"loss": f"{avg_loss:.6f}", "attack": log.get("attack_name", "-")})

        val_logs = []
        val_stats = epoch_attack_stats_init()
        val_pbar = tqdm(val_loader, desc=f"Val   {epoch}", leave=False)
        for batch in val_pbar:
            log = val_step(model, batch, device, channel_mode=args.channel, is_stage3=False)
            val_logs.append(log)
            update_attack_stats(val_stats, log.get("attack_name", "none"))

            avg_acc = sum(x["acc"] for x in val_logs) / len(val_logs)
            val_pbar.set_postfix({"acc": f"{avg_acc:.4f}", "attack": log.get("attack_name", "-")})

        # 计算平均指标
        train_loss = sum(x["total_loss"] for x in train_logs) / len(train_logs)
        train_lpips = sum(x["loss_vae_lpips"] for x in train_logs) / len(train_logs)
        train_rec = sum(x["loss_rec"] for x in train_logs) / len(train_logs)
        train_l1 = sum(x["loss_l1"] for x in train_logs) / len(train_logs)

        val_loss = sum(x["total_loss"] for x in val_logs) / len(val_logs)
        val_acc = sum(x["acc"] for x in val_logs) / len(val_logs)
        val_lpips = sum(x["loss_vae_lpips"] for x in val_logs) / len(val_logs)
        val_rec = sum(x["loss_rec"] for x in val_logs) / len(val_logs)
        val_l1 = sum(x["loss_l1"] for x in val_logs) / len(val_logs)

        # 打印日志
        print(f"[Stage2] train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.4f}")
        print(f"[Stage2] train_lpips={train_lpips:.8f} val_lpips={val_lpips:.8f}")
        print(f"[Stage2] train_attack_stats={train_stats}")
        print(f"[Stage2] val_attack_stats={val_stats}")

        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 记录指标
        swanlab.log({
            "train/total_loss": train_loss,
            "train/vae_lpips": train_lpips,
            "train/rec_loss": train_rec,
            "train/l1_loss": train_l1,
            "val/total_loss": val_loss,
            "val/vae_lpips": val_lpips,
            "val/rec_loss": val_rec,
            "val/l1_loss": val_l1,
            "val/accuracy": val_acc,
            "lr/current_lr": current_lr,
        }, step=epoch)

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), STAGE2_MODEL_PATH)
            print(f"[Stage2] Saved best to {STAGE2_MODEL_PATH}")

    print("[Stage2] Done.")
    swanlab.finish()


# ===================== Stage 3 =====================
def main_stage3(args):
    config = dict(
        batch_size=2,
        epochs=50,
        lr=3e-5,
        base_ch=64,
        lambda_rec=10.0,
        lambda_vae_lpips=0.0,   # 关闭KL(原本为LPIPS)
        model_id=MODEL_ID,
        channel=args.channel,
    )

    swanlab.init(
        project=SWANLAB_PROJECT,
        workspace=SWANLAB_WORKSPACE,
        experiment_name=f"stega_stage3_{args.channel}",
        config=config
    )
    cfg = swanlab.config

    device = device_global
    print(f"[Stage3] device={device} bs={cfg.batch_size} channel={cfg.channel}")

    train_loader, val_loader = build_loaders(cfg.batch_size, max_samples=250)

    # 加载模型
    model = StegaModel(
        lambda_rec=cfg.lambda_rec,
        lambda_vae_lpips=cfg.lambda_vae_lpips,
        base_ch=cfg.base_ch
    ).to(device)

    if not os.path.exists(STAGE2_MODEL_PATH):
        raise FileNotFoundError(f"Stage2 model not found: {STAGE2_MODEL_PATH}")
    state_dict = torch.load(STAGE2_MODEL_PATH, map_location=device)
    # 旧的 Stage2 ckpt 里没有 Q，这里用 strict=False，保持当前初始化的 Q
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[Stage3] Loaded weights from {STAGE2_MODEL_PATH}")
    if missing:
        print(f"[Stage3] Missing keys (expected, e.g. Q): {missing}")
    if unexpected:
        print(f"[Stage3] Unexpected keys: {unexpected}")
        
    print(f"[Stage3] Loaded Stage2 weights: {STAGE2_MODEL_PATH}")

    # 构建多专家 MoE 去噪模块，并在 Stage3 中进行微调
    global MoE_DENOISER
    MoE_DENOISER = MultiExpertDenoiser(
        classifier_ckpt=CLASSIFIER_CKPT_PATH,
        denoiser_ckpt_dir=DENOISER_CKPT_DIR,
        denoiser_prefix=DENOISER_CKPT_PREFIX,
        noise_list=NOISE_LIST,
        device=device,
    ).to(device)

    # 冻结HideNet，解冻RevealNet、多专家 MoE 和可学习 Q
    for p in model.hide_net.parameters():
        p.requires_grad = False  # 冻结HideNet
    for p in model.reveal_net.parameters():
        p.requires_grad = True   # 解冻RevealNet
    for p in MoE_DENOISER.parameters():
        p.requires_grad = True   # 解冻多专家去噪网络

    # Stage3：让 Q 可学习
    model.Q.requires_grad = True

    print("[Stage3] Frozen HideNet, Unfrozen RevealNet, NAFNet and learnable Q")

    # 优化器包含 RevealNet + Q 和 多专家 MoE，设置差异化学习率
    trainable_params = [
        {
            "params": list(model.reveal_net.parameters()) + [model.Q],
            "lr": cfg.lr
        },
        {
            "params": MoE_DENOISER.parameters(),
            "lr": cfg.lr / 10
        }
    ]
    optimizer = optim.Adam(trainable_params, lr=cfg.lr)
    scheduler = lr_scheduler.StepLR(
        optimizer, 
        step_size=20, 
        gamma=0.5
    )

    best_val_loss = float("inf")

    for epoch in tqdm(range(1, cfg.epochs + 1), desc="[Stage3] Epoch"):
        print(f"\n===== [Stage3] Epoch {epoch}/{cfg.epochs} | channel={args.channel} =====")

        train_logs = []
        train_stats = epoch_attack_stats_init()
        train_pbar = tqdm(train_loader, desc=f"Train {epoch}", leave=False)
        for batch in train_pbar:
            log = train_step(model, optimizer, batch, device, channel_mode=args.channel, is_stage3=True)
            train_logs.append(log)
            update_attack_stats(train_stats, log.get("attack_name", "none"))

            avg_loss = sum(x["total_loss"] for x in train_logs) / len(train_logs)
            train_pbar.set_postfix({"loss": f"{avg_loss:.6f}", "attack": log.get("attack_name", "-")})

        val_logs = []
        val_stats = epoch_attack_stats_init()
        val_pbar = tqdm(val_loader, desc=f"Val   {epoch}", leave=False)
        for batch in val_pbar:
            log = val_step(model, batch, device, channel_mode=args.channel, is_stage3=True)
            val_logs.append(log)
            update_attack_stats(val_stats, log.get("attack_name", "none"))

            avg_acc = sum(x["acc"] for x in val_logs) / len(val_logs)
            val_pbar.set_postfix({"acc": f"{avg_acc:.4f}", "attack": log.get("attack_name", "-")})

        # 计算平均指标
        train_loss = sum(x["total_loss"] for x in train_logs) / len(train_logs)
        train_lpips = sum(x["loss_vae_lpips"] for x in train_logs) / len(train_logs)
        train_rec = sum(x["loss_rec"] for x in train_logs) / len(train_logs)

        val_loss = sum(x["total_loss"] for x in val_logs) / len(val_logs)
        val_acc = sum(x["acc"] for x in val_logs) / len(val_logs)
        val_lpips = sum(x["loss_vae_lpips"] for x in val_logs) / len(val_logs)
        val_rec = sum(x["loss_rec"] for x in val_logs) / len(val_logs)

        # 打印日志
        print(f"[Stage3] train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.4f}")
        print(f"[Stage3] train_attack_stats={train_stats}")
        print(f"[Stage3] val_attack_stats={val_stats}")

        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        nafnet_lr = optimizer.param_groups[1]['lr']

        # 记录指标（新增NAFNet学习率字段，语义上现在是 MoE 学习率）
        swanlab.log({
            "train/total_loss": train_loss,
            "train/vae_lpips": train_lpips,
            "train/rec_loss": train_rec,
            "val/total_loss": val_loss,
            "val/vae_lpips": val_lpips,
            "val/rec_loss": val_rec,
            "val/accuracy": val_acc,
            "train/attack_stats": train_stats,
            "val/attack_stats": val_stats,
            "lr/reveal_net_lr": current_lr,
            "lr/nafnet_lr": nafnet_lr,
        }, step=epoch)

        # 保存RevealNet + MoE + Q 的联合参数
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict = {
                "model": model.state_dict(),                # StegaModel（含 learnable Q）
                "moe_denoiser": MoE_DENOISER.state_dict(),  # 多专家 MoE 去噪网络
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "epoch": epoch
            }
            torch.save(save_dict, STAGE3_MODEL_PATH)
            print(f"[Stage3] Saved best model + NAFNet to {STAGE3_MODEL_PATH}")

    print("[Stage3] Done.")
    swanlab.finish()


# ===================== Entry =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3],
                        help="1: train Stage1; 2: train Stage2; 3: train Stage3")
    parser.add_argument("--channel", type=str, default="random_attack",
                        choices=["random_attack", "none"],
                        help="channel mode: random_attack | none")

    args = parser.parse_args()

    if args.stage == 1:
        main_stage1(args)
    elif args.stage == 2:
        main_stage2(args)
    else:
        main_stage3(args)