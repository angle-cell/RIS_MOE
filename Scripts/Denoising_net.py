# train_nafnet_from_pt.py
# Train NAFNet-style restoration network: processed (B,4,64,64) -> original (B,4,64,64)
# Loss: MSE

import os
import time
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional
from tqdm import tqdm  # 新增：导入tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1) Model: NAFNet (single-stage UNet) + NAFBlock
# ============================================================

class LayerNorm2d(nn.Module):
    """LayerNorm over channel dimension for NCHW."""
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, c, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class SimpleGate(nn.Module):
    """Split channels into two halves and multiply element-wise."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SCA(nn.Module):
    """Simplified Channel Attention: X * Conv1x1(GAP(X))."""
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool2d(x, 1)
        w = self.conv(w)
        return x * w


class NAFBlock(nn.Module):
    """
    NAFBlock:
      x = x + beta  * F1(LN(x))
      x = x + gamma * F2(LN(x))

    F1: 1x1 -> DW3x3 -> SimpleGate -> SCA -> 1x1
    F2: 1x1 -> SimpleGate -> 1x1
    """
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        # Sub-block 1
        self.norm1 = LayerNorm2d(c)
        c_dw = c * dw_expand
        self.pw1 = nn.Conv2d(c, c_dw, kernel_size=1, bias=True)
        self.dw = nn.Conv2d(c_dw, c_dw, kernel_size=3, padding=1, groups=c_dw, bias=True)
        self.sg1 = SimpleGate()
        self.sca = SCA(c_dw // 2)
        self.pw2 = nn.Conv2d(c_dw // 2, c, kernel_size=1, bias=True)

        # Sub-block 2
        self.norm2 = LayerNorm2d(c)
        c_ffn = c * ffn_expand
        self.pw3 = nn.Conv2d(c, c_ffn, kernel_size=1, bias=True)
        self.sg2 = SimpleGate()
        self.pw4 = nn.Conv2d(c_ffn // 2, c, kernel_size=1, bias=True)

        # Learnable residual scales
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.pw1(y)
        y = self.dw(y)
        y = self.sg1(y)
        y = self.sca(y)
        y = self.pw2(y)
        x = x + y * self.beta

        y2 = self.norm2(x)
        y2 = self.pw3(y2)
        y2 = self.sg2(y2)
        y2 = self.pw4(y2)
        x = x + y2 * self.gamma
        return x


class NAFNet(nn.Module):
    """
    Single-stage UNet with skip-add fusion.
    Residual learning: out = inp + net(inp)
    """
    def __init__(
        self,
        in_ch: int = 4,
        out_ch: int = 4,
        width: int = 32,
        enc_blk_nums: Tuple[int, ...] = (2, 2, 4, 8),
        dec_blk_nums: Tuple[int, ...] = (2, 2, 2, 2),
        middle_blk_num: int = 12,
        dw_expand: int = 2,
        ffn_expand: int = 2,
    ):
        super().__init__()
        assert len(enc_blk_nums) == len(dec_blk_nums), "enc/dec stages must match"

        self.intro = nn.Conv2d(in_ch, width, kernel_size=3, padding=1, bias=True)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        c = width
        for n_blk in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[
                NAFBlock(c, dw_expand=dw_expand, ffn_expand=ffn_expand) for _ in range(n_blk)
            ]))
            self.downs.append(nn.Conv2d(c, c * 2, kernel_size=2, stride=2, bias=True))
            c *= 2

        # Middle
        self.middle = nn.Sequential(*[
            NAFBlock(c, dw_expand=dw_expand, ffn_expand=ffn_expand) for _ in range(middle_blk_num)
        ])

        # Decoder
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n_blk in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(c, c * 2, kernel_size=1, bias=True),
                nn.PixelShuffle(2),
            ))
            c //= 2
            self.decoders.append(nn.Sequential(*[
                NAFBlock(c, dw_expand=dw_expand, ffn_expand=ffn_expand) for _ in range(n_blk)
            ]))

        self.ending = nn.Conv2d(width, out_ch, kernel_size=3, padding=1, bias=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.intro(inp)

        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = x + skip
            x = dec(x)

        x = self.ending(x)
        return inp + x


def build_default_nafnet_4x64(width: int = 32) -> NAFNet:
    # Paper-like layout: total 36 blocks
    return NAFNet(
        in_ch=4, out_ch=4, width=width,
        enc_blk_nums=(2, 2, 4, 8),
        dec_blk_nums=(2, 2, 2, 2),
        middle_blk_num=12,
        dw_expand=2, ffn_expand=2,
    )


# ============================================================
# 2) Dataset: load your .pt tensors (B,4,64,64)
# ============================================================

class PTensorPairDataset(Dataset):
    """
    processed_path: torch tensor (B,4,64,64)
    original_path : torch tensor (B,4,64,64)
    returns: (processed[i], original[i]) each (4,64,64)
    """
    def __init__(self, processed_path: str, original_path: str, dtype=torch.float32):
        super().__init__()
        self.processed = torch.load(processed_path, map_location="cpu")
        self.original = torch.load(original_path, map_location="cpu")

        if not isinstance(self.processed, torch.Tensor) or not isinstance(self.original, torch.Tensor):
            raise TypeError("Your .pt files must contain a single torch.Tensor.")

        if self.processed.shape != self.original.shape:
            raise ValueError(f"Shape mismatch: processed {tuple(self.processed.shape)} vs original {tuple(self.original.shape)}")

        if self.processed.ndim != 4 or self.processed.shape[1:] != (4, 64, 64):
            raise ValueError(f"Expected shape (B,4,64,64), got {tuple(self.processed.shape)}")

        # Ensure dtype float32 for training stability/perf
        self.processed = self.processed.to(dtype)
        self.original = self.original.to(dtype)

    def __len__(self):
        return self.processed.shape[0]

    def __getitem__(self, idx: int):
        return self.processed[idx], self.original[idx]


# ============================================================
# 3) Train / Eval / Checkpoint (新增进度条)
# ============================================================

@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-2
    num_workers: int = 4
    device: str = "cuda"
    grad_clip: float = 0.0
    log_every: int = 50
    ckpt_dir: str = "./checkpoints"
    ckpt_name: str = "nafnet_4x64.pt"


def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_val: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
    }, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf"))


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: str, epoch: int, log_every: int = 50, grad_clip: float = 0.0) -> float:
    """新增epoch参数，用于进度条显示；添加tqdm进度条"""
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    n = 0
    t0 = time.time()

    # 创建训练进度条，显示epoch和实时loss
    pbar = tqdm(enumerate(loader, start=1), total=len(loader), 
                desc=f"[Train] Epoch {epoch+1}", leave=True, colour="green")
    
    for step, (x, y) in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs

        # 实时更新进度条的附加信息
        avg_loss = total_loss / max(n, 1)
        pbar.set_postfix({
            "step_loss": f"{loss.item():.6f}",
            "avg_loss": f"{avg_loss:.6f}",
            "time": f"{time.time()-t0:.1f}s"
        })

        # 保留原有日志（可选，进度条已覆盖核心信息）
        if log_every > 0 and step % log_every == 0:
            dt = time.time() - t0
            print(f"  Step {step:5d}/{len(loader)} | MSE {loss.item():.6f} | Time {dt:.1f}s")
            t0 = time.time()

    pbar.close()
    return avg_loss


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, epoch: int) -> float:
    """新增epoch参数，添加验证进度条"""
    model.eval()
    criterion = nn.MSELoss(reduction="mean")
    total = 0.0
    n = 0

    # 创建验证进度条
    pbar = tqdm(loader, total=len(loader), 
                desc=f"[Eval]  Epoch {epoch+1}", leave=True, colour="blue")
    
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        loss = criterion(pred, y)
        
        bs = x.size(0)
        total += loss.item() * bs
        n += bs

        # 更新验证进度条信息
        avg_val_loss = total / max(n, 1)
        pbar.set_postfix({
            "val_loss": f"{loss.item():.6f}",
            "avg_val_loss": f"{avg_val_loss:.6f}"
        })

    pbar.close()
    return avg_val_loss


# ============================================================
# 4) Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_processed", type=str, default="/home/ygf/GNet/GNet_V3/train_processed.pt")
    p.add_argument("--train_original",  type=str, default="/home/ygf/GNet/GNet_V3/train_original.pt")
    p.add_argument("--val_processed",   type=str, default="/home/ygf/GNet/GNet_V3/val_processed.pt")
    p.add_argument("--val_original",    type=str, default="/home/ygf/GNet/GNet_V3/val_original.pt")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--width", type=int, default=32)

    p.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    p.add_argument("--ckpt_name", type=str, default="nafnet_4x64.pt")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print(f"CUDA unavailable, fallback to CPU")

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
        device=device,
        log_every=args.log_every,
        ckpt_dir=args.ckpt_dir,
        ckpt_name=args.ckpt_name,
    )
    ckpt_path = os.path.join(cfg.ckpt_dir, cfg.ckpt_name)

    # Datasets
    train_ds = PTensorPairDataset(args.train_processed, args.train_original, dtype=torch.float32)
    val_ds   = PTensorPairDataset(args.val_processed, args.val_original, dtype=torch.float32)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(cfg.device != "cpu")
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(cfg.device != "cpu")
    )

    # Model
    model = build_default_nafnet_4x64(width=args.width).to(cfg.device)
    print(f"=== Training Setup ===")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"Model params: {count_params(model)/1e6:.3f} M")
    print(f"Device: {cfg.device} | Batch size: {cfg.batch_size}")
    print(f"Epochs: {cfg.epochs} | LR: {cfg.lr}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    start_epoch = 0
    best_val = float("inf")
    if args.resume and os.path.exists(ckpt_path):
        e, best_val = load_checkpoint(ckpt_path, model, optimizer)
        start_epoch = e + 1
        print(f"Resumed from {ckpt_path} (epoch {start_epoch}, best_val={best_val:.6f})")

    # 全局训练进度条（可选，外层进度条）
    global_pbar = tqdm(range(start_epoch, cfg.epochs), desc="Total Training", colour="magenta")
    
    # Train
    for epoch in global_pbar:
        # 训练一个epoch（传入epoch参数用于进度条显示）
        tr_loss = train_one_epoch(model, train_loader, optimizer, cfg.device, epoch,
                                 log_every=cfg.log_every, grad_clip=cfg.grad_clip)
        # 验证
        val_loss = evaluate(model, val_loader, cfg.device, epoch)
        
        # 更新全局进度条信息
        global_pbar.set_postfix({
            "train_mse": f"{tr_loss:.6f}",
            "val_mse": f"{val_loss:.6f}",
            "best_val": f"{best_val:.6f}"
        })
        
        # 打印epoch总结
        print(f"\n=== Epoch {epoch+1}/{cfg.epochs} Summary ===")
        print(f"Train MSE: {tr_loss:.6f} | Val MSE: {val_loss:.6f}")

        # 保存最优模型
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ckpt_path, model, optimizer, epoch, best_val)
            print(f"✅ Saved best checkpoint to {ckpt_path} (best_val={best_val:.6f})")

    global_pbar.close()
    print("\n=== Training Finished ===")
    print(f"Best validation MSE: {best_val:.6f}")
    print(f"Final checkpoint saved at: {ckpt_path}")


if __name__ == "__main__":
    main()