import os
import argparse
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =========================================================
# 0) 固定配置：与你 pack 的 attack_layers 完全一致（小写）
# =========================================================
NOISE_LIST = ['resize', 'jpeg', 'gblur', 'awgn', 'mblur']
NOISE2LABEL = {name: idx for idx, name in enumerate(NOISE_LIST)}
LABEL2NOISE = {idx: name for name, idx in NOISE2LABEL.items()}
NUM_CLASSES = len(NOISE_LIST)
INPUT_SHAPE = (4, 64, 64)


# =========================================================
# 1) Dataset：从 train_pack.pt/val_pack.pt 读取 processed + attack_layers
#    labels 字段你说是错的，这里完全不用 labels，现场从 attack_layers 生成 multi-hot
# =========================================================
class MultiAttackPackDataset(Dataset):
    def __init__(self, pack_path: str, dtype=torch.float32, augment: bool = True):
        super().__init__()
        pack = torch.load(pack_path, map_location="cpu")

        if "processed" not in pack or "attack_layers" not in pack:
            raise KeyError("pack 必须包含 'processed' 和 'attack_layers' 两个字段")

        self.x = pack["processed"]  # (N,4,64,64)
        self.attack_layers = pack["attack_layers"]  # list[list[str]]
        self.dtype = dtype
        self.augment = augment

        assert isinstance(self.x, torch.Tensor), "pack['processed'] 必须是 torch.Tensor"
        assert self.x.ndim == 4 and self.x.shape[1:] == INPUT_SHAPE, \
            f"processed shape 应为 (N,{INPUT_SHAPE})，当前是 {self.x.shape}"
        assert len(self.attack_layers) == self.x.shape[0], \
            f"attack_layers 数量 {len(self.attack_layers)} 与样本数 {self.x.shape[0]} 不一致"

        self.x = self.x.to(dtype)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        noisy_x = self.x[idx]  # (4,64,64)
        layers: List[str] = self.attack_layers[idx]  # e.g. ['resize','jpeg'] 或 []

        # multi-hot label (5,)
        y = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for layer in layers:
            if layer not in NOISE2LABEL:
                raise ValueError(f"未知攻击类型: {layer}, 期望属于 {NOISE_LIST}")
            y[NOISE2LABEL[layer]] = 1.0

        # 可选：极简增强
        if self.augment:
            if torch.rand(1) > 0.5:
                noisy_x = torch.flip(noisy_x, dims=[2])  # 水平翻转
            if torch.rand(1) > 0.5:
                noisy_x = torch.flip(noisy_x, dims=[1])  # 垂直翻转

        return noisy_x, y


# =========================================================
# 2) ResNet34（你原版结构保留），但 forward 只返回 logits（不做 softmax）
# =========================================================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class ResNet34_MultiLabel(nn.Module):
    """
    输入:  (B,4,64,64)
    输出:  logits (B,5) —— 多标签用 BCEWithLogitsLoss
    推理:  probs = sigmoid(logits) -> 每类权重(0~1)
    """
    def __init__(self, num_classes=NUM_CLASSES, in_channels=4):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits


# =========================================================
# 3) 指标：micro-F1 + exact match
# =========================================================
@torch.no_grad()
def multilabel_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """
    logits:   (N,5)
    targets:  (N,5) float{0,1}
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    exact = (preds == targets).all(dim=1).float().mean()
    return f1.item(), exact.item()


# =========================================================
# 4) 训练配置
# =========================================================
@dataclass
class TrainConfig:
    train_pack: str = "/home/ygf/Denoising_net/dataset/train_pack.pt"
    val_pack: str = "/home/ygf/Denoising_net/dataset/val_pack.pt"

    epochs: int = 25
    batch_size: int = 128
    lr: float = 7e-4
    weight_decay: float = 1e-4
    threshold: float = 0.475

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir: str = "./checkpoints"
    ckpt_name: str = "noise_multilabel_resnet34_best.pt"
    seed: int = 42
    num_workers: int = 4


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def save_best_checkpoint(model, best_f1, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "best_val_f1": best_f1,
        "noise_list": NOISE_LIST,
        "label2noise": LABEL2NOISE,
        "threshold": 0.5
    }, save_path)
    print(f"\n✅ 最优权重已保存至：{save_path} | Best Val micro-F1：{best_f1:.4f}")


# =========================================================
# 5) Train / Eval
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="[Train]", colour="green")
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg": f"{total_loss/(pbar.n+1):.4f}"})
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []

    pbar = tqdm(loader, desc="[Val]", colour="blue")
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()

        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    val_f1, val_exact = multilabel_metrics_from_logits(all_logits, all_targets, threshold=threshold)
    return total_loss / max(len(loader), 1), val_f1, val_exact


# =========================================================
# 6) 推理：输出攻击集合 + 权重（sigmoid），可选归一化让和=1
# =========================================================
@torch.no_grad()
def infer_noise_weights(noisy_tensor: torch.Tensor,
                       model_path: str,
                       device: str = "cuda",
                       threshold: float = 0.475,
                       normalize: bool = True):
    ckpt = torch.load(model_path, map_location=device)
    model = ResNet34_MultiLabel()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    if noisy_tensor.dim() == 3:
        noisy_tensor = noisy_tensor.unsqueeze(0)  # (1,4,64,64)

    noisy_tensor = noisy_tensor.to(device)
    logits = model(noisy_tensor)                 # (B,5)
    probs = torch.sigmoid(logits)                # (B,5) independent

    weights = probs
    if normalize:
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

    pred_mask = (probs >= threshold)

    pred_names = []
    for i in range(noisy_tensor.shape[0]):
        names = [NOISE_LIST[j] for j in range(NUM_CLASSES) if pred_mask[i, j].item()]
        pred_names.append(names)

    return pred_names, weights.detach().cpu().numpy().tolist()


# =========================================================
# 7) 主函数
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pack", type=str, default=TrainConfig.train_pack)
    parser.add_argument("--val_pack", type=str, default=TrainConfig.val_pack)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--threshold", type=float, default=TrainConfig.threshold)
    parser.add_argument("--ckpt_dir", type=str, default=TrainConfig.ckpt_dir)
    parser.add_argument("--ckpt_name", type=str, default=TrainConfig.ckpt_name)
    parser.add_argument("--num_workers", type=int, default=TrainConfig.num_workers)
    args = parser.parse_args()

    cfg = TrainConfig(
        train_pack=args.train_pack,
        val_pack=args.val_pack,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        threshold=args.threshold,
        ckpt_dir=args.ckpt_dir,
        ckpt_name=args.ckpt_name,
        num_workers=args.num_workers
    )

    # seed_everything(cfg.seed)
    save_path = os.path.join(cfg.ckpt_dir, cfg.ckpt_name)

    # print(f"📌 Train pack: {cfg.train_pack}")
    # print(f"📌 Val   pack: {cfg.val_pack}")
    # print(f"📌 Device: {cfg.device}")
    # print(f"📌 Noise order: {NOISE_LIST}  (这决定输出权重顺序)")

    # train_ds = MultiAttackPackDataset(cfg.train_pack, augment=True)
    val_ds = MultiAttackPackDataset(cfg.val_pack, augment=False)
    # print(f"📌 数据集加载完成 | Train: {len(train_ds)} | Val: {len(val_ds)}")

    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=cfg.batch_size,
    #     shuffle=True,
    #     num_workers=cfg.num_workers,
    #     pin_memory=True,
    #     drop_last=False
    # )
    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=cfg.batch_size,
    #     shuffle=False,
    #     num_workers=cfg.num_workers,
    #     pin_memory=True,
    #     drop_last=False
    # )

    # model = ResNet34_MultiLabel().to(cfg.device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # criterion = nn.BCEWithLogitsLoss()

    # print(f"📌 模型参数量：{sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    # print("=" * 80)

    # best_val_f1 = 0.0
    # for epoch in range(cfg.epochs):
    #     train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg.device)
    #     val_loss, val_f1, val_exact = eval_one_epoch(model, val_loader, criterion, cfg.device, threshold=cfg.threshold)

    #     print(f"✅ Epoch {epoch+1}/{cfg.epochs} | "
    #           f"TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
    #           f"Val micro-F1 {val_f1:.4f} | Val Exact {val_exact:.4f} | BestF1 {best_val_f1:.4f}")

    #     if val_f1 > best_val_f1:
    #         best_val_f1 = val_f1
    #         save_best_checkpoint(model, best_val_f1, save_path)

    # print("=" * 80)
    # print(f"🎉 训练完成 | Best Val micro-F1: {best_val_f1:.4f}")
    # print(f"📌 Best checkpoint: {save_path}")

    # 训练完给一个快速推理示例（随机取一个 val 样本）
    sample_x, sample_y = val_ds[11]
    pred_names, weights = infer_noise_weights(sample_x, save_path, device=cfg.device, threshold=cfg.threshold, normalize=True)
    # print("\n🔍 推理示例（val_ds[1]）")
    print(f"GT multi-hot: {sample_y.tolist()}  (顺序 {NOISE_LIST})")
    print(f"Pred attacks: {pred_names[0]}")
    print(f"Weights(归一化): {weights[0]}  (顺序 {NOISE_LIST})")


if __name__ == "__main__":
    main()
