import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from dataclasses import dataclass
from tqdm import tqdm

# ===================== 🔴 1. 基础配置与路径 =====================
NOISE_TYPES = {
    "AWGN": ("/home/ygf/Denoising_net/dataset/train_processed_AWGN.pt", "/home/ygf/Denoising_net/dataset/val_processed_AWGN.pt"),
    "GBLUR": ("/home/ygf/Denoising_net/dataset/train_processed_GBLUR.pt", "/home/ygf/Denoising_net/dataset/val_processed_GBLUR.pt"),
    "JPEG": ("/home/ygf/Denoising_net/dataset/train_processed_JPEG.pt", "/home/ygf/Denoising_net/dataset/val_processed_JPEG.pt"),
    "MBLUR": ("/home/ygf/Denoising_net/dataset/train_processed_MBLUR.pt", "/home/ygf/Denoising_net/dataset/val_processed_MBLUR.pt"),
    "RESIZE": ("/home/ygf/Denoising_net/dataset/train_processed_RESIZE.pt", "/home/ygf/Denoising_net/dataset/val_processed_RESIZE.pt"),
}

NOISE_LIST = ["AWGN", "GBLUR", "JPEG", "MBLUR", "RESIZE"]
NOISE2LABEL = {name: idx for idx, name in enumerate(NOISE_LIST)}
LABEL2NOISE = {idx: name for name, idx in NOISE2LABEL.items()}
NUM_CLASSES = 5
INPUT_SHAPE = (4, 64, 64)

@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir: str = "./checkpoints"
    ckpt_name: str = "noise_classifier_best_0224.pt"

# ===================== 🔴 2. 数据处理 =====================
class NoiseClassificationDataset(Dataset):
    def __init__(self, noisy_pt_path: str, noise_label: int):
        super().__init__()
        self.noisy_data = torch.load(noisy_pt_path, map_location="cpu")
        self.noise_label = noise_label
        
        assert self.noisy_data.shape[1:] == INPUT_SHAPE, f"输入尺寸不符: {self.noisy_data.shape}"

    def __len__(self):
        return len(self.noisy_data)
    
    def __getitem__(self, idx):
        x = self.noisy_data[idx].float()
        # 简单增强：随机翻转
        if torch.rand(1) > 0.5: x = torch.flip(x, [1])
        if torch.rand(1) > 0.5: x = torch.flip(x, [2])
        return x, self.noise_label

# ===================== 🔴 3. 网络结构 (ResNet34) =====================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet34_Noise_Classifier(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet34_Noise_Classifier, self).__init__()
        self.in_planes = 64
        # 适配4通道输入
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4), # 防止过拟合
            nn.Linear(512, num_classes)
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        logits = self.classifier(out)
        return logits

# ===================== 🔴 4. 训练与判定逻辑 =====================
def train_model():
    cfg = TrainConfig()
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    
    # 数据加载
    train_sets = [NoiseClassificationDataset(v[0], NOISE2LABEL[k]) for k, v in NOISE_TYPES.items()]
    val_sets = [NoiseClassificationDataset(v[1], NOISE2LABEL[k]) for k, v in NOISE_TYPES.items()]
    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(ConcatDataset(val_sets), batch_size=cfg.batch_size, shuffle=False)

    model = ResNet34_Noise_Classifier(num_classes=NUM_CLASSES).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(cfg.device), y.to(cfg.device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(cfg.device), y.to(cfg.device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        acc = correct / total
        print(f"Validation Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, cfg.ckpt_name))

# ===================== 🔴 5. 直接判定攻击类别的接口 =====================
def detect_attack(noisy_tensor, model_weight_path="./checkpoints/noise_classifier_best_0224.pt"):
    """
    输入: (4, 64, 64) 或 (B, 4, 64, 64) 的张量
    输出: 攻击名称 (例如: 'AWGN')
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet34_Noise_Classifier(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    if noisy_tensor.ndim == 3:
        noisy_tensor = noisy_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(noisy_tensor.to(device))
        pred_idx = output.argmax(dim=1).item()
        
    return NOISE_LIST[pred_idx]

if __name__ == "__main__":
    # 第一步：运行训练 (训练好后可以注释掉)
    train_model()

    # 第二步：测试判定功能
    # 模拟一个受 JPEG 攻击的张量
    test_tensor = torch.randn(4, 64, 64)
    attack_type = detect_attack(test_tensor)
    print(f"\n[结果] 检测到攻击类别为: {attack_type}")