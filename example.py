import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
scripts_dir = current_dir / "Scripts"
sys.path.append(str(scripts_dir))
Mapping_path = current_dir / "Mapping_model"
sys.path.append(str(Mapping_path))
Denosing_path = current_dir / "Denoising_model"
sys.path.append(str(Denosing_path))

import torch
from extract_prompt import read_json_8k, read_json
from PIL import Image
import os
import utils
import logging
from diffusers import StableDiffusionPipeline, DDIMScheduler
import datetime
import attack
import numpy as np
from tqdm import tqdm
from Denoising_net import build_default_nafnet_4x64, LayerNorm2d, SimpleGate, SCA, NAFBlock, NAFNet
import torch.nn as nn
# ========== 修改1：替换为训练代码中最新的 StegaModel 定义（关键！） ==========
# 注释掉旧的导入，直接嵌入训练时的 StegaModel 完整定义，避免版本不一致
# from newdenosiing_withQ_0206 import StegaModel, generate_orthogonal_matrix
from noise_classifier_0127 import ResNet34_MultiLabel
from diffusers import UNet2DConditionModel, DDIMInverseScheduler
import torch.nn.functional as F



# ========== 嵌入训练代码中完整的网络定义（确保和训练时一致） ==========
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
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
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

class HideNet(nn.Module):
    # 训练时的核心修改：in_ch = 4 (Cover) + 1 (M) = 5
    def __init__(self, in_ch=5, out_ch=4, base_ch=64):
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
    # 训练时的核心修改：out_ch 改为 1，提取1通道M
    def __init__(self, in_ch=12, out_ch=1, base_ch=64):
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
def generate_orthogonal_matrix(n: int = 64, seed: int | None = None, device: str = "cuda", dtype=torch.float32) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    A = torch.randn((n, n), dtype=dtype, device=device)
    Q, R = torch.linalg.qr(A)
    d = torch.sign(torch.diag(R))
    d[d == 0] = 1.0
    Q = Q * d.unsqueeze(0)
    return Q


class StegaModel(nn.Module):
    def __init__(
        self,
        lambda_rec: float = 1.0,
        lambda_vae_lpips: float = 1.0,
        base_ch: int = 64,
    ):
        super().__init__()
        self.hide_net = HideNet(in_ch=5, out_ch=4, base_ch=base_ch)
        self.reveal_net = RevealNet(in_ch=12, out_ch=1, base_ch=base_ch)

        self.lambda_rec = float(lambda_rec)
        self.lambda_vae_lpips = float(lambda_vae_lpips)

        # Learnable Q
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
            B = tensor1.shape[0]
            device = tensor1.device
            dtype = tensor1.dtype
            
            t = torch.randint(0, 1000, (B,), device=device).long()
            noise = torch.randn_like(tensor1)
            
            alphas_cumprod = inversion_scheduler.alphas_cumprod.to(device)
            alpha_prod_t = alphas_cumprod[t].view(B, 1, 1, 1)
            beta_prod_t = 1 - alpha_prod_t
            
            x_t_cover = torch.sqrt(alpha_prod_t) * tensor1 + torch.sqrt(beta_prod_t) * noise
            x_t_stego = torch.sqrt(alpha_prod_t) * tensor2_clean + torch.sqrt(beta_prod_t) * noise
            
            encoder_hidden_states = torch.zeros((B, 77, 1024), device=device, dtype=dtype)
            
            noise_pred_cover = sd_unet(x_t_cover, t, encoder_hidden_states=encoder_hidden_states).sample
            noise_pred_stego = sd_unet(x_t_stego, t, encoder_hidden_states=encoder_hidden_states).sample
            
            score_mse = F.mse_loss(noise_pred_stego, noise_pred_cover)
            loss_vae = score_mse
            
            if delta is not None:
                tv_loss = torch.mean(torch.abs(delta[:, :, :, :-1] - delta[:, :, :, 1:])) + \
                          torch.mean(torch.abs(delta[:, :, :-1, :] - delta[:, :, 1:, :]))
                loss_vae = loss_vae + 0.05 * tv_loss
        else:
            loss_vae = loss_rec.new_tensor(0.0)
            score_mse = loss_rec.new_tensor(0.0)

        total_loss = self.lambda_rec * loss_rec + self.lambda_vae_lpips * loss_vae

        log = {
            "loss_rec": float(loss_rec.detach().item()),
            "loss_vae_lpips": float(loss_vae.detach().item()) if self.lambda_vae_lpips > 0.0 else 0.0,
            "loss_l1": float(score_mse.detach().item()) if self.lambda_vae_lpips > 0.0 else 0.0,
            "total_loss": float(total_loss.detach().item()),
        }
        return total_loss, log

# 设备初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 目录创建
os.makedirs("./output_coco8K_4096", exist_ok=True)
os.makedirs("./log", exist_ok=True)
os.makedirs("./output_coco8K_4096/img/", exist_ok=True)
os.makedirs("./output_coco8K_4096/stego/image/", exist_ok=True)
os.makedirs("./output_coco8K_4096/stego/pt/", exist_ok=True)
os.makedirs("./output_coco8K_4096/cover/image/", exist_ok=True)
os.makedirs("./output_coco8K_4096/cover/pt/", exist_ok=True)
os.makedirs("./output_coco8K_4096/cover/pt/mean/", exist_ok=True)
os.makedirs("./output_coco8K_4096/cover/pt/diff/", exist_ok=True)
os.makedirs("./output_coco8K_4096/zt/", exist_ok=True)

# 初始化日志
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log_filename = f"./log/embed_{timestamp}.log"
file_handler = logging.FileHandler(log_filename, mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)

# 超参数
num_steps = 50
base_seed = 12345
# ========== 修改2：消息长度改为4096bit（1*64*64） ==========
message_len = 1 * 64 * 64  # 原4*64*64=16384，改为1*64*64=4096
eta = 0

# 噪声类型列表（与多专家训练时一致）
NOISE_LIST = ['RESIZE', 'JPEG', 'GBLUR', 'AWGN', 'MBLUR']
MOE_THRESHOLD = 0.475  

# 准备prompts
dataset_path = "/home/ygf/First_paper/Input/text_prompt_dataset/coco_dataset.txt"
prompt_num = 5000
prompts = read_json(dataset_path, prompt_num)
logging.info(f"Prompt解析完成,路径:{dataset_path},长度{len(prompts)}")

# 加载SD模型
model_id = "/home/ygf/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, eta=eta)
pipe = pipe.to("cuda")
# 保存VAE的初始设备和数据类型
vae_device = pipe.vae.device
vae_dtype = pipe.vae.dtype
logging.info("SD模型加载成功")

# ===================== StegaModel / 多专家去噪网络加载 =====================

# 1) 构建 StegaModel 结构（超参需与 Stage3 训练时一致）
model = StegaModel(
    lambda_rec=10.0,       # Stage3 的 lambda_rec
    lambda_vae_lpips=0.0,  # Stage3 关闭 LPIPS
    base_ch=64
).to(device)

# ========== 修改3：加载新的Stage3权重文件 train_3_0317_4096.pt ==========
stage3_ckpt_path = "/home/ygf/First_paper/Mapping_model/train_3_0531_4096.pt"  # 替换为新权重路径
stage3_ckpt = torch.load(stage3_ckpt_path, map_location=device)

# 加载 Stage3 训练好的 StegaModel 参数（含 learnable Q）
missing_keys, unexpected_keys = model.load_state_dict(stage3_ckpt["model"], strict=False)
logging.info(f"Stage3 StegaModel 加载完成，missing_keys={missing_keys}, unexpected_keys={unexpected_keys}")
# 推理模式：冻结 StegaModel
model.eval()
for p in model.parameters():
    p.requires_grad = False
logging.info("StegaModel 冻结并切换到 eval 模式")

# 2) 定义与训练时一致的多专家 MoE 去噪网络结构
class MultiExpertDenoiser(torch.nn.Module):
    """
    多专家去噪模块（与训练逻辑一致）：
    - classifier: ResNet34_MultiLabel（多标签分类，输出logits）
    - experts: ModuleDict{noise_type: NAFNet}
    前向逻辑：
        1. logits -> sigmoid -> 各噪声类型置信度 (B,5)
        2. 按阈值筛选激活的专家（置信度 ≥ THRESHOLD）
        3. 无激活专家时，选置信度最高的1个
        4. 激活专家权重归一化（和为1）
        5. 仅激活的专家参与加权融合
    """
    def __init__(self, noise_list, threshold=0.475, width: int = 32):
        super().__init__()
        self.noise_list = noise_list
        self.threshold = threshold  # 专家激活阈值

        # 分类器结构与训练阶段完全一致（多标签分类，输出logits）
        self.classifier = ResNet34_MultiLabel(
            num_classes=len(noise_list),
            in_channels=4
        )

        # 5 个专家 NAFNet 结构与训练阶段一致
        experts = {}
        for nt in noise_list:
            experts[nt] = build_default_nafnet_4x64(width=width)
        self.experts = torch.nn.ModuleDict(experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_single = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            is_single = True

        B = x.shape[0]
        device = next(self.parameters()).device
        x = x.to(device)

        # 1. 分类器输出logits -> sigmoid得到各噪声类型置信度（多标签）
        logits = self.classifier(x)  # (B,5) 原始logits
        confs = torch.sigmoid(logits)  # (B,5) 置信度∈[0,1]
        
        # 2. 按阈值筛选激活的专家（每个样本独立筛选）
        activated_experts = []  # 存储每个样本激活的专家索引
        activated_weights = []  # 存储每个样本激活专家的权重
        for b in range(B):
            # 筛选当前样本置信度≥阈值的专家索引
            sample_confs = confs[b]  # (5,)
            active_idx = torch.where(sample_confs >= self.threshold)[0]
            
            # 兜底策略：无激活专家时，选置信度最高的1个
            if len(active_idx) == 0:
                active_idx = torch.argmax(sample_confs, dim=0, keepdim=True)
            
            # 提取激活专家的权重并归一化（和为1）
            sample_weights = sample_confs[active_idx]
            sample_weights = sample_weights / sample_weights.sum()  # 归一化
            
            activated_experts.append(active_idx)
            activated_weights.append(sample_weights)
        
        # 3. 打印调试信息（可选）
        confs_cpu = confs.detach().cpu().view(-1)
        weight_dict = {nt: float(confs_cpu[j]) for j, nt in enumerate(self.noise_list)}
        active_idx_0 = activated_experts[0].cpu().tolist()  # 第一个样本的激活专家索引
        active_names_0 = [self.noise_list[idx] for idx in active_idx_0]
        logging.info(f"[MoE] 噪声置信度: {weight_dict} | 激活专家: {active_names_0} | 阈值: {self.threshold}")
        
        # 4. 计算所有专家的输出（提前计算，后续只取激活的）
        expert_outs = []
        for nt in self.noise_list:
            out_nt = self.experts[nt](x)  # (B,4,64,64)
            expert_outs.append(out_nt)
        expert_outs = torch.stack(expert_outs, dim=1)  # (B,5,4,64,64)
        
        # 5. 仅激活的专家参与加权融合
        fused = torch.zeros_like(x)  # (B,4,64,64)
        for b in range(B):
            # 当前样本的激活专家索引和归一化权重
            active_idx = activated_experts[b]  # (K,) K为激活专家数
            weights = activated_weights[b]     # (K,)
            
            # 提取激活专家的输出并加权
            active_outs = expert_outs[b, active_idx]  # (K,4,64,64)
            weights = weights.view(-1, 1, 1, 1)       # (K,1,1,1)
            fused[b] = (active_outs * weights).sum(dim=0)  # (4,64,64)

        if is_single:
            fused = fused.squeeze(0)
        return fused

# 3) 从 Stage3 ckpt 加载微调后的多专家 MoE 权重
def load_moe_from_stage3(
    ckpt,
    noise_list,
    threshold=0.475,
    width: int = 32,
    device: str = "cuda"
) -> MultiExpertDenoiser:
    """
    从 Stage3 的 ckpt 中恢复多专家去噪网络（新增阈值参数）
    """
    if "moe_denoiser" not in ckpt:
        raise KeyError("Stage3 权重文件中未找到多专家去噪网络权重（'moe_denoiser' 字段）")

    moe = MultiExpertDenoiser(
        noise_list=noise_list,
        threshold=threshold,  # 传入训练时的阈值
        width=width
    ).to(device)
    moe.load_state_dict(ckpt["moe_denoiser"], strict=True)
    moe.eval()
    for p in moe.parameters():
        p.requires_grad = False

    total_params = sum(p.numel() for p in moe.parameters())
    logging.info(f"多专家 MoE 去噪网络（Stage3 微调后）加载成功，参数总量: {total_params/1e6:.3f} M | 激活阈值: {threshold}")
    return moe

Denoising_model = load_moe_from_stage3(
    ckpt=stage3_ckpt,
    noise_list=NOISE_LIST,
    threshold=MOE_THRESHOLD,
    width=32,
    device=device
)

# ===================== 多专家 MoE 推理函数（沿用 infer_nafnet 接口） =====================
def infer_nafnet(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 32,
    is_training: bool = False
) -> torch.Tensor:
    """
    MoE 去噪推理函数（兼容原有接口，内部已按训练逻辑修改）
    """
    model.eval()
    model.to(device)

    # 确保是 (B,4,64,64)
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    pred_list = []
    total_samples = input_tensor.shape[0]
    num_batches = (total_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_samples)
            batch_data = input_tensor[start_idx:end_idx].to(device, non_blocking=True, dtype=torch.float32)

            pred = model(batch_data)
            pred_list.append(pred)

    pred_tensor = torch.cat(pred_list, dim=0)
    if squeeze_back:
        pred_tensor = pred_tensor.squeeze(0)

    assert pred_tensor.shape == input_tensor.shape, \
        f"output_coco8K_4096 shape {pred_tensor.shape} != Input shape {input_tensor.shape}"
    return pred_tensor

# ===================== 其余函数（latent转图像、鲁棒性测试等）保持不变 =====================
def safe_latent_to_image(latent, save_path, pipe, lossy=True, quant_colors=256):
    """
    安全的latent转图像函数，支持无损/有损PNG保存
    Args:
        latent: 输入latent张量
        save_path: 保存路径（xxx.png）
        pipe: SD模型的pipe
        lossy: 是否启用有损压缩（默认True）
        quant_colors: 量化颜色数（1-256，默认256，越小体积越小但损失越大）
    """
    with torch.no_grad():
        latent = latent.to(device=pipe.vae.device, dtype=pipe.vae.dtype)
        latent_scaled = latent / pipe.vae.config.scaling_factor
        decoder_tensor = pipe.vae.decode(latent_scaled).sample
        decoder_tensor = (decoder_tensor / 2 + 0.5).clamp(0, 1)
        decoder_tensor = decoder_tensor.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((decoder_tensor * 255).astype(np.uint8))
        
        # 有损PNG处理：颜色量化+优化
        if lossy:
            # 转为调色板模式（颜色量化核心步骤）
            image = image.convert(
                'P', 
                palette=Image.ADAPTIVE,  # 自适应调色板，保留视觉重要颜色
                colors=quant_colors      # 量化到指定颜色数
            )
            # 启用PNG优化（压缩率提升，仍无损保存量化结果）
            image.save(save_path, optimize=True, compress_level=9)
        else:
            # 保留无损PNG（原逻辑）
            image.save(save_path, compress_level=6)
    return image

def capture_normal(step, timestep, latents, run_data, num_steps):
    if step == num_steps - 2:
        model_output_coco8K_4096 = pipe.unet(latents, timestep, encoder_hidden_states=text_embeddings).sample
        t = int(timestep.item())
        max_t = len(pipe.scheduler.alphas_cumprod) - 1
        t = max(0, min(t, max_t))
        alpha_cumprod = pipe.scheduler.alphas_cumprod[t]
        sqrt_alpha = torch.sqrt(alpha_cumprod).to(latents.device)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha_cumprod).to(latents.device)
        pred_original = (latents - sqrt_one_minus_alpha * model_output_coco8K_4096) / sqrt_alpha
        timesteps = pipe.scheduler.timesteps
        t_index = torch.where(timesteps == timestep)[0].item()
        timestep_prev = timesteps[t_index + 1]
        t = int(timestep.item())
        t_prev = int(timestep_prev.item())
        alpha_cumprod_t = pipe.scheduler.alphas_cumprod[t]
        alpha_cumprod_prev = pipe.scheduler.alphas_cumprod[t_prev]

        sigma_t = ((1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_prev)) * (eta ** 2)
        local_means = torch.sqrt(alpha_cumprod_prev) * pred_original + torch.sqrt(1 - alpha_cumprod_prev - sigma_t) * model_output_coco8K_4096

        if "local_means_list" not in run_data:
            run_data["local_means_list"] = []
        run_data["local_means_list"].append(local_means.detach().to("cpu", dtype=torch.float32).clone())

        if "local_vars_list" not in run_data:
            run_data["local_vars_list"] = []
        run_data["local_vars_list"].append(sigma_t.detach().to("cpu", dtype=torch.float32).clone())

    if step == num_steps - 1:
        run_data["z_t_normal"] = latents.detach().to("cpu", dtype=torch.float32).clone()
        run_data["local_means"] = torch.stack(run_data["local_means_list"]).mean(dim=0)
        run_data["local_vars"] = torch.stack(run_data["local_vars_list"]).mean(dim=0)

def robust_test(factor, attack_layer, img_tensor, pipe, tensor1, message, image_tensor, Q):
    with torch.no_grad():
        attack_fn = getattr(attack, f"{attack_layer}_attack")
        jpeg_tensor = attack_fn(img_tensor, pipe, factor)

        # 攻击结果转 latent（RGB -> latent 或直接是 latent）
        if jpeg_tensor.dim() == 4 and jpeg_tensor.shape[1] == 3:
            temp_vae = pipe.vae.to(device=jpeg_tensor.device, dtype=jpeg_tensor.dtype)
            jpeg_latents = temp_vae.encode(jpeg_tensor).latent_dist.mode()
            jpeg_latents = jpeg_latents * pipe.vae.config.scaling_factor
            del temp_vae
        else:
            jpeg_latents = jpeg_tensor

    # 多专家 MoE 去噪
    jpeg_latents = jpeg_latents.to('cuda', dtype=torch.float32)
    jpeg_latents_denoised = infer_nafnet(
        model=Denoising_model,
        input_tensor=jpeg_latents,
        device='cuda',
        batch_size=8,
        is_training=False
    )

    # 计算 MSE（attack latent vs 原始 latent）
    with torch.no_grad():
        img_lat = image_tensor.detach().clone()
        z_lat = jpeg_latents.detach().clone()

        while img_lat.dim() < z_lat.dim():
            img_lat = img_lat.unsqueeze(0)
        while z_lat.dim() < img_lat.dim():
            z_lat = z_lat.unsqueeze(0)

        img_lat = img_lat.to(device=z_lat.device, dtype=z_lat.dtype)

        min_shape = tuple(min(a, b) for a, b in zip(img_lat.shape, z_lat.shape))
        slices = tuple(slice(0, m) for m in min_shape)
        img_crop = img_lat[slices]
        z_crop = z_lat[slices]

        mse = torch.mean((img_crop - z_crop) ** 2).item()
        logging.info(f"{factor},{attack_layer} {mse:.6f}")

    # 用去噪后的 latent 提取消息
    message_ex = model.reveal(tensor1, jpeg_latents_denoised)
    B2, C2, H2, W2 = message_ex.shape
    MQ_hat_flat = message_ex.view(-1, H2, W2)
    Q_inv = torch.linalg.inv(Q)
    M_hat_flat = torch.matmul(MQ_hat_flat, Q_inv)
    M_hat = M_hat_flat.view(B2, C2, H2, W2)
    message_ex = torch.sigmoid(M_hat)
    message_hat = (message_ex > 0.5).float()
    correct_acc = (message_hat == message).float().mean()
    logging.info(f"{attack_layer}<{factor}>消息提取准确率: {correct_acc:.4f} ")

def robust_test_multi(
        attack_layers: list,
        attack_factors: list,
        img_tensor,
        pipe,
        tensor1,
        message,
        image_tensor,
        Q
):
    with torch.no_grad():
        attack_fn = getattr(attack, f"mutil_attack")
        attack_tensor = attack_fn(img_tensor, pipe, attack_layers, attack_factors)
        if attack_tensor.dim() == 4 and attack_tensor.shape[1] == 3:
            temp_vae = pipe.vae.to(device=attack_tensor.device, dtype=attack_tensor.dtype)
            jpeg_latents = temp_vae.encode(attack_tensor).latent_dist.mode()
            jpeg_latents = jpeg_latents * pipe.vae.config.scaling_factor
            del temp_vae
        else:
            jpeg_latents = attack_tensor

    # 多专家 MoE 去噪
    jpeg_latents = jpeg_latents.to('cuda', dtype=torch.float32)
    jpeg_latents_denoised = infer_nafnet(
        model=Denoising_model,
        input_tensor=jpeg_latents,
        device='cuda',
        batch_size=8,
        is_training=False
    )

    # 计算 MSE（attack latent vs 原始 latent）
    with torch.no_grad():
        img_lat = image_tensor.detach().clone()
        z_lat = jpeg_latents.detach().clone()

        while img_lat.dim() < z_lat.dim():
            img_lat = img_lat.unsqueeze(0)
        while z_lat.dim() < img_lat.dim():
            z_lat = z_lat.unsqueeze(0)

        img_lat = img_lat.to(device=z_lat.device, dtype=z_lat.dtype)

        min_shape = tuple(min(a, b) for a, b in zip(img_lat.shape, z_lat.shape))
        slices = tuple(slice(0, m) for m in min_shape)
        img_crop = img_lat[slices]
        z_crop = z_lat[slices]

        mse = torch.mean((img_crop - z_crop) ** 2).item()

    # 用去噪后的 latent 提取消息
    message_ex = model.reveal(tensor1, jpeg_latents_denoised)
    B2, C2, H2, W2 = message_ex.shape
    MQ_hat_flat = message_ex.view(-1, H2, W2)
    Q_inv = torch.linalg.inv(Q)
    M_hat_flat = torch.matmul(MQ_hat_flat, Q_inv)
    M_hat = M_hat_flat.view(B2, C2, H2, W2)
    message_ex = torch.sigmoid(M_hat)
    message_hat = (message_ex > 0.5).float()
    correct_acc = (message_hat == message).float().mean()
    logging.info(f"多攻击组合提取准确率: {correct_acc:.4f} ")

def generate_orthogonal_matrix(n: int = 64, seed: int | None = None, device: str = "cuda", dtype=torch.float32) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    A = torch.randn((n, n), dtype=dtype, device=device)
    Q, R = torch.linalg.qr(A)
    d = torch.sign(torch.diag(R))
    d[d == 0] = 1.0
    Q = Q * d.unsqueeze(0)
    return Q

# ===================== 主处理逻辑 =====================
p = 0
for prompt in prompts:
    logging.info(f"开始处理第{p+1}个prompt")
    p += 1

    # 预处理文本
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(pipe.device)
    with torch.no_grad():
        text_embeddings = pipe.text_encoder(text_inputs.input_ids)[0]
    logging.info("prompt处理成功")

    # 初始化生成器和运行数据
    gen = torch.Generator(device="cuda").manual_seed(p+12345)
    run_data = {}

    # 正常扩散过程
    pipe(
        prompt,
        num_inference_steps=num_steps,
        generator=gen,
        output_type="latent",  # 修正：原代码是output_coco8K_4096_type，应为output_type
        callback=lambda s, t, l: capture_normal(s, t, l, run_data, num_steps),
        callback_steps=1,
    )
    assert all(key in run_data for key in ["z_t_normal", "local_means", "local_vars"]), "数据捕获失败"

    # 保存正常图片
    pipe.vae = pipe.vae.to(device=vae_device, dtype=vae_dtype)
    image_path = f"./output_coco8K_4096/cover/image/{p}.png"
    safe_latent_to_image(run_data["z_t_normal"].to(device="cuda", dtype=torch.float32), image_path, pipe)
    logging.info(f"正常图像已经保存到{image_path}")

    # ========== 修改4：生成单通道4096bit秘密消息 ==========
    # 原：message = (torch.rand(1, 4, 64, 64, device="cuda") < 0.5).float()
    message = (torch.rand(1, 1, 64, 64, device="cuda") < 0.5).float()  # 1通道，4096bit
    
    # 使用训练好的Q（或随机生成，和训练时保持一致）
    Q = generate_orthogonal_matrix(64, device=device)
    B, C, H, W = message.shape  # 现在C=1
    M_flat = message.view(-1, H, W)  # (1*1, 64, 64) = (1,64,64)
    MQ_flat = torch.matmul(M_flat, Q)
    MQ = MQ_flat.view(B, C, H, W)  # (1,1,64,64)

    # 嵌入消息（HideNet）
    tensor1 = run_data["z_t_normal"].to(device='cuda', dtype=torch.float32)
    if tensor1.dim() == 4 and tensor1.shape[0] == 1:
        tensor1 = tensor1.squeeze(0)
    tensor1 = tensor1.unsqueeze(0).to(device)  # (1,4,64,64)
    img_tensor, _ = model.hide(tensor1, MQ)  # MQ是(1,1,64,64)，拼接后输入为5通道

    # 保存载密图片
    pipe.vae = pipe.vae.to(device=vae_device, dtype=vae_dtype)
    stego_path = f"./output_coco8K_4096/stego/image/{p}.png"
    safe_latent_to_image(img_tensor, stego_path, pipe)
    logging.info(f"载密图像已经保存到{stego_path}")

    # 提取消息：cover->latent->多专家 MoE 去噪->RevealNet
    z_enc_normal = utils.image_to_latent(Image.open(stego_path).convert("RGB"), pipe).to('cuda')
    z_enc_normal_denoising = infer_nafnet(
        model=Denoising_model,
        input_tensor=z_enc_normal,
        device='cuda',
        batch_size=8,
        is_training=False
    )

    # 使用去噪后的 latent 提取消息
    message_ex = model.reveal(tensor1, z_enc_normal_denoising)
    B2, C2, H2, W2 = message_ex.shape  # C2=1
    MQ_hat_flat = message_ex.view(-1, H2, W2)  # (1,64,64)
    Q_inv = torch.linalg.inv(Q)
    M_hat_flat = torch.matmul(MQ_hat_flat, Q_inv)
    M_hat = M_hat_flat.view(B2, C2, H2, W2)  # (1,1,64,64)
    message_ex = torch.sigmoid(M_hat)
    message_hat = (message_ex > 0.5).float()
    correct_acc = (message_hat == message).float().mean()
    logging.info(f"去噪后消息提取准确率: {correct_acc:.4f}")

    # 可选：启用鲁棒性测试
    logging.info("开始执行鲁棒性测试")
    test_cases = [
        (90, 'jpeg'), (70, 'jpeg'), (50, 'jpeg'),
        (1.5, 'resize'), (1.25, 'resize'), (0.75, 'resize'), (0.5, 'resize'),
        (3, 'mblur'), (5, 'mblur'), (7, 'mblur'),
        (3, 'gblur'), (5, 'gblur'), (7, 'gblur'),
        (0.01, 'awgn'), (0.05, 'awgn'), (0.1, 'awgn')
    ]
    for factor, attack_layer in test_cases:
        robust_test(factor, attack_layer, img_tensor, pipe, tensor1, message, img_tensor, Q)

    # 可选：启用多攻击组合测试
    # logging.info("开始执行多攻击组合测试")
    # attack_layers = ['jpeg', 'gblur', 'resize']
    # attack_factors = [90, 3, 1.5]
    # robust_test_multi(
    #     attack_layers,
    #     attack_factors,
    #     img_tensor, pipe, tensor1, message, img_tensor, Q
    # )

# 清理资源
torch.cuda.empty_cache()
logging.info("所有prompt处理完成")