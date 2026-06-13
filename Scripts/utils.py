import torch
import numpy as np
import random
from PIL import Image
import imageio.v3 as iio

import PIL
from matplotlib import pyplot as plt
import torchvision.transforms as T
# import robust_eval

# 将图像编码回隐空间
# def image_to_latent(img, pipe):
#     """图像编码回潜变量（无错误覆盖）"""
#     img_np = np.array(img).astype(np.float32) / 255.0
#     img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=pipe.vae.dtype)
#     with torch.no_grad():
#         posterior = pipe.vae.encode(img_tensor * 2.0 - 1.0)
#         z_enc = posterior.latent_dist.mode() * pipe.vae.config.scaling_factor
#     return z_enc.detach().to("cpu", dtype=torch.float32).clone()

# def latent_to_image(latent,image_path,pipe):
#     with torch.no_grad():
#         device = latent.device
#         dtype = latent.dtype

#         pipe.vae = pipe.vae.to(device=device, dtype=dtype)
#         z_normal_cuda = latent.to("cuda", dtype=pipe.vae.dtype) / pipe.vae.config.scaling_factor
#         decoded_normal = pipe.vae.decode(z_normal_cuda).sample

    
#     x_normal = (decoded_normal / 2 + 0.5).clamp(0, 1)
#     x_normal_np = (x_normal.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype("uint8")
#     # normal_img_path = f"./output/image_example.png"
#     normal_img_path = image_path
#     Image.fromarray(x_normal_np[0]).save(normal_img_path, optimize=True)

# def image_to_latent(img, pipe):
#     """图像编码回潜变量（优化版：降低VAE编码损失）"""
#     # 1. 严格统一设备和数据类型，避免频繁转换导致的精度损失
#     device = pipe.vae.device
#     dtype = pipe.vae.dtype
#     img_np = np.array(img, dtype=np.float32) / 255.0  # 保持float32精度
    
#     # 2. 图像张量转换：直接指定设备和dtype，减少中间转换
#     img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
#     img_tensor = img_tensor.to(device=device, dtype=dtype)
    
#     # 3. 标准化：严格匹配VAE训练时的输入范围（[-1, 1]），避免偏移
#     img_tensor = img_tensor * 2.0 - 1.0
    
#     with torch.no_grad():
#         posterior = pipe.vae.encode(img_tensor)
#         # 优化：不只用mode（均值），而是从后验分布采样（保留更多信息，降低还原损失）
#         # mode是确定型，采样更符合VAE的生成特性，还原度更高
#         z_enc = posterior.latent_dist.sample()  # 替换mode()为sample()
#         # 4. 约束latent数值范围：匹配VAE训练的latent分布（关键！）
#         z_enc = z_enc * pipe.vae.config.scaling_factor
#         # 限制latent范围，避免超出VAE训练时的分布导致解码失真
#         z_enc = torch.clamp(z_enc, -pipe.vae.config.scaling_factor*2, pipe.vae.config.scaling_factor*2)
    
#     # 5. 仅在最后转CPU，且保留float32（避免dtype降级）
#     return z_enc.detach().to("cpu", dtype=torch.float32).clone()

# def latent_to_image(latent, image_path, pipe):
#     """潜变量解码为图像（优化版：降低VAE解码损失）"""
#     # 1. 统一设备和dtype，避免模型频繁切换设备
#     device = pipe.vae.device
#     dtype = pipe.vae.dtype
#     pipe.vae = pipe.vae.to(device=device, dtype=dtype)
    
#     with torch.no_grad():
#         # 2. Latent反归一化：先转设备，再做除法，减少精度损失
#         z_normal = latent.to(device=device, dtype=dtype) / pipe.vae.config.scaling_factor
#         # 3. 约束latent范围，避免解码时数值溢出
#         # z_normal = torch.clamp(z_normal, -2.0, 2.0)  # 匹配SD VAE的latent常规范围
        
#         # 4. 解码：添加torch.compile加速（可选，不影响损失但提升稳定性）
#         decoded_normal = pipe.vae.decode(z_normal).sample
#         # 确保解码结果在设备/dtype上统一
#         decoded_normal = decoded_normal.to(device=device, dtype=dtype)
    
#     # 5. 反归一化优化：调整计算顺序，减少数值误差
#     # 先在GPU上完成反归一化，再转CPU（减少数据传输时的精度损失）
#     x_normal = (decoded_normal / 2.0 + 0.5).clamp(0.0, 1.0)  # 用float精度的0.0/1.0，避免整数隐式转换
    
#     # 6. 后处理优化：减少量化损失
#     # 先转float32再量化，避免uint8的截断误差
#     x_normal_np = x_normal.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
#     # 增加轻微的高斯模糊（可选，减少量化噪点）
#     # from scipy.ndimage import gaussian_filter
#     # x_normal_np = gaussian_filter(x_normal_np, sigma=0.5)  # 如需启用，取消注释并导入
    
#     # 量化为uint8时，使用round而非直接截断，降低量化损失
#     x_normal_np = (x_normal_np * 255.0).round().astype(np.uint8)
    
#     # 7. 保存优化：提升图像质量，减少压缩损失
#     img = Image.fromarray(x_normal_np[0])
#     # 保存时调整参数，避免过度压缩
#     img.save(
#         image_path,
#         optimize=True,
#         quality=95,  # 提升质量（默认可能偏低）
#         subsampling=0  # 关闭色度子采样，保留更多色彩信息
#     )


def latent_to_image(latent, image_path, pipe):
    with torch.no_grad():
        z = latent.to(device='cuda',dtype=torch.float32) / pipe.vae.config.scaling_factor
        decoder_tensor = pipe.vae.decode(z).sample.to(device='cuda',dtype=torch.float32)
    x_normal = (decoder_tensor / 2 + 0.5).clamp(0, 1)
    x_normal_np = (x_normal.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype("uint8")
    img = Image.fromarray(x_normal_np[0])
    img.save(
        image_path,
        optimize=True
    )

    img = Image.open(image_path).convert("RGB")

def image_to_latent(img, pipe):
    img_np = np.array(img).astype(np.float32) / 255.0
    # img_tensor = torch.from_numpy(img_np).permute(0, 3, 1, 2).to("cuda", dtype=pipe.vae.dtype)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=pipe.vae.dtype)
    with torch.no_grad():
        encoder_tensor = pipe.vae.encode(img_tensor * 2 - 1)
        z_enc_normal = encoder_tensor.latent_dist.mode() * pipe.vae.config.scaling_factor
    return z_enc_normal.detach().to("cpu", dtype=torch.float32).clone()



def show_torch_img(img):
    img = to_np_image(img)
    plt.imshow(img)
    plt.axis("off")


def to_np_image(all_images):
    all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
    return all_images


def tensor_to_pil(tensor_imgs):
    if type(tensor_imgs) == list:
        tensor_imgs = torch.cat(tensor_imgs)
    tensor_imgs = (tensor_imgs / 2 + 0.5).clamp(0, 1)
    to_pil = T.ToPILImage()
    pil_imgs = [to_pil(img) for img in tensor_imgs]
    return pil_imgs


def pil_to_tensor(pil_imgs):
    to_torch = T.ToTensor()
    if type(pil_imgs) == PIL.Image.Image:
        tensor_imgs = to_torch(pil_imgs).unsqueeze(0) * 2 - 1
    elif type(pil_imgs) == list:
        tensor_imgs = torch.cat([to_torch(pil_imgs).unsqueeze(0) * 2 - 1 for img in pil_imgs]).to(device)
    else:
        raise Exception("Input need to be PIL.Image or list of PIL.Image")
    return tensor_imgs


def add_margin(pil_img, top=0, right=0, bottom=0,
               left=0, color=(255, 255, 255)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)

    result.paste(pil_img, (left, top))
    return result


def image_grid(imgs, rows=1, cols=None,
               size=None):
    if type(imgs) == list and type(imgs[0]) == torch.Tensor:
        imgs = torch.cat(imgs)
    if type(imgs) == torch.Tensor:
        imgs = tensor_to_pil(imgs)

    if not size is None:
        imgs = [img.resize((size, size)) for img in imgs]
    if cols is None:
        cols = len(imgs)
    assert len(imgs) >= rows * cols

    top = 20
    w, h = imgs[0].size
    delta = 0
    if len(imgs) > 1 and not imgs[1].size[1] == h:
        delta = top
        h = imgs[1].size[1]
    grid = Image.new('RGB', size=(cols * w, rows * h + delta))
    for i, img in enumerate(imgs):
        if not delta == 0 and i > 0:
            grid.paste(img, box=(i % cols * w, i // cols * h + delta))
        else:
            grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid


# 读取一张图像 返回值大小 [1 3 512 512]
def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path).convert('RGB'))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0)

    return image

def gray_code(n):
    """
    格雷码生成函数
    :param n: 格雷码位数
    :return: 格雷码列表
    """
    if n == 1:
        return ['0', '1']
    else:
        res = []
    old_gray_code = gray_code(n - 1)
    for i in range(len(old_gray_code)):
        res.append('0' + old_gray_code[i])
    for i in range(len(old_gray_code) - 1, -1, -1):
        res.append('1' + old_gray_code[i])
    return res


def generate_orthogonal(n=64, seed=None, ensure_det1=False):
    """
    生成 n x n 正交矩阵。使用 QR 分解并修正符号以消除 R 对角线的负号引入的不确定性。
    若 ensure_det1=True，则确保行列式为 +1（特殊正交矩阵）。
    """
    if seed is not None:
        np.random.seed(seed)
    A = np.random.randn(n, n)
    Q, R = np.linalg.qr(A)
    # 统一 R 对角线的符号，保证 Q 的列方向确定
    diag_sign = np.sign(np.diag(R))
    diag_sign[diag_sign == 0] = 1
    Q = Q @ np.diag(diag_sign)
    if ensure_det1 and np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


