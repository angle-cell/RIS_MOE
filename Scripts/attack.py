import robust_eval
import torch
from PIL import Image

def _vae_device_dtype(vae):
    p = next(vae.parameters())
    return p.device, p.dtype

@torch.no_grad()
def _decode_pixels_from_latents(pipe, latents):
    # latents -> 与 VAE 对齐，并做 SD 的 scaling 修正
    device, dtype = _vae_device_dtype(pipe.vae)
    z = (latents / pipe.vae.config.scaling_factor).to(device=device, dtype=dtype)
    x = pipe.vae.decode(z).sample  # 输出范围约 [-1, 1]
    return x.clamp(-1, 1)

def jpeg_attack(latent,pipe,factor):
    # with torch.no_grad():
    #     z_normal_cuda = latent.to("cuda", dtype=pipe.vae.dtype) / pipe.vae.config.scaling_factor
    #     decoded_normal = pipe.vae.decode(z_normal_cuda).sample
    
    # x_normal = (decoded_normal / 2 + 0.5).clamp(0, 1)
    # x_normal = decoded_normal.clamp(-1, 1)
    x_normal = _decode_pixels_from_latents(pipe, latent)
    attacked_samples = robust_eval.jpeg(x_normal, factor=factor, tmp_image_name='tmp_jpeg')
    return attacked_samples   

def resize_attack(latent,pipe,factor):
    # with torch.no_grad():
    #     z_normal_cuda = latent.to("cuda", dtype=pipe.vae.dtype) / pipe.vae.config.scaling_factor
    #     decoded_normal = pipe.vae.decode(z_normal_cuda).sample
    
    # x_normal = (decoded_normal / 2 + 0.5).clamp(0, 1)
    # x_normal = decoded_normal.clamp(-1, 1)
    x_normal = _decode_pixels_from_latents(pipe, latent)

    attacked_samples = robust_eval.resize(x_normal, factor=factor, tmp_image_name='tmp_resize')
    return attacked_samples   

def mblur_attack(latent,pipe,factor):
    # with torch.no_grad():
    #     z_normal_cuda = latent.to("cuda", dtype=pipe.vae.dtype) / pipe.vae.config.scaling_factor
    #     decoded_normal = pipe.vae.decode(z_normal_cuda).sample
    
    # x_normal = (decoded_normal / 2 + 0.5).clamp(0, 1)
    # # x_normal = decoded_normal.clamp(-1, 1)
    x_normal = _decode_pixels_from_latents(pipe, latent)

    attacked_samples = robust_eval.mblur(x_normal, factor=factor, tmp_image_name='tmp_mblur')
    return attacked_samples   

def gblur_attack(latent,pipe,factor):
    # with torch.no_grad():
    #     z_normal_cuda = latent.to("cuda", dtype=pipe.vae.dtype) / pipe.vae.config.scaling_factor
    #     decoded_normal = pipe.vae.decode(z_normal_cuda).sample

    # x_normal = (decoded_normal / 2 + 0.5).clamp(0, 1)
    # # x_normal = decoded_normal.clamp(-1, 1)
    x_normal = _decode_pixels_from_latents(pipe, latent)

    attacked_samples = robust_eval.gblur(x_normal, factor=factor, tmp_image_name='tmp_gblur')
    return attacked_samples   

def awgn_attack(latent,pipe,factor):
    # with torch.no_grad():
    #     z_normal_cuda = latent.to("cuda", dtype=pipe.vae.dtype) / pipe.vae.config.scaling_factor
    #     decoded_normal = pipe.vae.decode(z_normal_cuda).sample

    # x_normal = (decoded_normal / 2 + 0.5).clamp(0, 1)
    # # x_normal = decoded_normal.clamp(-1, 1)
    x_normal = _decode_pixels_from_latents(pipe, latent)

    attacked_samples = robust_eval.awgn(x_normal, factor=factor, tmp_image_name='tmp_awgn')
    return attacked_samples  

def mutil_attack(latent,pipe,attack_list,factor_list):
    attacked_samples = _decode_pixels_from_latents(pipe, latent)
    for i in range(len(attack_list)):
        attack = attack_list[i]
        factor = factor_list[i]
        if attack == 'jpeg':
            attacked_samples = robust_eval.jpeg(attacked_samples, factor=factor, tmp_image_name='tmp_jpeg')
        elif attack == 'resize':
            attacked_samples = robust_eval.resize(attacked_samples, factor=factor, tmp_image_name='tmp_resize')
        elif attack == 'mblur':
            attacked_samples = robust_eval.mblur(attacked_samples, factor=factor, tmp_image_name='tmp_mblur')
        elif attack == 'gblur':
            attacked_samples = robust_eval.gblur(attacked_samples, factor=factor, tmp_image_name='tmp_gblur')
        elif attack == 'awgn':
            attacked_samples = robust_eval.awgn(attacked_samples, factor=factor, tmp_image_name='tmp_awgn')
    return attacked_samples



def tensor_to_latent(image_tensor, pipe):
    """图像编码回潜变量（无错误覆盖）"""
    img_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=pipe.vae.dtype)
    with torch.no_grad():
        posterior = pipe.vae.encode(img_tensor * 2.0 - 1.0)
        z_enc = posterior.latent_dist.mode() * pipe.vae.config.scaling_factor
    return z_enc.detach().to("cpu", dtype=torch.float32).clone()

