from typing import Tuple
import torch
from sampling.epsilon_net import ddim_step, EpsilonNet
import numpy as np
from PIL import Image
import os
from PIL import Image, ImageFont, ImageDraw, ImageOps
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from evaluation.perception import LPIPS

def dps_dpms(initial_noise: torch.Tensor,
            inverse_problem: Tuple,
            epsilon_net: EpsilonNet,
            lam: float = 1.0,
            gamma: float = 1.0,
            eta: float = 1.0,
            k: int = 10):
    
    obs, H_func, std = inverse_problem
    A = H_func.H
    shape = (initial_noise.shape[0], *(1,) * len(initial_noise.shape[1:]))

    def pot_func(x):
        return -torch.norm(obs.reshape(1, -1) - A(x)) ** 2.0

    def error(x):
        return torch.norm(obs.reshape(1, -1) - A(x), dim=-1)

    sample = initial_noise
    for i in range(len(epsilon_net.timesteps) - 1, k, -1):
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]

        sample.requires_grad_()
        e_t = epsilon_net.predict_x0(sample, t)
        grad_norm = error(e_t).reshape(*shape)
        pot_val = pot_func(e_t)
        grad_pot = torch.autograd.grad(pot_val, sample)[0]

        sample = ddim_step(
            x=sample, epsilon_net=epsilon_net, t=t, t_prev=t_prev, eta=eta, e_t=e_t
        ).detach()

        # gradient step
        grad_pot = gamma * grad_pot / grad_norm
        sample = sample + grad_pot

    for i in range(k, 1, -1):
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]
        sample.requires_grad_()
        e_t = epsilon_net.predict_x0(sample, t)
        grad_value, alpha_t = epsilon_net.approximate_grad_log_likelihood(x_t = sample, t=t_prev, H_funcs=H_func, y=obs, noise_std=std)

        sample = ddim_step(
            x=sample, epsilon_net=epsilon_net, t=t, t_prev=t_prev, eta=eta, e_t=e_t
        ).detach()

        # gradient step
        sample = sample + lam * grad_value * (1-alpha_t)/torch.sqrt(alpha_t)

    # last diffusion step
    sample.requires_grad_()
    grad_value, alpha_t = epsilon_net.approximate_grad_log_likelihood(x_t = sample, t=1, H_funcs=H_func, y=obs, noise_std=std)

    sample = epsilon_net.predict_x0(sample, epsilon_net.timesteps[1]) + lam * grad_value * (1-alpha_t)/torch.sqrt(alpha_t)

    return sample.detach()

def dps_dpms_save(initial_noise: torch.Tensor,
            inverse_problem: Tuple,
            epsilon_net: EpsilonNet,
            lam: float = 1.0,
            gamma: float = 1.0,
            eta: float = 1.0,
            k: int = 10,
            output_path: str = None,
            interval = 10):
    
    obs, H_func, std = inverse_problem
    A = H_func.H
    shape = (initial_noise.shape[0], *(1,) * len(initial_noise.shape[1:]))

    def pot_func(x):
        return -torch.norm(obs.reshape(1, -1) - A(x)) ** 2.0

    def error(x):
        return torch.norm(obs.reshape(1, -1) - A(x), dim=-1)

    sample = initial_noise
    samples = []
    samples.append(sample)
    lpips =  LPIPS()
    
    for i in range(len(epsilon_net.timesteps) - 1, k, -1):
        # save image every interval steps
        # make the progress folder
        if not os.path.exists(os.path.join(output_path, "progress")):
            os.makedirs(os.path.join(output_path, "progress"))
        
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]

        sample.requires_grad_()
        e_t = epsilon_net.predict_x0(sample, t)
        grad_norm = error(e_t).reshape(*shape)
        pot_val = pot_func(e_t)
        grad_pot = torch.autograd.grad(pot_val, sample)[0]

        sample = ddim_step(
            x=sample, epsilon_net=epsilon_net, t=t, t_prev=t_prev, eta=eta, e_t=e_t
        ).detach()

        # gradient step
        grad_pot = gamma * grad_pot / grad_norm
        sample = sample + grad_pot

        samples.append(sample)      
        if i % interval == 0 and i > 0:
            
            sample_i = samples[-1].squeeze(0).detach().cpu()
            sample_i_c = sample_i.clamp(-1, 1).unsqueeze(0)
            sample_i = ((sample_i.permute(1, 2, 0).clamp(-1, 1) + 1.0) * 127.5).numpy().astype(np.uint8)
            # sample_i_n = (sample_i - sample_i.min())/(sample_i.max() - sample_i.min())

            sample_i_1 = samples[-2].squeeze(0).detach().cpu()
            sample_i_1_c = sample_i_1.clamp(-1, 1)
            sample_i_1 = ((sample_i_1.permute(1, 2, 0).clamp(-1, 1) + 1.0) * 127.5).numpy().astype(np.uint8)
            # sample_i_1_n = (sample_i_1 - sample_i_1.min())/(sample_i_1.max() - sample_i_1.min())
            # sample_i_sr_1 = sample_i[1::2, 1::2, 0]
            # sample_i_sr_2 = sample_i[0::2, 0::2, 0]
            # print(sample_i_1_n.shape)
            ssim_i = str(round(ssim(sample_i_1[..., 0], sample_i[..., 0]), 3))
            psnr_i = str(round(psnr(sample_i_1, sample_i), 3))
            lpips_i = str(round(lpips.score(sample_i_c, sample_i_1_c).item(), 3))
            
            img_pil = Image.fromarray(sample_i)
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype("./font/LiberationSans-Bold.ttf", 30)


            draw.text((0,0), f"Iter {i}\nSSIM:"+ssim_i+"\n"+"PSNR:"+psnr_i+"dB\n"+"LPIPS:"+lpips_i ,(255,255,255), font= font)
        
            #save image
            img_pil.save(os.path.join(output_path, f"output_{str(i).zfill(4)}.png"))

    for i in range(k, 1, -1):
        # save image every interval step

        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]
        sample.requires_grad_()
        e_t = epsilon_net.predict_x0(sample, t)
        grad_value, alpha_t = epsilon_net.approximate_grad_log_likelihood(x_t = sample, t=t_prev, H_funcs=H_func, y=obs, noise_std=std)

        sample = ddim_step(
            x=sample, epsilon_net=epsilon_net, t=t, t_prev=t_prev, eta=eta, e_t=e_t
        ).detach()

        # gradient step
        sample = sample + lam * grad_value * (1-alpha_t)/torch.sqrt(alpha_t)
        
        samples.append(sample)
        if i % interval == 0 and i > 0:
            
            sample_i = samples[-1].squeeze(0).detach().cpu()
            sample_i_c = sample_i.clamp(-1, 1).unsqueeze(0)
            sample_i = ((sample_i.permute(1, 2, 0).clamp(-1, 1) + 1.0) * 127.5).numpy().astype(np.uint8)
            # sample_i_n = (sample_i - sample_i.min())/(sample_i.max() - sample_i.min())

            sample_i_1 = samples[-2].squeeze(0).detach().cpu()
            sample_i_1_c = sample_i_1.clamp(-1, 1)
            sample_i_1 = ((sample_i_1.permute(1, 2, 0).clamp(-1, 1) + 1.0) * 127.5).numpy().astype(np.uint8)
            # sample_i_1_n = (sample_i_1 - sample_i_1.min())/(sample_i_1.max() - sample_i_1.min())
            # sample_i_sr_1 = sample_i[1::2, 1::2, 0]
            # sample_i_sr_2 = sample_i[0::2, 0::2, 0]
            # print(sample_i_1_n.shape)
            ssim_i = str(round(ssim(sample_i_1[..., 0], sample_i[..., 0]), 3))
            psnr_i = str(round(psnr(sample_i_1, sample_i), 3))
            lpips_i = str(round(lpips.score(sample_i_c, sample_i_1_c).item(), 3))
            
            img_pil = Image.fromarray(sample_i)
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype("./font/LiberationSans-Bold.ttf", 30)


            draw.text((0,0), f"Iter {i}\nSSIM:"+ssim_i+"\n"+"PSNR:"+psnr_i+"dB\n"+"LPIPS:"+lpips_i ,(255,255,255), font= font)
        
            #save image
            img_pil.save(os.path.join(output_path, f"output_{str(i).zfill(4)}.png"))

    # last diffusion step
    sample.requires_grad_()
    grad_value, alpha_t = epsilon_net.approximate_grad_log_likelihood(x_t = sample, t=1, H_funcs=H_func, y=obs, noise_std=std)

    sample = epsilon_net.predict_x0(sample, epsilon_net.timesteps[1]) + lam * grad_value * (1-alpha_t)/torch.sqrt(alpha_t)

    return sample.detach()