from utils import load_epsilon_net
from sampling.dps_dpms import dps_dpms
from sampling.dps import dps
import itertools
import time
import numpy as np
from dmps.util.img_utils import clip, clear_color, normalize_np
from skimage.metrics import peak_signal_noise_ratio
import math
import os
from pathlib import Path
from utils import load_image, display_image
import matplotlib.pyplot as plt
from evaluation.perception import LPIPS
from dmps.data.dataloader import get_dataloader, get_dataset
import torchvision.transforms as transforms
import yaml
import torch
import sys

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

args = {}
args['model_config'] = './dmps/configs/model_config.yaml'
args['diffusion_config'] = './dmps/configs/diffusion_config.yaml'
args['task_config'] = './dmps/configs/sr4_config.yaml'
args['gpu'] = 0
args['save_dir'] = './saved_results'
args['seed'] = 0

device = "cuda:0"
method = "outpainting_expand"
out_path = f"saved_results/{method}"
Path(out_path).mkdir(parents=True, exist_ok=True)
diff_methods = ["dps", "dps_dpms"]
torch.set_default_device(device)
for diff_method in diff_methods:
    diff_path = out_path+"/"+diff_method
    Path(diff_path).mkdir(parents=True, exist_ok=True)

path_operator = f"./material/degradation_operators/{method}.pt"
degradation_operator = torch.load(path_operator, map_location=device)

transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
task_config = load_yaml(args["task_config"])
data_config = task_config['data']

dataset = get_dataset(**data_config, transforms=transform)
loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

# Grid
lamb = [0.5, 1, 1.5]
n_steps = [100, 300, 500]
sigmas = [0.01, 0.05, 0.1]
psnr_results = []
num_samples = 1

lpips = LPIPS()

# Fixed seed 
torch.manual_seed(0)

# Num samples


for lam in lamb:
    for n_step in n_steps:
        for sigma in sigmas:
            K = [n_step//10, n_step//5, n_step//2]
            eps_net = load_epsilon_net("celebahq", n_step, device).to(device)
            for k in K:
                # Iterate over dataset
                for diff_method in diff_methods:
            
                    psnr_list = []
                    lpips_list = []
                    time_list = []

                    for i, ref_img in enumerate(loader):
                        
                        if i == num_samples:
                            break
        
                        initial_noise = torch.randn((1, 3, 256, 256), device=device)
                        
                        y = degradation_operator.H(ref_img[None].to(device))
                        
                        y = y.squeeze(0)
                        y = y + sigma * torch.randn_like(y)

                        inverse_problem = (y, degradation_operator, sigma)

                        start_time = time.time()
                        
                        # Diffusion methods
                        if diff_method == "dps":
                            
                            reconstruction = dps(initial_noise, inverse_problem, eps_net)
                        if diff_method == "dps_dpms":
                        
                            reconstruction = dps_dpms(initial_noise, inverse_problem, eps_net)
                        
                        end_time = time.time()
                        exec_time = end_time - start_time
                        print('DMPS running time: {}'.format(exec_time))

                        psnr = peak_signal_noise_ratio(ref_img[0].cpu().numpy(), reconstruction[0].cpu().numpy())
                        lpips_score = lpips.score(reconstruction.clamp(-1, 1), ref_img)
                        print('PSNR: {}'.format(psnr), 'LPIPS: {}'.format(lpips_score))
                        
                        psnr_list.append(psnr)
                        lpips_list.append(lpips)
                        time_list.append(exec_time)

                        fig, axes = plt.subplots(1, 3)                    
                        
                        # Reshaping y
                        if method in ["outpainting_expand", "inpainting_middle"]:
                            y_reshaped =  -torch.ones(3 * 256 * 256, device=device)
                            y_reshaped[: y.shape[0]] = y
                            y_reshaped = degradation_operator.V(y_reshaped[None])
                            y_reshaped = y_reshaped.reshape(3, 256, 256)

                        images = (ref_img, y_reshaped, reconstruction[0])
                        titles = ("original", "degraded", "reconstruction")

                        # display figures
                        for ax, img, title in zip(axes, images,titles):
                            display_image(img, ax)
                            ax.set_title(title)
                    
                        fig.tight_layout()

                        if diff_method == "dps_dpms":
                            fig.suptitle(f"{method}, n_steps={n_step}, s={sigma}, k={k}, lpips={round(lpips_score.item(),2)}, time={round(end_time-start_time,2)}")
                            fig.savefig(f"{out_path}/{diff_method}/image_{i}_k_{k}_sigma_{sigma}_nstep_{n_step}.pdf", bbox_inches = 'tight')

                        elif diff_method == "dps":
                            fig.suptitle(f"{method}, n_steps={n_step}, s={sigma}, lpips={round(lpips_score.item(),2)}, time={round(end_time-start_time,2)}")
                            fig.savefig(f"{out_path}/{diff_method}/image_{i}_sigma_{sigma}_nstep_{n_step}.pdf", bbox_inches = 'tight')
                    
                    np.savetxt("{out_path}/{diff_method}/psnr_sigma_{sigma}_nstep_{n_step}.csv", np.asarray(psnr_list), delimiter=",")
                    np.savetxt("{out_path}/{diff_method}/lpips_sigma_{sigma}_nstep_{n_step}.csv", np.asarray(lpips_list), delimiter=",")
                    np.savetxt("{out_path}/{diff_method}/time_sigma_{sigma}_nstep_{n_step}.csv", np.asarray(time_list), delimiter=",")
