
from typing import Tuple
import torch
from sampling.epsilon_net import ddim_step, EpsilonNet


def dpms(
    initial_noise: torch.Tensor,
    inverse_problem: Tuple,
    epsilon_net: EpsilonNet,
    lam: float = 1.0,
    gamma: float = 1.0,
    eta: float = 1.0,
):
    """Use DPMS algorithm to solve an inverse problem.

    This is an implementation of [1].
    Use ``gamma`` to control the strength of the gradient perturbation.

    Note
    ----
    - Use ``initial_noise`` to set the number of samples ``(n_samples, *shape_of_data)``.

    References
    ----------
    .. Meng, Xiangming, and Yoshiyuki Kabashima. "Diffusion model based posterior sampling 
    for noisy linear inverse problems." arXiv preprint arXiv:2211.12343 (2022).

    """
    obs, H_func, std = inverse_problem
    A = H_func.H
    shape = (initial_noise.shape[0], *(1,) * len(initial_noise.shape[1:]))

    sample = initial_noise
    for i in range(len(epsilon_net.timesteps) - 1, 1, -1):
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]
        sample.requires_grad_()
        e_t = epsilon_net.predict_x0(sample, t)
        ### NEW
        grad_value, alpha_t = epsilon_net.approximate_grad_log_likelihood(x_t = sample, t=t_prev, H_funcs=H_func, y=obs, noise_std=std)
        ###

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