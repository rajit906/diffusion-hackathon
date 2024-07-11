import torch
import numpy as np
from torch import vmap
from torch.func import jacrev


class EpsilonNet(torch.nn.Module):
    def __init__(self, net, alphas_cumprod, timesteps):
        super().__init__()
        self.net = net
        self.alphas_cumprod = alphas_cumprod
        self.timesteps = timesteps

    def forward(self, x, t):
        return self.net(x, torch.tensor(t))

    def predict_x0(self, x, t):
        alpha_cum_t = (
            self.alphas_cumprod[t] / self.alphas_cumprod[self.timesteps[0].int()]
        )
        return (x - (1 - alpha_cum_t) ** 0.5 * self.forward(x, t)) / (alpha_cum_t**0.5)
    
    def approximate_grad_log_likelihood(self, x_t, t, H_funcs, y, noise_std):
        #U, S, V, Ut, Vt = H_func.U, H_func.singulars, H_func.V, H_func.Ut, H_func.Vt
        #alpha_cum_t = self.alphas_cumprod[t]
        #return V(torch.mul(S, torch.mul(torch.reciprocal(std**2 * torch.ones_like(x) + (1 - alpha_cum_t)/(alpha_cum_t) * torch.square(S))(Ut(y) - torch.mul(S, Vt(x)) / torch.sqrt(alpha_cum_t))))) / torch.sqrt(alpha_cum_t)
        singulars = H_funcs.singulars()
        S = singulars*singulars.to(x_t.device)
        alpha_bar = self.alphas_cumprod[t] #TODO: Verify
        alpha_t = self.alphas_cumprod[t] / self.alphas_cumprod[t-1]#self.timesteps[0]] #TODO: Verify
        scale_S = (1-alpha_bar)/(alpha_bar)
        S_vector = (1/(S*scale_S +noise_std**2)).to(x_t.device).reshape(-1,1)
        Temp_value = H_funcs.Ut(y - H_funcs.H(x_t)/torch.sqrt(alpha_bar)).t()
        grad_value = H_funcs.Ht(H_funcs.U((S_vector*Temp_value).t()))
        grad_value = grad_value.reshape(x_t.shape)/torch.sqrt(alpha_bar)
        return grad_value, alpha_t

    
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, H_funcs, noise_std, alpha_t, alpha_bar, pseudonoise_scale,  **kwargs):
        singulars = H_funcs.singulars()
        S = singulars*singulars.to(x_t.device)
        alpha_bar = np.clip(alpha_bar, 1e-16, 1-1e-16)
        alpha_t = np.clip(alpha_t, 1e-16, 1-1e-16)
        scale_S = (1-alpha_bar)/(alpha_bar)

        S_vector = (1/(S*scale_S +noise_std**2)).to(x_t.device).reshape(-1,1)
        Temp_value = H_funcs.Ut(measurement - H_funcs.H(x_t)/np.sqrt(alpha_bar)).t()
        grad_value = H_funcs.Ht(H_funcs.U((S_vector*Temp_value).t()))
 
        grad_value = grad_value.reshape(x_t.shape)/np.sqrt(alpha_bar)
        x_t += self.scale*grad_value *(1-alpha_t)/np.sqrt(alpha_t)
        return x_t
    
    def score(self, x, t):
        alpha_cum_t = self.alphas_cumprod[t] / self.alphas_cumprod[self.timesteps[0]]
        return -self.forward(x, t) / (1 - alpha_cum_t) ** 0.5

    def value_and_grad_predx0(self, x, t):
        x = x.requires_grad_()
        pred_x0 = self.predict_x0(x, t)
        grad_pred_x0 = torch.autograd.grad(pred_x0.sum(), x)[0]
        return pred_x0, grad_pred_x0

    def value_and_jac_predx0(self, x, t):
        def pred(x):
            return self.predict_x0(x, t)

        pred_x0 = self.predict_x0(x, t)
        return pred_x0, vmap(jacrev(pred))(x)


class EpsilonNetSVD(EpsilonNet):
    def __init__(self, net, alphas_cumprod, timesteps, H_func, device="cuda"):
        super().__init__(net, alphas_cumprod, timesteps)
        self.net = net
        self.alphas_cumprod = alphas_cumprod
        self.H_func = H_func
        self.timesteps = timesteps
        self.device = device

    def forward(self, x, t):
        shape = (x.shape[0], 3, int(np.sqrt((x.shape[-1] // 3))), -1)
        x = self.H_func.V(x.to(self.device)).reshape(shape)
        return self.H_func.Vt(self.net(x, t))


# -----------
# ----------- diffusion kernels
# -----------
def bridge_kernel_statistics(
    x_ell: torch.Tensor,
    x_s: torch.Tensor,
    epsilon_net: EpsilonNet,
    ell: int,
    t: int,
    s: int,
    eta: float,
):
    """s < t < ell"""
    alpha_cum_s_to_t = epsilon_net.alphas_cumprod[t] / epsilon_net.alphas_cumprod[s]
    alpha_cum_t_to_ell = epsilon_net.alphas_cumprod[ell] / epsilon_net.alphas_cumprod[t]
    alpha_cum_s_to_ell = epsilon_net.alphas_cumprod[ell] / epsilon_net.alphas_cumprod[s]
    std = (
        eta
        * ((1 - alpha_cum_t_to_ell) * (1 - alpha_cum_s_to_t) / (1 - alpha_cum_s_to_ell))
        ** 0.5
    )
    coeff_xell = ((1 - alpha_cum_s_to_t - std**2) / (1 - alpha_cum_s_to_ell)) ** 0.5
    coeff_xs = (alpha_cum_s_to_t**0.5) - coeff_xell * (alpha_cum_s_to_ell**0.5)
    return coeff_xell * x_ell + coeff_xs * x_s, std


def sample_bridge_kernel(
    x_ell: torch.Tensor,
    x_s: torch.Tensor,
    epsilon_net: EpsilonNet,
    ell: int,
    t: int,
    s: int,
    eta: float,
):
    mean, std = bridge_kernel_statistics(x_ell, x_s, epsilon_net, ell, t, s, eta)
    return mean + std * torch.randn_like(mean)


def ddim_statistics(
    x: torch.Tensor,
    epsilon_net: EpsilonNet,
    t: float,
    t_prev: float,
    eta: float,
    e_t: torch.Tensor = None,
):
    t_0 = epsilon_net.timesteps[0]
    if e_t is None:
        e_t = epsilon_net.predict_x0(x, t)
    return bridge_kernel_statistics(
        x_ell=x, x_s=e_t, epsilon_net=epsilon_net, ell=t, t=t_prev, s=t_0, eta=eta
    )


def ddim_step(
    x: torch.Tensor,
    epsilon_net: EpsilonNet,
    t: float,
    t_prev: float,
    eta: float,
    e_t: torch.Tensor = None,
):
    t_0 = epsilon_net.timesteps[0]
    if e_t is None:
        e_t = epsilon_net.predict_x0(x, t)
    return sample_bridge_kernel(
        x_ell=x, x_s=e_t, epsilon_net=epsilon_net, ell=t, t=t_prev, s=t_0, eta=eta
    )


def ddim(
    initial_noise_sample: torch.Tensor, epsilon_net: EpsilonNet, eta: float = 1.0
) -> torch.Tensor:
    """
    This function implements the (subsampled) generation from https://arxiv.org/pdf/2010.02502.pdf (eqs 9,10, 12)

    Parameters
    ----------
    initial_noise_sample :
        Initial "noise"
    timesteps :
        List containing the timesteps. Should start by 999 and end by 0

    score_model :
        The score model

    eta :
        the parameter eta from https://arxiv.org/pdf/2010.02502.pdf (eq 16)
    """
    sample = initial_noise_sample
    for i in range(len(epsilon_net.timesteps) - 1, 1, -1):
        sample = ddim_step(
            x=sample,
            epsilon_net=epsilon_net,
            t=epsilon_net.timesteps[i],
            t_prev=epsilon_net.timesteps[i - 1],
            eta=eta,
        )
    return epsilon_net.predict_x0(sample, epsilon_net.timesteps[1])
