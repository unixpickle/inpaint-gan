"""
Mostly borrowed from https://github.com/unixpickle/ddim/blob/27950a639afbbe3de8f8f72d8605f14f91a5742a/ddim/ddim.py.
"""

import numpy as np
import torch
from tqdm.auto import tqdm


def create_alpha_schedule(num_steps=100, beta_0=0.0001, beta_T=0.02):
    betas = np.linspace(beta_0, beta_T, num_steps)
    result = [1.0]
    alpha = 1.0
    for beta in betas:
        alpha *= 1 - beta
        result.append(alpha)
    return np.array(result, dtype=np.float64)


class Diffusion:
    """
    A PyTorch implementation of conditional DDPM sampling in the special case
    where the distribution being modeled is a unit Gaussian.
    """

    def __init__(self, alphas):
        self.alphas = alphas

    @property
    def num_steps(self):
        return len(self.alphas) - 1

    def sample_q(self, x_0, ts, epsilon=None):
        """
        Sample from q(x_t | x_0) for a batch of x_0.
        """
        if epsilon is None:
            epsilon = torch.randn_like(x_0)
        alphas = self.alphas_for_ts(ts, x_0.shape).to(x_0)
        return alphas.sqrt() * x_0 + (1 - alphas).sqrt() * epsilon

    def sample_xstart(self, x_t, ts, epsilon=None):
        # Equivalent when x_0 is Gaussian.
        return self.sample_q(x_t, ts, epsilon=epsilon)

    def ddpm_previous(self, x_t, ts, epsilon=None, cond_prediction=None):
        if epsilon is None:
            epsilon = torch.randn_like(x_t)
        alphas_t = self.alphas_for_ts(ts, x_t.shape)
        alphas_prev = self.alphas_for_ts(ts - 1, x_t.shape)
        alphas = alphas_t / alphas_prev
        betas = 1 - alphas
        prev_mean = (1 / alphas.sqrt()) * x_t
        if cond_prediction is not None:
            prev_mean += betas * cond_prediction
        return prev_mean + betas.sqrt() * epsilon

    def ddpm_sample_energy(self, x_T, cond_fn):
        """
        Create a sample using an energy function cond_fn as a conditioning
        signal, to compute p(x)*p(y|x), where cond_fn is log(p(y|x)).
        """
        x_t = x_T
        for t in tqdm(range(1, self.num_steps + 1)[::-1]):
            ts = torch.tensor([t] * x_T.shape[0]).long()
            alphas = self.alphas_for_ts(ts, shape=x_T.shape).to(x_T)

            with torch.enable_grad():
                x_t_grad = x_t.detach().requires_grad_(True)
                # Using one x_start sample gives us an unbiased estimate
                # of the energy function.
                x_start_sample = alphas.sqrt() * x_t_grad + (
                    1 - alphas
                ).sqrt() * torch.randn_like(x_t_grad)
                energy = cond_fn(x_start_sample)
                grad = torch.autograd.grad(energy, x_t_grad)[0]

            with torch.no_grad():
                x_t = self.ddpm_previous(x_t, ts, cond_prediction=grad)
        return x_t

    def ddpm_sample_inpaint(self, x_T, decoder_fn, target, mask):
        def cond_fn(z):
            image = decoder_fn(z)
            return -((image - target) ** 2 * mask).sum()

        return self.ddpm_sample_energy(x_T, cond_fn)

    def alphas_for_ts(self, ts, shape=None):
        alphas = torch.from_numpy(self.alphas).to(ts.device)[ts]
        if shape is None:
            return alphas
        while len(alphas.shape) < len(shape):
            alphas = alphas[..., None]
        return alphas.float()
