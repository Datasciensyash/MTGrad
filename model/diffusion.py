from functools import partial

import torch
import torch.nn as nn

import numpy as np

from model.mtformer import MTFormer
from model.transform import ResistivityTransform


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusion(nn.Module):
    """Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.
    Input:
        x: tensor of shape (N, img_channels, *img_size)
        y: tensor of shape (N)
    Output:
        scalar loss tensor
    Args:
        model (nn.Module): model which estimates diffusion noise
        betas (np.ndarray): numpy array of diffusion betas
    """

    def __init__(
        self,
        model: MTFormer,
        betas: np.ndarray,
    ):
        super().__init__()

        self._rt = ResistivityTransform()

        self.model = model
        self.step = 0

        self.num_timesteps = len(betas)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    @torch.no_grad()
    def remove_noise(
        self,
        resistivity: torch.Tensor,
        t: torch.Tensor,
        estimated_noise: torch.Tensor
    ):
        return (
            resistivity
            - extract(self.remove_noise_coeff, t, resistivity.shape)
            * estimated_noise
        ) * extract(self.reciprocal_sqrt_alphas, t, resistivity.shape)

    @torch.no_grad()
    def sample(
        self,
        resistivity: torch.Tensor,
        apparent_resistivity: torch.Tensor,
        impedance_phase: torch.Tensor,
        periods: torch.Tensor,
        layer_powers: torch.Tensor,
        history: bool = False,
        n: int = 40
    ):
        batch_size, _, width = apparent_resistivity.shape
        device = apparent_resistivity.device
        resistivity = self._rt.forward_transform(resistivity)
        t = torch.Tensor([self.num_timesteps - n], device=device).long()
        x = torch.randn(1, layer_powers.shape[1], width, device=device)
        x = self.perturb_x(resistivity, t, x)

        history_x = [] if history else None

        for t in range(self.num_timesteps - n - 1, -1, -1):
            t_tensor = torch.Tensor([t], device=device).long()
            eps = self.model(
                resistivity=x,
                time=t_tensor,
                apparent_resistivity=apparent_resistivity,
                impedance_phase=impedance_phase,
                periods=periods,
                layer_powers=layer_powers
            )
            x = self.remove_noise(x, t_tensor, eps)

            if t > 0:
                x += extract(self.sigma, t_tensor, x.shape) * torch.randn_like(x)

            if history:
                history_x.append(x)

            # x = self._rt.forward_transform(x)

        return x, history_x

    def perturb_x(self, x, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

    def get_losses(
        self,
        resistivity: torch.Tensor,
        t: torch.Tensor,
        apparent_resistivity: torch.Tensor,
        impedance_phase: torch.Tensor,
        periods: torch.Tensor,
        layer_powers: torch.Tensor,
    ) -> torch.Tensor:
        noise = torch.randn_like(resistivity)
        resistivity = self._rt.forward_transform(resistivity)
        noisy_resistivity = self.perturb_x(resistivity, t, noise)
        # noisy_resistivity = self._rt.backward_transform(noisy_resistivity)

        estimated_noise = self.model(
            resistivity=noisy_resistivity,
            time=t,
            apparent_resistivity=apparent_resistivity,
            impedance_phase=impedance_phase,
            periods=periods,
            layer_powers=layer_powers
        )

        return (estimated_noise - resistivity).square().mean()

    def forward(
        self,
        resistivity: torch.Tensor,
        apparent_resistivity: torch.Tensor,
        impedance_phase: torch.Tensor,
        periods: torch.Tensor,
        layer_powers: torch.Tensor,
    ) -> torch.Tensor:
        t = torch.randint(0, self.num_timesteps, (resistivity.shape[0],), device=resistivity.device)
        return self.get_losses(
            resistivity, t, apparent_resistivity, impedance_phase, periods, layer_powers
        )


def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2

    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)

    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))

    return np.array(betas)


def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)
