import dataclasses
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional
import numpy as np
from tqdm import tqdm

from model.mtformer import MTFormer
from model.transform import ResistivityTransform


@dataclasses.dataclass
class MTConditioningData:
    impedance_phase: torch.Tensor
    apparent_resistivity: torch.Tensor
    periods: torch.Tensor
    layer_powers: torch.Tensor


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class MTDiffusion(nn.Module):

    def __init__(
        self,
        model: MTFormer,
        timesteps: int = 100,
    ):

        super(MTDiffusion, self).__init__()

        self._noise_estimator = model

        betas = cosine_beta_schedule(timesteps)

        alphas: torch.Tensor = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = functional.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        self.register_buffer('posterior_variance', posterior_variance)

        self._num_timesteps = timesteps

        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    @property
    def device(self) -> torch.device:
        return self.betas.device

    @staticmethod
    def cosine_beta_schedule(timesteps, s: float = 0.008) -> torch.Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def q_mean_variance(self, x_start: torch.Tensor, t: torch.Tensor):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def get_mean_and_variance(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            conditioning: MTConditioningData
    ):
        noise_predicted = self._noise_estimator(x, t, **conditioning.__dict__)
        x_recon = self.remove_noise(x, t=t, noise=noise_predicted)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def add_noise(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        x_weight = extract(self.sqrt_alphas_cumprod, t, x.shape)
        n_weight = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        return x_weight * x + n_weight * noise

    def remove_noise(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        x_weight = extract(self.sqrt_recip_alphas_cumprod, t, x.shape)
        n_weight = extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        return x_weight * x - n_weight * noise

    def forward(
        self,
        resistivity: torch.Tensor,
        conditioning: MTConditioningData
    ):
        batch_size, device = resistivity.shape[0], resistivity.device
        t = torch.randint(0, self._num_timesteps, (batch_size,), device=device).long()
        return self.get_loss(resistivity, t, conditioning)

    def get_loss(
            self,
            resistivity: torch.Tensor,
            t: torch.Tensor,
            conditioning: MTConditioningData
    ) -> torch.Tensor:

        # Normalize resistivity
        resistivity = ResistivityTransform.normalize(resistivity)

        # Sample noise
        noise = torch.randn_like(resistivity)

        # Add noise to resistivity
        x_noisy = self.add_noise(x=resistivity, t=t, noise=noise)
        # TODO: noise -> x_noisy! DEBUG!
        x_recon = self._noise_estimator(x_noisy, t, **conditioning.__dict__)

        # Compute loss
        loss = functional.mse_loss(noise, x_recon)

        return loss

    @torch.no_grad()
    def denoise(self, conditioning: MTConditioningData) -> torch.Tensor:
        resistivity_shape = conditioning.layer_powers.shape
        noise = torch.randn(resistivity_shape, device=self.device)
        for i in reversed(range(0, self._num_timesteps)):
            t = torch.full((resistivity_shape[0],), i, device=self.device, dtype=torch.long)
            noise = self.denoise_step(noise, t, conditioning)
            print(noise.abs().max())
        return noise

    @torch.no_grad()
    def denoise_step(
        self,
        noise: torch.Tensor,
        t: torch.Tensor,
        conditioning: MTConditioningData
    ):
        model_mean, _, model_log_variance = self.get_mean_and_variance(
            x=noise, t=t, conditioning=conditioning
        )
        noise = torch.randn(noise.shape, device=self.device)
        nonzero_mask = (1 - (t == 0).float()).reshape(noise.shape[0], *((1,) * (noise.ndim - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise


def cosine_beta_schedule(timesteps, s=0.007):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: MTFormer,
        timesteps: int = 100,
        loss_type='l2'
    ):
        super().__init__()

        self._denoise_model = model

        betas = cosine_beta_schedule(timesteps)

        alphas: torch.Tensor = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = functional.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool = False):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self._denoise_model(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True):

        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = torch.randn(x.shape, device=device)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = t if t is not None else self.num_timesteps - 1

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = noise if noise is not None else torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None):
        noise = noise if noise is not None else torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self._denoise_model(x_noisy, t)

        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = functional.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        return self.p_losses(x, t, *args, **kwargs)


class GaussianDiffusion2(nn.Module):
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
        resistivity = self._rt.normalize(resistivity)
        t = torch.Tensor([self.num_timesteps - n]).long().to(device)
        x = torch.randn(1, layer_powers.shape[1], width).to(device)
        x = self.perturb_x(resistivity, t, x)

        history_x = [] if history else None

        for t in range(self.num_timesteps - n - 1, -1, -1):
            t_tensor = torch.Tensor([t]).long().to(device)
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
                x += extract(self.sigma, t_tensor, x.shape) * torch.randn_like(x).to(device)

            if history:
                history_x.append(x)

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
        resistivity = self._rt.normalize(resistivity)
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

        return (estimated_noise - noise).square().mean()

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