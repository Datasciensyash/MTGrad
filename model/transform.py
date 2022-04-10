import torch


class ResistivityTransform:

    def __init__(self, norm_b: float = 4, norm_w: float = 10000):
        self._norm_b = norm_b
        self._norm_w = norm_w

    def backward_transform(self, noise: torch.Tensor) -> torch.Tensor:
        # resistivity = torch.exp(torch.clamp(noise, min=0.01))
        return torch.clamp(noise * 20000, min=1)

    def forward_transform(self, resistivity: torch.Tensor) -> torch.Tensor:
        return resistivity / 20000