import torch


class ResistivityTransform:

    def __init__(self, norm_b: float = 3, norm_w: float = 0.8):
        self._norm_b = norm_b
        self._norm_w = norm_w

    def forward_transform(self, resistivity: torch.Tensor) -> torch.Tensor:
        resistivity = torch.clamp(resistivity, min=1)
        return (torch.log10(resistivity) - self._norm_b) * self._norm_w

    def backward_transform(self, resistivity: torch.Tensor) -> torch.Tensor:
        # resistivity = torch.clamp(resistivity, min=1)
        return 10 ** ((resistivity / self._norm_w) + self._norm_b)