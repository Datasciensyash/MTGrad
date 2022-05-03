import torch


class ResistivityTransform:
    @staticmethod
    def normalize(resistivity: torch.Tensor) -> torch.Tensor:
        return resistivity / 10000 - 1

    @staticmethod
    def denormalize(resistivity: torch.Tensor) -> torch.Tensor:
        return torch.clamp((resistivity + 1) * 10000, min=1.0e-4)


class FieldsTransform:
    @staticmethod
    def normalize_app_res(resistivity: torch.Tensor) -> torch.Tensor:
        return torch.log(resistivity)

    @staticmethod
    def normalize_imp_phs(impedance_phase: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.abs(impedance_phase))