import torch


class ResistivityTransform:
    @staticmethod
    def normalize(resistivity: torch.Tensor) -> torch.Tensor:
        return resistivity / 20000

    @staticmethod
    def denormalize(resistivity: torch.Tensor) -> torch.Tensor:
        return torch.clamp(resistivity * 20000, min=1.0e-4)


class FieldsTransform:
    @staticmethod
    def normalize_app_res(resistivity: torch.Tensor) -> torch.Tensor:
        return resistivity / 10000

    @staticmethod
    def normalize_imp_phs(impedance_phase: torch.Tensor) -> torch.Tensor:
        return torch.deg2rad(impedance_phase)
