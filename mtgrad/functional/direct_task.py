import torch
import pymt.direct_task as direct_tasks


@torch.no_grad()
def direct_task(
        periods: torch.Tensor,
        layer_resistivity: torch.Tensor,
        layer_power: torch.Tensor
):
    device, ndim = periods.device, layer_resistivity.ndim
    periods = periods.detach().cpu().numpy()
    layer_resistivity = layer_resistivity.detach().cpu().numpy()
    layer_power = layer_power.detach().cpu().numpy()
    rho_t, phi_t = getattr(direct_tasks, f'direct_task_{ndim}d')(periods, layer_resistivity, layer_power)
    return torch.Tensor(rho_t).to(device), torch.Tensor(phi_t).to(device)
