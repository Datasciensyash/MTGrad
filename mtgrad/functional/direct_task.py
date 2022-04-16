import pymt.direct_task as direct_tasks
import torch


@torch.no_grad()
def direct_task(
    periods: torch.Tensor, layer_resistivity: torch.Tensor, layer_power: torch.Tensor
):
    batch_size, period_num = periods.shape
    device, ndim = periods.device, layer_resistivity.ndim
    periods = periods.detach().cpu().numpy()
    layer_resistivity = layer_resistivity.detach().cpu().numpy()
    layer_power = layer_power.detach().cpu().numpy()

    rho_t = torch.empty((batch_size, period_num, layer_resistivity.shape[-1]))
    phi_t = torch.empty((batch_size, period_num, layer_resistivity.shape[-1]))

    print(periods.shape, layer_resistivity.shape, layer_power.shape)
    for i in range(batch_size):
        r, p = getattr(direct_tasks, f"direct_task_{ndim - 1}d")(
            periods[i], layer_resistivity[i].T, layer_power[i].T
        )
        phi_t[i, :] = torch.Tensor(p).T
        rho_t[i, :] = torch.Tensor(r).T
    return rho_t.to(device), phi_t.to(device)
