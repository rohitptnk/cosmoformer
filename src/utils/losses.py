import torch


def mse_loss(mean, logvar, target, cfg=None):
    # logvar ignored
    return torch.mean((mean - target) ** 2)


def heteroscedastic_loss(mean, logvar, target, hetero_cfg):
    if logvar is None:
        raise ValueError("Heteroscedastic loss requires logvar output")

    if hetero_cfg.get("clamp_logvar", False):
        logvar = torch.clamp(
            logvar,
            hetero_cfg.get("logvar_clamp_min", -20.0),
            hetero_cfg.get("logvar_clamp_max", 20.0),
        )

    return torch.mean(torch.exp(-logvar) * (target - mean) ** 2 + logvar)


def build_loss(cfg):
    loss_cfg = cfg["loss"]
    loss_type = loss_cfg["type"]

    if loss_type == "mse":
        return lambda mean, logvar, target: mse_loss(mean, logvar, target)

    elif loss_type == "heteroscedastic":
        hetero_cfg = loss_cfg.get("hetero", {})
        return lambda mean, logvar, target: heteroscedastic_loss(
            mean, logvar, target, hetero_cfg
        )

    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
