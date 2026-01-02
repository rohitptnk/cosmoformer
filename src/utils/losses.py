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

    # weight for noise loss (prioritize Cl reconstruction or noise reconstruction)
    lambda_noise = loss_cfg.get("lambda_noise", 1.0)

    if loss_type == "mse":
        
        def loss_fn(
            clean_mean, clean_logvar, 
            noise_mean, noise_logvar,
            y_clean, y_noise,
        ):
            loss_clean = mse_loss(clean_mean, None, y_clean)
            loss_noise = mse_loss(noise_mean, None, y_noise)
            return loss_clean + lambda_noise * loss_noise
        
        return loss_fn
     
    elif loss_type == "heteroscedastic":
        hetero_cfg = loss_cfg.get("hetero", {})
        
        def loss_fn(
            clean_mean, clean_logvar,
            noise_mean, noise_logvar,
            y_clean, y_noise,
        ):
            loss_clean = heteroscedastic_loss(
                clean_mean, clean_logvar, y_clean, hetero_cfg,
            )
            loss_noise = heteroscedastic_loss(
                noise_mean, noise_logvar, y_noise, hetero_cfg,
            )
            return loss_clean + lambda_noise * loss_noise
        
        return loss_fn
    
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
