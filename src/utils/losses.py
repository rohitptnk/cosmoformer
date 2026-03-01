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

    # weights for different loss components
    lambda_fg1 = loss_cfg.get("lambda_fg1", 1.0)
    lambda_fg2 = loss_cfg.get("lambda_fg2", 1.0)

    if loss_type == "mse":
        
        def loss_fn(
            c_mean, c_logvar, 
            f1_mean, f1_logvar,
            f2_mean, f2_logvar,
            y_clean, y_fg1, y_fg2,
        ):
            loss_clean = mse_loss(c_mean, None, y_clean)
            loss_fg1 = mse_loss(f1_mean, None, y_fg1)
            loss_fg2 = mse_loss(f2_mean, None, y_fg2)
            return loss_clean + lambda_fg1 * loss_fg1 + lambda_fg2 * loss_fg2
        
        return loss_fn
     
    elif loss_type == "heteroscedastic":
        hetero_cfg = loss_cfg.get("hetero", {})
        
        def loss_fn(
            c_mean, c_logvar,
            f1_mean, f1_logvar,
            f2_mean, f2_logvar,
            y_clean, y_fg1, y_fg2,
        ):
            loss_clean = heteroscedastic_loss(
                c_mean, c_logvar, y_clean, hetero_cfg,
            )
            loss_fg1 = heteroscedastic_loss(
                f1_mean, f1_logvar, y_fg1, hetero_cfg,
            )
            loss_fg2 = heteroscedastic_loss(
                f2_mean, f2_logvar, y_fg2, hetero_cfg,
            )
            return loss_clean + lambda_fg1 * loss_fg1 + lambda_fg2 * loss_fg2
        
        return loss_fn
    
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
