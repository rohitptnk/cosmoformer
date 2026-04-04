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
    lambda_clean = loss_cfg.get("lambda_clean", 1.0)
    lambda_freq1 = loss_cfg.get("lambda_freq1", 1.0)
    lambda_freq2 = loss_cfg.get("lambda_freq2", 1.0)
    lambda_freq3 = loss_cfg.get("lambda_freq3", 1.0)
    lambda_freq4 = loss_cfg.get("lambda_freq4", 1.0)

    if loss_type == "mse":
        
        def loss_fn(
            c_mean, c_logvar, 
            f1_mean, f1_logvar,
            f2_mean, f2_logvar,
            f3_mean, f3_logvar,
            f4_mean, f4_logvar,
            y_clean, y_freq1, y_freq2, y_freq3, y_freq4,
        ):
            loss_clean = mse_loss(c_mean, None, y_clean)
            loss_freq1 = mse_loss(f1_mean, None, y_freq1)
            loss_freq2 = mse_loss(f2_mean, None, y_freq2)
            loss_freq3 = mse_loss(f3_mean, None, y_freq3)
            loss_freq4 = mse_loss(f4_mean, None, y_freq4)
            return lambda_clean * loss_clean + lambda_freq1 * loss_freq1 + lambda_freq2 * loss_freq2 + lambda_freq3 * loss_freq3 + lambda_freq4 * loss_freq4
        
        return loss_fn
     
    elif loss_type == "heteroscedastic":
        hetero_cfg = loss_cfg.get("hetero", {})
        
        def loss_fn(
            c_mean, c_logvar,
            f1_mean, f1_logvar,
            f2_mean, f2_logvar,
            f3_mean, f3_logvar,
            f4_mean, f4_logvar,
            y_clean, y_freq1, y_freq2, y_freq3, y_freq4,
        ):
            loss_clean = heteroscedastic_loss(
                c_mean, c_logvar, y_clean, hetero_cfg,
            )
            loss_freq1 = heteroscedastic_loss(
                f1_mean, f1_logvar, y_freq1, hetero_cfg,
            )
            loss_freq2 = heteroscedastic_loss(
                f2_mean, f2_logvar, y_freq2, hetero_cfg,
            )
            loss_freq3 = heteroscedastic_loss(
                f3_mean, f3_logvar, y_freq3, hetero_cfg,
            )
            loss_freq4 = heteroscedastic_loss(
                f4_mean, f4_logvar, y_freq4, hetero_cfg,
            )
            return lambda_clean * loss_clean + lambda_freq1 * loss_freq1 + lambda_freq2 * loss_freq2 + lambda_freq3 * loss_freq3 + lambda_freq4 * loss_freq4
        
        return loss_fn
    
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
