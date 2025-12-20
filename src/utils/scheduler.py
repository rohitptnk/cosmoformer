from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def build_scheduler(optimizer, scheduler_cfg, total_steps):
    name = scheduler_cfg["name"]

    if name == "cosine_with_warmup":
        warmup_frac = scheduler_cfg["warmup_frac"]
        min_lr = scheduler_cfg["min_lr"]

        warmup_steps = max(1, int(warmup_frac * total_steps))

        warmup = LinearLR(
            optimizer,
            start_factor=0.0,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=min_lr,
        )

        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

    else:
        raise ValueError(f"Unsupported scheduler: {name}")
