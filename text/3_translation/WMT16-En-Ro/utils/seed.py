"""Reproducible seeding for python, numpy and torch."""

from __future__ import annotations

import os
import random


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG this project touches.

    `deterministic=True` additionally forces cuDNN onto deterministic
    algorithms. That costs throughput and is off by default: it matters for
    reproducing one run exactly, not for comparing two configurations.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import numpy as np
    np.random.seed(seed)

    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: give each worker a distinct, derived seed."""
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
