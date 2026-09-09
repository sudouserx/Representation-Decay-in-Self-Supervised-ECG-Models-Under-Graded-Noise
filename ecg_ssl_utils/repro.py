"""Reproducibility helpers: global seeds and deterministic DataLoaders."""

import os
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def parse_pretrain_seed(default: int = 42) -> int:
    """CLI ``--seed`` or env ``PRETRAIN_SEED``."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=int(os.environ.get("PRETRAIN_SEED", default)))
    args, _ = parser.parse_known_args()
    return int(args.seed)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) and tighten cuDNN flags.

    Residual nondeterminism from AMP / some cuDNN kernels may remain; document
    that limitation in the README.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_deterministic_loader(
    dataset: Dataset,
    batch_size: int,
    seed: int,
    workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    """DataLoader with a seeded generator and worker_init_fn."""
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        generator=g,
        worker_init_fn=_seed_worker if workers > 0 else None,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=False,
    )
