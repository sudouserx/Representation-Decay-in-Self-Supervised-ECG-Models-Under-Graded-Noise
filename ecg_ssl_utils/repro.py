"""Reproducibility helpers: global seeds and deterministic DataLoaders."""

import os
import random
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset


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


def accumulation_state(
    step_idx: int, total_steps: int, grad_accum_steps: int,
) -> Tuple[int, bool]:
    """Return the current accumulation-window size and whether to step.

    The final window may be shorter than ``grad_accum_steps``. Scaling every
    loss by its actual window size and stepping on the final batch prevents
    silently discarding the tail of an epoch.
    """
    if total_steps <= 0 or not 0 <= step_idx < total_steps:
        raise ValueError("step_idx must index a non-empty epoch")
    accum = max(1, int(grad_accum_steps))
    window_start = (step_idx // accum) * accum
    window_size = min(accum, total_steps - window_start)
    should_step = (step_idx - window_start + 1) == window_size
    return window_size, should_step


def capture_rng_state() -> Dict[str, object]:
    """Capture Python, NumPy, and PyTorch RNG state for exact resume."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Dict[str, object]]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint_runtime_state(scheduler=None, scaler=None, loader=None) -> Dict[str, object]:
    """Serializable scheduler, AMP, and RNG state shared by training scripts."""
    state = {
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng_state": capture_rng_state(),
    }
    generator = getattr(loader, "generator", None)
    if generator is not None:
        state["loader_generator"] = generator.get_state()
    batch_sampler = getattr(loader, "batch_sampler", None)
    if hasattr(batch_sampler, "epoch"):
        state["batch_sampler_epoch"] = batch_sampler.epoch
    return state


def restore_runtime_state(checkpoint, scheduler=None, scaler=None, loader=None) -> bool:
    """Restore runtime state; return False for legacy checkpoints."""
    if "rng_state" not in checkpoint:
        return False
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    generator = getattr(loader, "generator", None)
    if generator is not None and checkpoint.get("loader_generator") is not None:
        generator.set_state(checkpoint["loader_generator"])
    batch_sampler = getattr(loader, "batch_sampler", None)
    if hasattr(batch_sampler, "epoch") and "batch_sampler_epoch" in checkpoint:
        batch_sampler.epoch = checkpoint["batch_sampler_epoch"]
    restore_rng_state(checkpoint["rng_state"])
    return True


class PatientPairBatchSampler(BatchSampler):
    """Batch every ECG once while guaranteeing distinct-record patient pairs.

    Each full batch receives ``pairs_per_batch`` patients with at least two
    recordings. The remaining indices are shuffled and used exactly once.
    This makes a patient contrastive term intentional rather than an
    accidental consequence of random record batching.
    """

    def __init__(
        self,
        patient_ids: Sequence[int],
        batch_size: int,
        seed: int,
        pairs_per_batch: int = 2,
        drop_last: bool = True,
    ):
        self.patient_ids = np.asarray(patient_ids)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.pairs_per_batch = int(pairs_per_batch)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if self.batch_size < 2 * self.pairs_per_batch:
            raise ValueError("batch_size is too small for requested patient pairs")
        self._groups = {
            pid: np.flatnonzero(self.patient_ids == pid).tolist()
            for pid in np.unique(self.patient_ids)
        }
        self._eligible = [pid for pid, idx in self._groups.items() if len(idx) >= 2]
        if len(self._eligible) < self.pairs_per_batch * len(self):
            raise ValueError("not enough multi-record patients to populate every batch")

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.patient_ids) // self.batch_size
        return int(np.ceil(len(self.patient_ids) / self.batch_size))

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1
        n_batches = len(self)
        eligible = rng.permutation(self._eligible)
        reserved: List[List[int]] = [[] for _ in range(n_batches)]
        used = set()
        for b in range(n_batches):
            for pid in eligible[
                b * self.pairs_per_batch:(b + 1) * self.pairs_per_batch
            ]:
                pair = rng.choice(self._groups[pid], size=2, replace=False).tolist()
                reserved[b].extend(pair)
                used.update(pair)
        remaining = [i for i in rng.permutation(len(self.patient_ids)) if i not in used]
        cursor = 0
        for batch in reserved:
            need = self.batch_size - len(batch)
            batch.extend(remaining[cursor:cursor + need])
            cursor += need
            if len(batch) == self.batch_size or not self.drop_last:
                rng.shuffle(batch)
                yield batch
