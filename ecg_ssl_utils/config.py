"""
Central configuration for the ECG SSL robustness pipeline.
All hyperparameters are defined here as dataclasses for reproducibility.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ──────────────────────────────────────────────────────────────
# Data Configuration
# ──────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    """PTB-XL data loading and preprocessing settings."""
    sampling_rate: int = 500
    signal_length: int = 5000           # 10 s × 500 Hz
    n_leads: int = 12
    n_classes: int = 71                 # SCP-ECG diagnostic codes (fine-grained)
    n_superclasses: int = 5             # Diagnostic superclasses (primary eval target)
    superclass_names: List[str] = field(default_factory=lambda: [
        'NORM', 'MI', 'STTC', 'CD', 'HYP'
    ])
    lead_names: List[str] = field(default_factory=lambda: [
        'I', 'II', 'III', 'aVR', 'aVL', 'aVF',
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
    ])
    train_folds: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    val_folds: List[int] = field(default_factory=lambda: [9])
    test_folds: List[int] = field(default_factory=lambda: [10])
    bandpass_low: float = 0.5           # Hz
    bandpass_high: float = 45.0         # Hz
    filter_order: int = 4


# ──────────────────────────────────────────────────────────────
# Noise Configuration
# ──────────────────────────────────────────────────────────────
@dataclass
class NoiseConfig:
    """Noise injection pipeline settings."""
    snr_grid: List[float] = field(default_factory=lambda: [24, 18, 12, 6, 0, -6])
    # Seeds control noise injection determinism ONLY.
    # Each corruption condition is realized 3 times using these seeds.
    # SSL pretraining runs once per paradigm (not repeated).
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    noise_types_single: List[str] = field(default_factory=lambda: [
        'bw', 'ma', 'em', 'powerline', 'electrode_pop', 'inverter'
    ])
    mixed_noise_configs: List[Dict] = field(default_factory=lambda: [
        {'name': 'bw_ma',          'types': ['bw', 'ma'],              'weights': [0.5, 0.5]},
        {'name': 'em_powerline',   'types': ['em', 'powerline'],       'weights': [0.6, 0.4]},
        {'name': 'bw_ma_power',    'types': ['bw', 'ma', 'powerline'], 'weights': [0.33, 0.33, 0.34]},
    ])
    # Synthetic noise parameters
    powerline_freq: float = 50.0        # Hz (mains)
    powerline_harmonics: int = 3
    electrode_pop_rate: float = 0.5     # Hz (Poisson rate)
    electrode_pop_sigma: float = 0.005  # seconds
    inverter_duty_cycle: float = 0.1
    inverter_burst_freq: float = 200.0  # Hz (aliased from ~20 kHz)


# ──────────────────────────────────────────────────────────────
# Backbone Configuration
# ──────────────────────────────────────────────────────────────
@dataclass
class BackboneConfig:
    """ViT-Small 1D backbone architecture."""
    arch: str = 'vit_small_1d'
    patch_size: int = 50                # samples per patch
    embed_dim: int = 384
    depth: int = 12                     # transformer layers
    num_heads: int = 6
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1


# ──────────────────────────────────────────────────────────────
# SSL Training Configuration
# ──────────────────────────────────────────────────────────────
@dataclass
class SSLTrainingConfig:
    """Shared compute-budget settings for all SSL paradigms."""
    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    use_amp: bool = True
    checkpoint_every: int = 20
    num_workers: int = 4
    pin_memory: bool = True
    grad_accum_steps: int = 4            # effective batch = 256 × 4 = 1024
    grad_clip_norm: float = 1.0          # max gradient norm for clipping


@dataclass
class SimCLRConfig:
    """SimCLR-specific hyperparameters."""
    temperature: float = 0.5             # Chen et al. 2020 default
    proj_hidden_dim: int = 384
    proj_output_dim: int = 128


@dataclass
class CLOCSConfig:
    """CLOCS-specific hyperparameters."""
    temperature: float = 0.5             # τ=0.1 too aggressive with B=64
    proj_hidden_dim: int = 384
    proj_output_dim: int = 128
    lambda_temporal: float = 1.0
    lambda_spatial: float = 1.0
    lambda_patient: float = 0.5
    temporal_segment_len: int = 2500    # half-record (5 s)
    lead_group_a: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    lead_group_b: List[int] = field(default_factory=lambda: [6, 7, 8, 9, 10, 11])


@dataclass
class MAEConfig:
    """MAE-specific hyperparameters."""
    mask_ratio: float = 0.75
    decoder_embed_dim: int = 192
    decoder_depth: int = 4
    decoder_num_heads: int = 4
    norm_pix_loss: bool = True          # per-patch normalization before loss
    lr: float = 1.5e-3                  # MAE uses higher LR
    weight_decay: float = 0.05
    warmup_epochs: int = 20


@dataclass
class JEPAConfig:
    """JEPA-specific hyperparameters."""
    ema_momentum_start: float = 0.996
    ema_momentum_end: float = 1.0
    predictor_depth: int = 4
    predictor_embed_dim: int = 384
    predictor_num_heads: int = 6
    context_ratio: float = 0.50         # fraction of patches as context
    num_target_blocks: int = 4
    target_block_size_range: List[int] = field(default_factory=lambda: [10, 15])
    weight_decay: float = 0.05
    warmup_epochs: int = 20


@dataclass
class BYOLConfig:
    """BYOL-specific hyperparameters."""
    ema_momentum_start: float = 0.996
    ema_momentum_end: float = 1.0
    projector_hidden_dim: int = 4096
    projector_output_dim: int = 256
    predictor_hidden_dim: int = 4096
    predictor_output_dim: int = 256


@dataclass
class SwAVConfig:
    """SwAV-specific hyperparameters."""
    num_prototypes: int = 256
    temperature: float = 0.1
    sinkhorn_iterations: int = 3
    sinkhorn_epsilon: float = 0.05
    proj_hidden_dim: int = 384
    proj_output_dim: int = 128


# ──────────────────────────────────────────────────────────────
# Linear Probe Configuration
# ──────────────────────────────────────────────────────────────
@dataclass
class ProbeConfig:
    """Linear probe training settings."""
    epochs: int = 50
    batch_size: int = 512
    lr: float = 1e-2
    weight_decay: float = 0.0
    patience: int = 10
    n_calibration_bins: int = 15


# ──────────────────────────────────────────────────────────────
# Evaluation Configuration
# ──────────────────────────────────────────────────────────────
@dataclass
class EvalConfig:
    """Metric computation settings."""
    ece_bins: int = 15
    bootstrap_n: int = 1000
    bootstrap_seed: int = 42
    min_class_positives: int = 5        # skip classes with fewer positives
    age_bins: List[int] = field(default_factory=lambda: [0, 40, 60, 80, 120])
    age_bin_labels: List[str] = field(default_factory=lambda: [
        '0-40', '40-60', '60-80', '80+'
    ])
    delong_alpha: float = 0.05


# ──────────────────────────────────────────────────────────────
# Deployment Configuration
# ──────────────────────────────────────────────────────────────
@dataclass
class DeployConfig:
    """ONNX export and profiling settings."""
    quantization_modes: List[str] = field(default_factory=lambda: [
        'fp32', 'int8_dynamic', 'int8_static', 'selective'
    ])
    providers: List[str] = field(default_factory=lambda: [
        'CPUExecutionProvider', 'CUDAExecutionProvider'
    ])
    warmup_runs: int = 50
    benchmark_runs: int = 1000
    calibration_samples: int = 200
    opset_version: int = 17
    estimated_inference_power_w: float = 30.0   # T4 est. inference power


# ──────────────────────────────────────────────────────────────
# DSS Configuration
# ──────────────────────────────────────────────────────────────
@dataclass
class DSSConfig:
    """Robustness Score settings (formerly Deployment Safety Score)."""
    weights: Dict[str, float] = field(default_factory=lambda: {
        'cka': 0.25, 'erank': 0.25, 'ece': 0.25, 'auroc_decay': 0.25
    })
    ece_gate: float = 0.15              # non-compensatory safety gate
    auroc_gate: float = 0.70
    normalization: str = 'reference_anchored_minmax'
    sobol_samples: int = 1024
    bootstrap_n: int = 1000
    epsilon: float = 1e-8


# ──────────────────────────────────────────────────────────────
# Master Config
# ──────────────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    """Top-level configuration aggregating all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    ssl_training: SSLTrainingConfig = field(default_factory=SSLTrainingConfig)
    simclr: SimCLRConfig = field(default_factory=SimCLRConfig)
    clocs: CLOCSConfig = field(default_factory=CLOCSConfig)
    mae: MAEConfig = field(default_factory=MAEConfig)
    jepa: JEPAConfig = field(default_factory=JEPAConfig)
    byol: BYOLConfig = field(default_factory=BYOLConfig)
    swav: SwAVConfig = field(default_factory=SwAVConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)
    dss: DSSConfig = field(default_factory=DSSConfig)


def get_config() -> PipelineConfig:
    """Return the default pipeline configuration."""
    return PipelineConfig()
