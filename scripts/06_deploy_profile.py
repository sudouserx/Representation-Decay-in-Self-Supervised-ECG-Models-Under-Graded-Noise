#!/usr/bin/env python3
"""
Script 06 — Edge Deployment Profiling
======================================
Export models to ONNX and profile FP32 and INT8 performance.

Kaggle Inputs: ssl-* models
Kaggle Output: /kaggle/working/deployment-profiles/
"""
import os, sys, glob, pandas as pd, numpy as np
from dataclasses import asdict

UTILS_DIR = os.environ.get('UTILS_DIR', '/kaggle/input/ecg-ssl-utils')
OUTPUT_DIR = '/kaggle/working/deployment-profiles'
if UTILS_DIR not in sys.path: sys.path.insert(0, UTILS_DIR)

from ecg_ssl_utils.deploy.onnx_export import export_to_onnx
from ecg_ssl_utils.deploy.quantization import quantize_model
from ecg_ssl_utils.deploy.profiler import profile_model
from ecg_ssl_utils.config import get_config
import torch, json
from ecg_ssl_utils.models.vit_small_1d import ViTSmall1D
from ecg_ssl_utils.models.resnet18_1d import ResNet18_1D


def load_encoder(model_dir, cfg):
    with open(os.path.join(model_dir, 'config.json')) as f:
        m_cfg = json.load(f)
    if m_cfg['backbone'] == 'vit_small_1d':
        encoder = ViTSmall1D(patch_size=cfg.backbone.patch_size, embed_dim=cfg.backbone.embed_dim,
                             depth=cfg.backbone.depth, num_heads=cfg.backbone.num_heads)
    else:
        encoder = ResNet18_1D(in_channels=12, output_dim=cfg.backbone.embed_dim)
    
    encoder.load_state_dict(torch.load(os.path.join(model_dir, 'encoder.pt'), map_location='cpu'))
    return encoder

def main():
    cfg = get_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model_dirs = glob.glob('/kaggle/input/ssl-*') + glob.glob('/kaggle/working/ssl-*')
    unique_models = {os.path.basename(d): d for d in model_dirs if os.path.exists(os.path.join(d, 'encoder.pt'))}

    # Load calibration data for static quantization (clean test signals)
    clean_dir = os.environ.get('CLEAN_DIR', '/kaggle/input/ptbxl-clean-processed')
    calib_path = os.path.join(clean_dir, 'signals_test.npy')
    if os.path.exists(calib_path):
        calib_signals = np.load(calib_path)[:cfg.deploy.calibration_samples]
        # Static quant needs individual samples as list of (1, 12, 5000) arrays
        calib_data = [calib_signals[i:i+1].astype(np.float32) for i in range(len(calib_signals))]
        print(f"Loaded {len(calib_data)} calibration samples for static quantization")
    else:
        calib_data = None
        print("No calibration data found; static quantization will use random data")
    
    results = []
    
    for m_name, m_dir in unique_models.items():
        print(f"\nProfiling {m_name}")
        encoder = load_encoder(m_dir, cfg)
        
        # 1. Export FP32 ONNX
        fp32_path = os.path.join(OUTPUT_DIR, f"{m_name}_fp32.onnx")
        export_to_onnx(encoder, fp32_path, opset=cfg.deploy.opset_version)
        
        # 2. Quantize all modes from config
        quant_paths = {'fp32': fp32_path}
        for mode in cfg.deploy.quantization_modes:
            if mode == 'fp32':
                continue  # already have the FP32 model
            quant_path = os.path.join(OUTPUT_DIR, f"{m_name}_{mode}.onnx")
            try:
                quantize_model(fp32_path, quant_path, mode=mode,
                               calibration_data=calib_data)
                quant_paths[mode] = quant_path
            except Exception as e:
                print(f"  Quantization ({mode}) failed for {m_name}: {e}")
        
        # 3. Profile all available models × providers
        for precision, model_path in quant_paths.items():
            for provider in cfg.deploy.providers:
                try:
                    prof = profile_model(model_path, m_name, precision, provider,
                                          warmup=cfg.deploy.warmup_runs, n_runs=cfg.deploy.benchmark_runs,
                                          power_w=cfg.deploy.estimated_inference_power_w)
                    results.append(asdict(prof))
                except Exception as e:
                    print(f"  Profiling failed for {m_name}/{precision} on {provider}: {e}")
                
    df = pd.DataFrame(results)
    df.to_parquet(os.path.join(OUTPUT_DIR, 'deployment_profiles.parquet'), index=False)
    print("\n✓ Deployment profiling complete!")

if __name__ == '__main__': main()
