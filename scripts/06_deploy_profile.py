#!/usr/bin/env python3
"""
Script 06 — Edge Deployment Profiling
======================================
Export models to ONNX and profile FP32 and INT8 performance.

Kaggle Inputs: ssl-* models
Kaggle Output: /kaggle/working/deployment-profiles/
"""
import os, sys, glob, pandas as pd
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
    
    results = []
    
    for m_name, m_dir in unique_models.items():
        print(f"\nProfiling {m_name}")
        encoder = load_encoder(m_dir, cfg)
        
        # 1. Export FP32 ONNX
        fp32_path = os.path.join(OUTPUT_DIR, f"{m_name}_fp32.onnx")
        export_to_onnx(encoder, fp32_path, opset=cfg.deploy.opset_version)
        
        # 2. Quantize Dynamic INT8
        int8_path = os.path.join(OUTPUT_DIR, f"{m_name}_int8_dynamic.onnx")
        quantize_model(fp32_path, int8_path, mode='int8_dynamic')
        
        # 3. Profile
        for provider in cfg.deploy.providers:
            try:
                prof_fp32 = profile_model(fp32_path, m_name, 'fp32', provider,
                                          warmup=cfg.deploy.warmup_runs, n_runs=cfg.deploy.benchmark_runs,
                                          power_w=cfg.deploy.estimated_inference_power_w)
                results.append(asdict(prof_fp32))
                
                prof_int8 = profile_model(int8_path, m_name, 'int8_dynamic', provider,
                                          warmup=cfg.deploy.warmup_runs, n_runs=cfg.deploy.benchmark_runs,
                                          power_w=cfg.deploy.estimated_inference_power_w)
                results.append(asdict(prof_int8))
            except Exception as e:
                print(f"Profiling failed for {m_name} on {provider}: {e}")
                
    df = pd.DataFrame(results)
    df.to_parquet(os.path.join(OUTPUT_DIR, 'deployment_profiles.parquet'), index=False)
    print("\n✓ Deployment profiling complete!")

if __name__ == '__main__': main()
