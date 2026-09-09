"""Server-proxy ONNX Runtime latency / RSS / throughput profiler."""
import gc
import os
import platform
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


@dataclass
class DeploymentProfile:
    model_id: str
    precision: str
    provider: str
    latency_p50: float
    latency_p95: float
    memory_mb: float
    throughput: float
    model_size_mb: float
    device_name: str
    batch_size: int
    input_shape: str
    warmup_runs: int
    benchmark_runs: int
    measurement_notes: str
    cpu_model: str = ""
    gpu_model: str = ""
    runtime_version: str = ""
    os_platform: str = ""
    intra_op_num_threads: int = 0
    parity_cosine: Optional[float] = None
    parity_max_abs_err: Optional[float] = None


def _cpu_brand() -> str:
    brand = platform.processor() or ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return brand


def profile_model(
    model_path,
    model_id,
    precision,
    provider="CPUExecutionProvider",
    warmup=50,
    n_runs=1000,
    input_tensor=None,
):
    """Profile ONNX model on a real ECG tensor (server-proxy, not edge)."""
    import onnxruntime as ort
    import psutil

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model_path, opts, providers=[provider])

    if input_tensor is None:
        raise ValueError("input_tensor must be a real ECG array of shape (1, 12, L)")
    dummy = np.asarray(input_tensor, dtype=np.float32)

    for _ in range(warmup):
        sess.run(None, {"ecg": dummy})

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {"ecg": dummy})
        latencies.append(time.perf_counter() - t0)
    latencies = np.array(latencies)

    proc = psutil.Process(os.getpid())
    gc.collect()
    rss0 = proc.memory_info().rss
    for _ in range(20):
        sess.run(None, {"ecg": dummy})
    rss1 = proc.memory_info().rss
    memory_mb = max(0.0, (rss1 - rss0) / 1e6)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    throughput = 1.0 / float(np.mean(latencies))
    size_mb = os.path.getsize(model_path) / 1e6

    gpu_name = ""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return DeploymentProfile(
        model_id=model_id,
        precision=precision,
        provider=provider,
        latency_p50=p50,
        latency_p95=p95,
        memory_mb=memory_mb,
        throughput=throughput,
        model_size_mb=size_mb,
        device_name=provider,
        batch_size=1,
        input_shape=str(dummy.shape),
        warmup_runs=warmup,
        benchmark_runs=n_runs,
        measurement_notes=(
            "Server-proxy profiling. memory_mb is process RSS delta (psutil), "
            "not ORT native allocator peak. Energy is not measured."
        ),
        cpu_model=_cpu_brand(),
        gpu_model=gpu_name,
        runtime_version=ort.__version__,
        os_platform=platform.platform(),
        intra_op_num_threads=int(opts.intra_op_num_threads or 0),
    )
