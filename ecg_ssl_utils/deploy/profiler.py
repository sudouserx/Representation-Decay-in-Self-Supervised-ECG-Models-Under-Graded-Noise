"""ONNX Runtime latency/memory/throughput profiler."""
import time, os, numpy as np
import tracemalloc
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class DeploymentProfile:
    model_id: str
    precision: str
    provider: str
    latency_p50: float
    latency_p95: float
    memory_mb: float
    estimated_energy_j: float
    throughput: float
    model_size_mb: float
    device_name: str
    batch_size: int
    input_shape: str
    warmup_runs: int
    benchmark_runs: int
    measurement_notes: str


def profile_model(model_path, model_id, precision, provider='CPUExecutionProvider',
                  warmup=50, n_runs=1000, power_w=30.0):
    """Profile ONNX model inference performance."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model_path, opts, providers=[provider])

    dummy = np.random.randn(1, 12, 5000).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        sess.run(None, {'ecg': dummy})

    # Latency
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {'ecg': dummy})
        latencies.append(time.perf_counter() - t0)
    latencies = np.array(latencies)

    # Memory
    tracemalloc.start()
    sess.run(None, {'ecg': dummy})
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    throughput = 1.0 / float(np.mean(latencies))
    energy = p50 * power_w
    size_mb = os.path.getsize(model_path) / 1e6

    return DeploymentProfile(
        model_id=model_id, precision=precision, provider=provider,
        latency_p50=p50, latency_p95=p95, memory_mb=peak/1e6,
        estimated_energy_j=energy, throughput=throughput,
        model_size_mb=size_mb, device_name=provider,
        batch_size=1, input_shape=str(dummy.shape),
        warmup_runs=warmup, benchmark_runs=n_runs,
        measurement_notes="Energy is software-estimated (latency * assumed power draw). Not measured."
    )
