"""Server-proxy ONNX Runtime latency / RSS / throughput profiler."""
import gc
import os
import platform
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


class ProviderUnavailableError(RuntimeError):
    """Requested ORT EP is not the primary provider on the session."""


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
    actual_providers: str = ""
    parity_cosine: Optional[float] = None
    parity_max_abs_err: Optional[float] = None
    parity_failed: Optional[bool] = None


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


def session_primary_provider(sess) -> str:
    providers = list(sess.get_providers() or [])
    if not providers:
        raise ProviderUnavailableError("ORT session has no execution providers")
    return providers[0]


def assert_requested_provider(sess, requested: str) -> List[str]:
    """Reject sessions whose primary EP is not the one we asked to profile."""
    active = list(sess.get_providers() or [])
    primary = session_primary_provider(sess)
    if primary != requested:
        raise ProviderUnavailableError(
            f"Requested {requested} but session primary EP is {primary} "
            f"(active={active}). Refusing to record mislabeled timings."
        )
    return active


def _tiny_identity_onnx(path: str) -> None:
    import onnx
    from onnx import TensorProto, helper

    inp = helper.make_tensor_value_info("ecg", TensorProto.FLOAT, [1, 1])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1])
    node = helper.make_node("Identity", ["ecg"], ["out"])
    graph = helper.make_graph([node], "ep_probe", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)


def probe_usable_providers(requested: Sequence[str]) -> List[str]:
    """Bind a tiny session per requested EP; keep only those that actually run.

    Inference profiling is CPU-only. This still fails loud if CPU cannot load.
    """
    import onnxruntime as ort

    usable: List[str] = []
    listed = set(ort.get_available_providers())
    prev_severity = None
    try:
        prev_severity = ort.get_default_logger_severity()
        ort.set_default_logger_severity(3)
    except Exception:
        pass

    fd, probe_path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    try:
        _tiny_identity_onnx(probe_path)
        for provider in requested:
            if provider not in listed:
                print(f"  Skipping {provider}: not in ORT get_available_providers()")
                continue
            try:
                sess = ort.InferenceSession(probe_path, providers=[provider])
                assert_requested_provider(sess, provider)
                usable.append(provider)
                del sess
            except Exception as e:
                print(
                    f"  Skipping {provider}: EP did not bind as primary "
                    f"({type(e).__name__}: {e})"
                )
    finally:
        try:
            os.remove(probe_path)
        except OSError:
            pass
        if prev_severity is not None:
            try:
                ort.set_default_logger_severity(prev_severity)
            except Exception:
                pass
    if not usable:
        raise RuntimeError(
            "No usable ONNX Runtime execution providers among "
            f"{list(requested)}. CPUExecutionProvider should always work."
        )
    return usable


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
    active = assert_requested_provider(sess, provider)

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
            "CPU server-proxy profiling. memory_mb is process RSS delta (psutil), "
            "not ORT native allocator peak. Energy is not measured. "
            "GPU ORT is out of scope. "
            f"ORT session providers={active}."
        ),
        cpu_model=_cpu_brand(),
        gpu_model="",
        runtime_version=ort.__version__,
        os_platform=platform.platform(),
        intra_op_num_threads=int(opts.intra_op_num_threads or 0),
        actual_providers=",".join(active),
    )
