"""Deployment profiler / quantization honesty checks."""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from ecg_ssl_utils.deploy.profiler import (
    ProviderUnavailableError,
    assert_requested_provider,
    probe_usable_providers,
    profile_model,
)
from ecg_ssl_utils.deploy.quantization import (
    attention_sensitive_nodes,
    quant_pre_process_model,
    quantize_model,
)


class _SessionOptions:
    def __init__(self):
        self.graph_optimization_level = None
        self.intra_op_num_threads = 0


class _GraphOpt:
    ORT_ENABLE_ALL = 99


def _install_ort_stub(monkeypatch, session_factory, with_quant=False):
    """Minimal onnxruntime stand-in so tests run without the wheel."""
    ort = types.ModuleType("onnxruntime")
    ort.SessionOptions = _SessionOptions
    ort.GraphOptimizationLevel = _GraphOpt
    ort.__version__ = "stub"
    ort.InferenceSession = session_factory
    ort.get_available_providers = lambda: [
        "CPUExecutionProvider", "TensorrtExecutionProvider",
    ]
    ort.get_default_logger_severity = lambda: 2
    ort.set_default_logger_severity = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)

    if with_quant:
        q = types.ModuleType("onnxruntime.quantization")
        q.CalibrationDataReader = type("CalibrationDataReader", (), {})
        q.QuantFormat = types.SimpleNamespace(QDQ="QDQ")
        q.QuantType = types.SimpleNamespace(QInt8="QInt8")
        q.quantize_dynamic = lambda *a, **k: None
        q.quantize_static = lambda *a, **k: None
        si = types.ModuleType("onnxruntime.quantization.shape_inference")
        si.quant_pre_process = lambda **k: None
        q.shape_inference = si
        monkeypatch.setitem(sys.modules, "onnxruntime.quantization", q)
        monkeypatch.setitem(sys.modules, "onnxruntime.quantization.shape_inference", si)
        return ort, q, si
    return ort, None, None


def _install_psutil_stub(monkeypatch):
    ps = types.ModuleType("psutil")

    class Mem:
        rss = 10**7

    class Proc:
        def memory_info(self):
            return Mem()

    ps.Process = lambda *a, **k: Proc()
    monkeypatch.setitem(sys.modules, "psutil", ps)
    return ps


def test_assert_requested_provider_rejects_cpu_fallback():
    class Sess:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    with pytest.raises(ProviderUnavailableError, match="TensorrtExecutionProvider"):
        assert_requested_provider(Sess(), "TensorrtExecutionProvider")


def test_assert_requested_provider_accepts_match():
    class Sess:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    assert assert_requested_provider(Sess(), "CPUExecutionProvider") == [
        "CPUExecutionProvider"
    ]


def test_profile_rejects_provider_mismatch(monkeypatch, tmp_path):
    class Sess:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, *a, **k):
            return [np.zeros((1, 2), np.float32)]

    _install_ort_stub(monkeypatch, lambda *a, **k: Sess())
    onnx_path = tmp_path / "m.onnx"
    onnx_path.write_bytes(b"stub")
    x = np.zeros((1, 12, 8), np.float32)
    with pytest.raises(ProviderUnavailableError):
        profile_model(
            str(onnx_path), "m", "fp32", "TensorrtExecutionProvider",
            warmup=0, n_runs=1, input_tensor=x,
        )


def test_profile_records_actual_providers(monkeypatch, tmp_path):
    class Sess:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, *a, **k):
            return [np.zeros((1, 2), np.float32)]

    _install_ort_stub(monkeypatch, lambda *a, **k: Sess())
    _install_psutil_stub(monkeypatch)
    onnx_path = tmp_path / "m.onnx"
    onnx_path.write_bytes(b"stub")
    x = np.zeros((1, 12, 8), np.float32)
    prof = profile_model(
        str(onnx_path), "m", "fp32", "CPUExecutionProvider",
        warmup=1, n_runs=2, input_tensor=x,
    )
    assert prof.actual_providers == "CPUExecutionProvider"
    assert prof.provider == "CPUExecutionProvider"
    assert prof.gpu_model == ""
    assert "CPUExecutionProvider" in prof.measurement_notes


def test_probe_usable_providers_drops_unbound_extra_ep(monkeypatch):
    class CpuSess:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    class FallbackSess:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    def fake_session(path, providers=None, **k):
        if providers and providers[0] != "CPUExecutionProvider":
            return FallbackSess()
        return CpuSess()

    _install_ort_stub(monkeypatch, fake_session)
    monkeypatch.setattr(
        "ecg_ssl_utils.deploy.profiler._tiny_identity_onnx",
        lambda path: Path(path).write_bytes(b"x"),
    )
    usable = probe_usable_providers(
        ["CPUExecutionProvider", "TensorrtExecutionProvider"]
    )
    assert usable == ["CPUExecutionProvider"]


class _Node:
    def __init__(self, op_type, name, inputs, outputs):
        self.op_type = op_type
        self.name = name
        self.input = inputs
        self.output = outputs


def _fake_onnx_module(nodes):
    graph = types.SimpleNamespace(node=nodes)
    model = types.SimpleNamespace(graph=graph)
    return types.SimpleNamespace(load=lambda path: model)


def test_attention_nodes_exclude_qk_matmul_and_softmax(monkeypatch):
    nodes = [
        _Node("MatMul", "attn_qk_matmul", ["A", "B"], ["scores"]),
        _Node("Softmax", "attn_softmax", ["scores"], ["Y"]),
    ]
    monkeypatch.setitem(sys.modules, "onnx", _fake_onnx_module(nodes))
    names = attention_sensitive_nodes("unused.onnx")
    assert "attn_softmax" in names
    assert "attn_qk_matmul" in names


def test_conv_graph_has_no_attention_exclusions(monkeypatch):
    nodes = [_Node("Conv", "stem_conv", ["ecg", "W"], ["Y"])]
    monkeypatch.setitem(sys.modules, "onnx", _fake_onnx_module(nodes))
    assert attention_sensitive_nodes("unused.onnx") == []


def test_quantize_calls_preprocess(monkeypatch, tmp_path):
    src = tmp_path / "fp32.onnx"
    src.write_bytes(b"fp32-bytes")
    called = {}

    def fake_pre(input_model_path, output_model_path):
        called["pre"] = (input_model_path, output_model_path)
        Path(output_model_path).write_bytes(Path(input_model_path).read_bytes())

    def fake_dynamic(inp, out, **kwargs):
        called["dynamic_inp"] = inp
        Path(out).write_bytes(Path(inp).read_bytes())

    _, q, si = _install_ort_stub(monkeypatch, lambda *a, **k: None, with_quant=True)
    si.quant_pre_process = fake_pre
    q.quantize_dynamic = fake_dynamic

    out = tmp_path / "int8.onnx"
    quantize_model(str(src), str(out), mode="int8_dynamic")
    assert "pre" in called
    assert called["dynamic_inp"] != str(src)
    assert out.exists()


def test_quantize_static_passes_attention_exclusions(monkeypatch, tmp_path):
    src = tmp_path / "fp32.onnx"
    src.write_bytes(b"fp32")
    captured = {}

    def fake_pre(input_model_path, output_model_path):
        Path(output_model_path).write_bytes(Path(input_model_path).read_bytes())

    def fake_static(inp, out, **kwargs):
        captured.update(kwargs)
        Path(out).write_bytes(b"int8")

    _, q, si = _install_ort_stub(monkeypatch, lambda *a, **k: None, with_quant=True)
    si.quant_pre_process = fake_pre
    q.quantize_static = fake_static
    monkeypatch.setattr(
        "ecg_ssl_utils.deploy.quantization.attention_sensitive_nodes",
        lambda path: ["attn_softmax"],
    )
    out = tmp_path / "int8.onnx"
    quantize_model(str(src), str(out), mode="int8_static", calibration_data=[np.zeros((1, 12, 8), np.float32)])
    assert captured.get("nodes_to_exclude") == ["attn_softmax"]
    assert captured.get("per_channel") is True


def test_quantization_parity_reports_auroc_delta(monkeypatch, tmp_path):
    from ecg_ssl_utils.deploy.quantization import quantization_parity

    class Sess:
        def __init__(self, probs):
            self.probs = np.asarray(probs, dtype=np.float32)
            self.i = 0

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, *a, **k):
            row = self.probs[self.i:self.i + 1]
            self.i += 1
            return [row]

    fp_path, q_path = str(tmp_path / "fp.onnx"), str(tmp_path / "q.onnx")
    Path(fp_path).write_bytes(b"a")
    Path(q_path).write_bytes(b"b")
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    fp_p = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]], dtype=np.float32)
    q_p = fp_p.copy()

    def factory(path, providers=None, **k):
        return Sess(fp_p if path.endswith("fp.onnx") else q_p)

    _install_ort_stub(monkeypatch, factory)
    samples = [np.zeros((1, 12, 4), np.float32) for _ in range(4)]
    out = quantization_parity(fp_path, q_path, samples, y_true=y, min_positives=1)
    assert out["parity_cosine"] == pytest.approx(1.0, abs=1e-5)
    assert out["parity_delta_auroc"] == pytest.approx(0.0, abs=1e-6)
    assert out["parity_ece_fp32"] is not None


def test_quant_pre_process_model_writes_output(monkeypatch, tmp_path):
    src = tmp_path / "in.onnx"
    dst = tmp_path / "pre.onnx"
    src.write_bytes(b"abc")

    def fake_pre(input_model_path, output_model_path):
        Path(output_model_path).write_bytes(Path(input_model_path).read_bytes())

    _, _, si = _install_ort_stub(monkeypatch, lambda *a, **k: None, with_quant=True)
    si.quant_pre_process = fake_pre
    assert quant_pre_process_model(str(src), str(dst)) == str(dst)
    assert dst.read_bytes() == b"abc"
