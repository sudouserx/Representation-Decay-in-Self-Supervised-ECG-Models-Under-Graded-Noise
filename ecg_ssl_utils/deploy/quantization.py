"""ONNX quantization: dynamic and static INT8. Selective mode removed."""
import os
import tempfile

import numpy as np


ATTENTION_EXCLUDE_OPS = frozenset({
    "Softmax",
    "LayerNormalization",
    "SkipLayerNormalization",
    "InstanceNormalization",
})


def quant_pre_process_model(model_path, output_path=None):
    """ORT recommended graph cleanup / shape inference before quantization."""
    from onnxruntime.quantization.shape_inference import quant_pre_process

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
    quant_pre_process(input_model_path=model_path, output_model_path=output_path)
    return output_path


def attention_sensitive_nodes(model_path):
    """Node names that should stay FP32 (softmax / LayerNorm / QK MatMul)."""
    import onnx

    model = onnx.load(model_path)
    names = []
    softmax_inputs = set()
    for node in model.graph.node:
        if node.op_type in ATTENTION_EXCLUDE_OPS and node.name:
            names.append(node.name)
        if node.op_type == "Softmax":
            softmax_inputs.update(node.input)
    for node in model.graph.node:
        if node.op_type != "MatMul" or not node.name:
            continue
        feeds_softmax = any(o in softmax_inputs for o in node.output)
        nlow = node.name.lower()
        name_hint = any(k in nlow for k in ("attn", "attention", "softmax"))
        if feeds_softmax or name_hint:
            names.append(node.name)
    return sorted(set(names))


def quantize_model(model_path, output_path, mode="int8_dynamic", calibration_data=None):
    """Quantize ONNX model. Returns output path and size."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )

    preprocessed = quant_pre_process_model(model_path)
    try:
        if mode == "int8_dynamic":
            quantize_dynamic(preprocessed, output_path, weight_type=QuantType.QInt8)

        elif mode == "int8_static":
            class ECGCalibReader(CalibrationDataReader):
                def __init__(self, data):
                    self.data = iter([{"ecg": d} for d in data])

                def get_next(self):
                    return next(self.data, None)

            if calibration_data is None:
                raise ValueError(
                    "int8_static requires calibration_data from train/val signals "
                    "(never test, never random placeholders)."
                )
            exclude = attention_sensitive_nodes(preprocessed)
            quantize_static(
                preprocessed, output_path,
                calibration_data_reader=ECGCalibReader(calibration_data),
                quant_format=QuantFormat.QDQ,
                per_channel=True,
                nodes_to_exclude=exclude or None,
            )
        else:
            raise ValueError(f"Unknown quantization mode '{mode}'. Use int8_dynamic or int8_static.")
    finally:
        if preprocessed != model_path:
            try:
                os.remove(preprocessed)
            except OSError:
                pass

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Quantized ({mode}) → {output_path} ({size_mb:.1f} MB)")
    return output_path, size_mb


def _stack_outputs(session, samples):
    rows = []
    for x in samples:
        rows.append(session.run(None, {"ecg": x})[0].reshape(-1))
    return np.stack(rows, axis=0)


def quantization_parity(
    fp32_path,
    quant_path,
    samples,
    provider="CPUExecutionProvider",
    y_true=None,
    min_positives=5,
):
    """Compare FP32 vs quantized probabilities on the same inputs.

    Returns max absolute error, mean cosine similarity, and (if ``y_true`` is
    given) macro-AUROC / ECE on the same val slice.
    """
    import onnxruntime as ort

    from ecg_ssl_utils.eval.auroc import macro_auroc
    from ecg_ssl_utils.eval.ece import expected_calibration_error

    s_fp = ort.InferenceSession(fp32_path, providers=[provider])
    s_q = ort.InferenceSession(quant_path, providers=[provider])
    from .profiler import assert_requested_provider
    assert_requested_provider(s_fp, provider)
    assert_requested_provider(s_q, provider)

    a = _stack_outputs(s_fp, samples)
    b = _stack_outputs(s_q, samples)
    abs_errs = np.max(np.abs(a - b), axis=1)
    denom = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + 1e-12
    cosines = np.sum(a * b, axis=1) / denom
    out = {
        "parity_max_abs_err": float(np.max(abs_errs)),
        "parity_cosine": float(np.mean(cosines)),
        "parity_auroc_fp32": None,
        "parity_auroc_quant": None,
        "parity_delta_auroc": None,
        "parity_ece_fp32": None,
        "parity_ece_quant": None,
        "parity_delta_ece": None,
    }
    if y_true is not None:
        y = np.asarray(y_true)
        if y.shape[0] != a.shape[0]:
            raise ValueError(
                f"y_true rows ({y.shape[0]}) must match parity samples ({a.shape[0]})"
            )
        auc_fp = macro_auroc(y, a, min_positives=min_positives)
        auc_q = macro_auroc(y, b, min_positives=min_positives)
        ece_fp = expected_calibration_error(y, a)
        ece_q = expected_calibration_error(y, b)
        out["parity_auroc_fp32"] = float(auc_fp)
        out["parity_auroc_quant"] = float(auc_q)
        out["parity_delta_auroc"] = float(auc_fp - auc_q)
        out["parity_ece_fp32"] = float(ece_fp)
        out["parity_ece_quant"] = float(ece_q)
        out["parity_delta_ece"] = float(ece_q - ece_fp)
    return out
