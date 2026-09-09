"""ONNX quantization: dynamic and static INT8. Selective mode removed."""
import os

import numpy as np


def quantize_model(model_path, output_path, mode="int8_dynamic", calibration_data=None):
    """Quantize ONNX model. Returns output path and size."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )

    if mode == "int8_dynamic":
        quantize_dynamic(model_path, output_path, weight_type=QuantType.QInt8)

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
        quantize_static(
            model_path, output_path,
            calibration_data_reader=ECGCalibReader(calibration_data),
            quant_format=QuantFormat.QDQ,
        )
    else:
        raise ValueError(f"Unknown quantization mode '{mode}'. Use int8_dynamic or int8_static.")

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Quantized ({mode}) → {output_path} ({size_mb:.1f} MB)")
    return output_path, size_mb


def quantization_parity(fp32_path, quant_path, samples, provider="CPUExecutionProvider"):
    """Compare FP32 vs quantized embeddings/probs on the same inputs.

    Returns max absolute error and mean cosine similarity.
    """
    import onnxruntime as ort

    s_fp = ort.InferenceSession(fp32_path, providers=[provider])
    s_q = ort.InferenceSession(quant_path, providers=[provider])
    abs_errs, cosines = [], []
    for x in samples:
        a = s_fp.run(None, {"ecg": x})[0].reshape(-1)
        b = s_q.run(None, {"ecg": x})[0].reshape(-1)
        abs_errs.append(float(np.max(np.abs(a - b))))
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        cosines.append(float(np.dot(a, b) / denom))
    return {
        "parity_max_abs_err": float(np.max(abs_errs)),
        "parity_cosine": float(np.mean(cosines)),
    }
