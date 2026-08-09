"""ONNX quantization: dynamic, static, and selective INT8."""
import os, numpy as np


def quantize_model(model_path, output_path, mode='int8_dynamic',
                   calibration_data=None):
    """Quantize ONNX model. Returns output path and size."""
    from onnxruntime.quantization import quantize_dynamic, quantize_static
    from onnxruntime.quantization import QuantType, QuantFormat, CalibrationDataReader

    if mode == 'int8_dynamic':
        quantize_dynamic(model_path, output_path, weight_type=QuantType.QInt8)

    elif mode == 'int8_static':
        class ECGCalibReader(CalibrationDataReader):
            def __init__(self, data):
                self.data = iter([{'ecg': d} for d in data])
            def get_next(self):
                return next(self.data, None)

        if calibration_data is None:
            calibration_data = [np.random.randn(1,12,5000).astype(np.float32)
                                for _ in range(50)]
        quantize_static(model_path, output_path,
                        calibration_data_reader=ECGCalibReader(calibration_data),
                        quant_format=QuantFormat.QDQ)

    elif mode == 'selective':
        # Keep attention layers FP32, quantize FFN to INT8
        from onnxruntime.quantization import quantize_dynamic
        quantize_dynamic(model_path, output_path, weight_type=QuantType.QInt8,
                         nodes_to_exclude=[])  # placeholder: refine with op analysis

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Quantized ({mode}) → {output_path} ({size_mb:.1f} MB)")
    return output_path, size_mb
