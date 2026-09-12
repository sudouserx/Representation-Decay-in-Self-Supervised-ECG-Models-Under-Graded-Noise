"""Deployment helpers. Imports are lazy so unit tests can load profiler/quant without torch."""

def __getattr__(name):
    if name in ("export_to_onnx", "ClassifierWrapper"):
        from .onnx_export import ClassifierWrapper, export_to_onnx
        return export_to_onnx if name == "export_to_onnx" else ClassifierWrapper
    if name in ("profile_model", "ProviderUnavailableError", "probe_usable_providers"):
        from . import profiler
        return getattr(profiler, name)
    if name in ("quantize_model", "attention_sensitive_nodes", "quant_pre_process_model"):
        from . import quantization
        return getattr(quantization, name)
    raise AttributeError(name)
