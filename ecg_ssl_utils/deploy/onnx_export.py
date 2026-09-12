"""ONNX export for encoder + linear probe.

Band-pass filtering and per-lead z-score normalization stay outside the
graph (they are applied in scripts 00/04 before inference).
"""
import inspect
import os

import torch
import torch.nn as nn


class ClassifierWrapper(nn.Module):
    """Frozen encoder CLS embedding → linear probe logits/probabilities."""

    def __init__(self, encoder, probe):
        super().__init__()
        self.encoder = encoder
        self.probe = probe

    def forward(self, x):
        h = self.encoder(x)
        return self.probe.predict_proba(h)


def export_to_onnx(module, output_path, signal_length=5000, n_leads=12, opset=17,
                   output_name="probabilities"):
    """Export a torch module (encoder or ClassifierWrapper) to ONNX."""
    module.eval()
    dummy = torch.randn(1, n_leads, signal_length)
    if next(module.parameters()).is_cuda:
        dummy = dummy.cuda()
    # Torch >= 2.8 defaults dynamo=True, which imports onnxscript. The
    # TorchScript exporter is still the one onnxruntime INT8 quantization
    # expects, so keep dynamo off when the kwarg exists.
    export_kwargs = dict(
        input_names=["ecg"],
        output_names=[output_name],
        dynamic_axes={"ecg": {0: "batch"}, output_name: {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(module, dummy, output_path, **export_kwargs)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Exported to {output_path} ({size_mb:.1f} MB)")
    return output_path, size_mb
