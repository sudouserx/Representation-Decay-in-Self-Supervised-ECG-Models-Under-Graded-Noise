"""ONNX export for ViT-Small 1D encoder."""
import torch, os


def export_to_onnx(encoder, output_path, signal_length=5000, n_leads=12, opset=17):
    """Export PyTorch encoder to ONNX format."""
    encoder.eval()
    dummy = torch.randn(1, n_leads, signal_length)
    if next(encoder.parameters()).is_cuda:
        dummy = dummy.cuda()
    torch.onnx.export(
        encoder, dummy, output_path,
        input_names=['ecg'], output_names=['embedding'],
        dynamic_axes={'ecg': {0: 'batch'}, 'embedding': {0: 'batch'}},
        opset_version=opset, do_constant_folding=True,
    )
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Exported to {output_path} ({size_mb:.1f} MB)")
    return output_path, size_mb
