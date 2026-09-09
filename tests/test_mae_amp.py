"""MAE decoder scatter_ must accept AMP dtypes (fp16/bf16 tokens, fp32 parameters)."""
import torch

from ecg_ssl_utils.models.vit_small_1d import ViTSmall1D
from ecg_ssl_utils.ssl.mae import MAEDecoder, MAEModel


def _tiny_decoder():
    return MAEDecoder(
        num_patches=8, enc_dim=32, dec_dim=16, depth=1, nheads=2,
        patch_size=50, in_ch=12,
    )


def _tiny_model():
    encoder = ViTSmall1D(
        signal_length=200, patch_size=50, embed_dim=32, depth=1, num_heads=4,
    )
    decoder = MAEDecoder(
        num_patches=4, enc_dim=32, dec_dim=16, depth=1, nheads=2,
        patch_size=50, in_ch=12,
    )
    return MAEModel(encoder, decoder, mask_ratio=0.5)


def _run_decoder(device, amp_device, dtype):
    dec = _tiny_decoder().to(device)
    B, N, nv = 2, 8, 2
    enc_vis = torch.randn(B, nv, 32, device=device)
    vis_ids = torch.arange(nv, device=device).expand(B, -1)
    mask_ids = torch.arange(nv, N, device=device).expand(B, -1)
    with torch.amp.autocast(amp_device, enabled=True, dtype=dtype):
        out = dec(enc_vis, vis_ids, mask_ids, N)
    assert out.shape == (B, N - nv, 50 * 12)
    assert torch.isfinite(out.float()).all()
    assert dec.mask_token.dtype == torch.float32


def _run_model(device, amp_device, dtype):
    model = _tiny_model().to(device)
    x = torch.randn(2, 12, 200, device=device)
    with torch.amp.autocast(amp_device, enabled=True, dtype=dtype):
        loss, pred, mask_ids = model(x)
    assert torch.isfinite(loss.float())
    assert pred.shape[0] == x.shape[0]
    assert mask_ids.shape[0] == x.shape[0]


def test_mae_decoder_cpu_bf16_autocast():
    _run_decoder("cpu", "cpu", torch.bfloat16)


def test_mae_model_cpu_bf16_autocast():
    _run_model("cpu", "cpu", torch.bfloat16)


def test_mae_decoder_cuda_fp16_autocast():
    if not torch.cuda.is_available():
        return
    _run_decoder("cuda", "cuda", torch.float16)


def test_mae_model_cuda_fp16_autocast():
    if not torch.cuda.is_available():
        return
    _run_model("cuda", "cuda", torch.float16)
