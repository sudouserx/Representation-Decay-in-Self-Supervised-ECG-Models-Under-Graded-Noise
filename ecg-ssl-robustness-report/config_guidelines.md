# Configuration Guidelines
## Condition: min-SNR mean across noise types (-6.0 dB)

**Status:** RECOMMEND_QUALIFIED

### Recommended Configuration:
- **Model**: ssl-clocs-resnet18-seed42
- **Quantization**: fp32
- **Hardware**: CPUExecutionProvider (ONNX execution provider — server-proxy, not edge hardware identity)

### Evidence:
- **Robustness index**: 0.8855
- **Weight-stability (Kendall τ rank retention)**: 0.832
- **Latency p95 (ms)**: 58.94
- **Memory (process RSS delta, MB)**: 0.00
- **Quantization parity cosine**: n/a

### Applied Constraints:
- Min Robustness: 0.5
- Max Latency (ms): 100.0
- Max Memory (MB): 512.0

### Alternative Options:
- **ssl-clocs-resnet18-seed42 (int8_static on CPUExecutionProvider)**: index 0.8855, latency 30.0 ms, status RECOMMEND_QUALIFIED
- **ssl-supervised-vit-small-seed456 (fp32 on CPUExecutionProvider)**: index 0.8436, latency 39.2 ms, status RECOMMEND_QUALIFIED
- **ssl-supervised-vit-small-seed456 (int8_dynamic on CPUExecutionProvider)**: index 0.8436, latency 33.6 ms, status RECOMMEND_QUALIFIED
