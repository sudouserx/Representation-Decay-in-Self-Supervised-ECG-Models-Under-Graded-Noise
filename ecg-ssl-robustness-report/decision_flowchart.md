```mermaid
graph TD
    Start[New ECG Task] --> Q1{Is latency critical <100ms?}
    
    Q1 -- Yes --> Q2{Is strict calibration <5% ECE required?}
    Q1 -- No --> Q3{Is noise tolerance -6dB SNR critical?}
    
    Q2 -- Yes --> M1[ssl-byol-vit-small-seed42 (if it passes ECE gate) else fallback]
    Q2 -- No --> M2[ssl-byol-vit-small-seed42]
    
    Q3 -- Yes --> M3[ssl-supervised-vit-small-seed42]
    Q3 -- No --> M4[ssl-byol-vit-small-seed42 (Best balanced robustness index)]
```
