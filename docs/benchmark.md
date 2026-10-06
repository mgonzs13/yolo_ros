# Benchmark: CPU vs CUDA vs TensorRT

See the [README](../README.md) for the short project overview.

Same model (`yolo26m`), same 640x480 frames and same detection settings (`imgsz=640`, `conf=0.7`, ~8.7 detections/frame); only the execution provider changes. Latency is the median per-frame round trip with a single frame in flight, throughput the saturated end-to-end rate.

Measured on a 12th Gen Intel Core i7-12700F (12 cores / 20 threads, 32 GB RAM) with an NVIDIA GeForce RTX 3060 (12 GB, driver 580.178.04), on Ubuntu 22.04 with CUDA 12.6 / cuDNN 9 and ONNX Runtime 1.20.0.

| Backend | Provider | Latency (median) | Throughput | CPU | RSS | GPU memory |
| :-- | :-- | --: | --: | --: | --: | --: |
| CPU | CPU EP | 158 ms | 4.5 FPS | ~18.5 cores | 535 MiB | — |
| CUDA | CUDA EP (fp32) | 22.6 ms | 49 FPS | ~1.4 cores | 1002 MiB | 478 MiB |
| TensorRT | TensorRT EP (fp16) | 6.2 ms | 194 FPS | ~1.3 cores | 3234 MiB | 240 MiB |
