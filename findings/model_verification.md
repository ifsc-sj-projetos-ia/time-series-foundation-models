# Phase 0 — Model Verification

**Date**: 2026-06-17
**Environment**: Local laptop (GTX 1650, 4 GB VRAM, 8 GB RAM)

## Results

| Model | Params | Batch | Context | Forecast | Peak VRAM | Status |
|---|---|---|---|---|---|---|
| TTM2 (`r2`) | 805,280 | 69 | 512 | 96 | ~0.5 GB | ✅ |
| FlowState r1.0 | 9,069,312 | 1 | 2048 | 96 | ~3.5 GB | ✅ (batch=1) |
| FlowState r1.0 | 9,069,312 | 8 | 2048 | 96 | >3.9 GB | ❌ OOM |
| FlowState r1.1 | 18,500,000 | — | 4096 | 96 | >5 GB (est.) | ❌ needs Colab |

## Hardware Profile

- **GPU**: NVIDIA GeForce GTX 1650 (3.9 GB usable VRAM, CUDA 12.8)
- **RAM**: 8 GB system
- **OS**: Linux, Python 3.12
- **Torch**: 2.10.0+cu128

## Implications

1. **TTM2** runs all 69 prediction units in a single batch without issues.
2. **FlowState r1.0** fits at batch=1 — iterate over units one at a time and aggregate results. Peak VRAM ~3.5 GB leaves only ~0.4 GB margin on the GTX 1650.
3. **FlowState r1.1** cannot run locally — requires Google Colab (T4, 16 GB VRAM).
4. The OOM occurs in the S5 layer's FFT convolution (`modeling_flowstate.py:353`), where temporary `(2*L, B, H)` complex64 tensors are allocated.

## Commands

```bash
# Reproduce TTM2 test
python -c "
from tsfm_public import TinyTimeMixerForPrediction
import torch
m = TinyTimeMixerForPrediction.from_pretrained('ibm-granite/granite-timeseries-ttm-r2').to('cuda')
out = m(torch.randn(69, 512, 1).cuda())
print(out.prediction_outputs.shape)
"

# Reproduce FlowState r1.0 test (batch=1)
python -c "
from tsfm_public import FlowStateForPrediction
import torch
m = FlowStateForPrediction.from_pretrained('ibm-granite/granite-timeseries-flowstate-r1').to('cuda')
out = m(past_values=torch.randn(1, 2048, 1).cuda(), prediction_length=96, scale_factor=1.0)
print(out.prediction_outputs.shape)
"
```
