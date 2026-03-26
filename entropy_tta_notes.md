# Entropy-Threshold Selective TTA

## Goal
Replace fixed-quantile selective TTA with an entropy-based gate that adapts to per-run uncertainty.

In the current pipeline, low-confidence samples are selected using:
- top-1 softmax confidence
- fixed quantile (`UNCERTAIN_QUANTILE`)

This can be suboptimal because it always selects a fixed fraction, even when a run is globally easy or hard.

## Core Idea
For each sample:
1. Compute logits with base inference.
2. Convert logits to probabilities with optional temperature scaling.
3. Compute entropy:

`H(p) = -sum_i p_i * log(p_i)`

4. Run heavy TTA only when entropy exceeds a threshold `tau_entropy`.

## Why This Can Help
- Adaptive compute: number of TTA samples varies with uncertainty.
- Better uncertainty signal than max-probability in some cases.
- Can improve the accuracy/time tradeoff versus fixed quantile.

## Proposed Parameters
- `tau_entropy` (float): main threshold.
- `temperature` (float, default `1.0`): for calibrated probabilities before entropy.
- `max_tta_fraction` (float, optional): hard cap on fraction of samples receiving TTA.
- `min_tta_fraction` (float, optional): lower floor to avoid zero-TTA collapse on easy runs.

## Implementation Sketch (PyTorch)
```python
def select_uncertain_by_entropy(initial_logits, tau_entropy, temperature=1.0,
                                min_frac=0.0, max_frac=1.0):
    probs = torch.softmax(initial_logits / temperature, dim=1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)

    mask = entropy > tau_entropy
    idx = torch.where(mask)[0]

    # Optional budget controls
    n = initial_logits.shape[0]
    min_k = int(min_frac * n)
    max_k = int(max_frac * n)

    if idx.numel() < min_k:
        _, top_idx = torch.topk(entropy, k=min_k, largest=True, sorted=False)
        idx = top_idx
    elif idx.numel() > max_k:
        _, top_idx = torch.topk(entropy, k=max_k, largest=True, sorted=False)
        idx = top_idx

    return idx, entropy
```

Use `idx` instead of quantile-selected indices in the TTA branch.

## Integration Points in `cifar10_speedrun.py`
- `infer() -> tta()`:
  - Replace current quantile block:
    - `confidences = softmax(...).max(...)`
    - `topk(... largest=False ...)`
  - With entropy-based selection function.
- Keep the rest of TTA path unchanged (batching and replacing final logits for selected indices).

## Evaluation Plan
Run multi-run sweeps and compare against current quantile baseline:

1. Baseline:
- existing `UNCERTAIN_QUANTILE = 0.25`

2. Entropy sweep:
- `tau_entropy` grid (example): `0.4, 0.6, 0.8, 1.0`
- `temperature` grid: `1.0, 1.2, 1.5`
- optional `max_tta_fraction`: `0.15, 0.25, 0.35`

Track:
- `tta_val_acc` mean/std
- `time_seconds` mean/std
- selected TTA fraction per run

## Success Criteria
- Better or equal mean `tta_val_acc`
- Lower or equal mean `time_seconds`
- Stable variance over repeated runs

## Notes / Risks
- Entropy depends on calibration; temperature scaling may be required.
- Too high `tau_entropy` can under-select and hurt accuracy.
- Too low `tau_entropy` can over-select and erase speed gains.
- If run-to-run distribution shifts, dynamic thresholding with a budget cap is safer than a fixed global threshold.

