# Design of Experiments — Best Practices for TSFM Fine-tuning

Based on R. A. Fisher's principles (randomization, replication, blocking) and practical experience from TSFM-prediction-pipeline DOE sweeps.

---

## 1. Core Principles

### 1.1 Pre-register Response Variables

Decide which metric is "best" **before** running the first experiment. Do not cherry-pick after seeing results.

| Role | Metric | Why |
|---|---|---|
| **Primary** | SMAPE | Bounded 0–200%, robust to near-zero values in energy data |
| **Secondary** | MAE | Absolute error in original units, interpretable |
| **Tertiary** | RMSE | Catches peak-hour failures (large errors are quadratically penalized) |

A config that wins on SMAPE but ranks poorly on RMSE may be hiding dangerous peak-hour degradation.

### 1.2 Factor Selection

Limit to **3–5 factors**. Each extra factor doubles (full factorial) or nearly doubles (fractional) the run count.

| ID | Factor | Low (−1) | High (+1) | Type |
|---|---|---|---|---|
| A | `revision` | L2 (`ft-r2.1`) | L1 (`ft-l1-r2.1`) | Categorical |
| B | `freeze_backbone` | True (frozen) | False (unfrozen) | Categorical |
| C | `fewshot_fraction` | 0.05 | 0.20 | Continuous |
| D | `covariates` | None | Temperature + Price | Categorical |
| E | `context_length` | 512 | 1024 | Continuous |

**Rationale for each:**
- **A (revision)**: L1 pretraining produced better zero-shot results in LoadPrediction benchmarks. Does it also improve fine-tuning?
- **B (freeze_backbone)**: Freezing the backbone speeds training and reduces overfitting on small data. Unfreezing allows full adaptation.
- **C (fewshot_fraction)**: How much data is needed? 5% (~700 timesteps) vs 20% (~2800).
- **D (covariates)**: Does exogenous weather/price information improve forecasts after fine-tuning?
- **E (context_length)**: Longer context may capture weekly seasonality; shorter reduces memory and speeds training.

### 1.3 Design Matrix

**Never use one-factor-at-a-time (OFAT).** OFAT cannot detect interactions between factors.

Use a **Resolution V fractional factorial** (2⁵⁻¹ = 16 runs) for screening 5 factors. Resolution V means no main effect is aliased with any two-factor interaction (2FI), and no 2FI is aliased with another 2FI.

For 3 factors, use a full 2³ = 8 factorial.

**Always add center points** (mid-level values for continuous factors, safe defaults for categorical) to detect curvature. Add 2–4 center points.

**Always randomize run order** to prevent systematic bias from training order effects (GPU warm-up, learning rate decay alignment, data caching).

**Always replicate at least one config** (the center point or the expected best) to estimate pure error.

### 1.4 Execution Protocol

```
1. Generate design matrix (16 + center + replicates = ~20 runs)
2. Randomize run order
3. For each run:
   a. Log the config before starting
   b. Execute fine-tuning
   c. Save metrics immediately
   d. Log any errors (do not halt — continue to next run)
4. Collect all results into a single DataFrame
5. Run analysis protocol (never skip)
```

### 1.5 Anomaly Detection

Before analyzing, filter out anomalous runs:
- SMAPE > 2× median across all runs → likely diverged
- Training loss NaN or Inf → optimization failure
- Training error (exception during run) → log separately

Log anomalies in the summary report but **exclude from factor effect analysis**.

---

## 2. Analysis Protocol

### 2.1 Main Effects

```python
for factor in ["A", "B", "C", "D", "E"]:
    correlation = df[factor].corr(df["SMAPE"])
    print(f"{factor}: {correlation:.3f}")
```

Negative correlation → high level reduces SMAPE (good).
Positive correlation → high level increases SMAPE (bad).

### 2.2 Interaction Effects

The interactions most likely to matter for TSFM fine-tuning:

| Interaction | Hypothesis |
|---|---|
| A×D (revision × covariates) | L1 may benefit more from covariates (robust to outlier signals in weather) |
| B×D (freeze × covariates) | Unfreezing may be required to learn covariate signal |
| C×D (fewshot × covariates) | Covariates only useful with sufficient data |
| A×B (revision × freeze) | L1 may need unfrozen backbone to adapt; L2 may be fine frozen |

### 2.3 Cross-Metric Validation

```python
df["smape_rank"] = df["SMAPE"].rank()
df["rmse_rank"] = df["RMSE"].rank()
df["rank_delta"] = (df["smape_rank"] - df["rmse_rank"]).abs()
# Flag configs where SMAPE and RMSE disagree by > 3 ranks
df[df["rank_delta"] > 3]
```

A large rank delta means the config is good at proportional accuracy (SMAPE) but bad at absolute accuracy (RMSE), or vice versa. This indicates peak-hour failures or systematic bias.

### 2.4 Per-Horizon Diagnostic

After selecting the winning config, compute SMAPE per forecast horizon step (t+1, t+2, ..., t+96):

```python
by_horizon = errors.groupby("horizon").SMAPE.mean()
```

Expect SMAPE to increase with horizon. A sharp jump at a specific horizon indicates a structural issue (e.g., failure to capture daily ramp-up at t+24).

---

## 3. Reporting

For each DOE phase, write a summary containing:

1. **Design**: factors, levels, number of runs, response variables
2. **Top 3 configs**: ranked by primary metric, with secondary metrics alongside
3. **Significant main effects**: list with correlation values, ranked by absolute magnitude
4. **Significant interactions**: list with interpretation
5. **Anomalies**: runs excluded and why
6. **Winner**: the recommended config with per-horizon diagnostic
7. **Recommendation**: what to try next (RSM on top 2 factors, or move to validation)

---

## 4. Practical Constraints for This Project

| Constraint | Mitigation |
|---|---|
| Fine-tuning is ~50× slower than zero-shot | Run screening DOE on **one representative unit** (unit 0), not all 69 |
| GTX 1650 has 4GB VRAM | Keep batch_size = 16 or lower; use `gradient_accumulation_steps` if needed |
| Each run takes ~5 min on unit 0 | 18-run DOE = ~90 min total, feasible in one session |
| Per-unit validation takes longer | Only validate the winning config on all 69 units |

---

## 5. References

- Fisher, R. A. (1935). *The Design of Experiments.*
- Box, G. E. P., Hunter, J. S., & Hunter, W. G. (1978). *Statistics for Experimenters.*
- Montgomery, D. C. (2013). *Design and Analysis of Experiments* (8th ed.).

See also: `implementation_plan.md` (Phase 4) for the project-specific DOE design.
