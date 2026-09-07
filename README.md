# WaferFusion-Cascade: SanDisk Die Yield Prediction System

WaferFusion-Cascade is a multi-resolution ML system for the SanDisk Die Yield
Hackathon. Model A screens die-level parametric and wafer-spatial context with
LightGBM; Model B adds a PyTorch CNN over 2,000 block readings. Evaluation uses
five-fold wafer-grouped out-of-fold (OOF) predictions with preprocessing fitted
inside each fold.

The final saved operating policy is the single-model CNN cascade with a
cost-weighted 8:1 false-negative/false-positive ratio. The test-set result is a
confirmatory check, not a fully untouched final evaluation, because candidate
cost ratios were inspected on that test set. See `outputs/reports/final_report.md`.

---

## 1. System Architecture

```
Raw Data
   ↓
Data Validation (schema, nulls, types, block length)
   ↓
Feature Engineering (500+ parametric, multi-scale spatial 3x3/5x5/7x7, local z-scores)
   ↓
Model A: Fast Screening (LightGBM with XGBoost/HistGBM fallbacks)
   ↓
OOF-derived uncertainty gate (routing heuristic; no coverage guarantee)
   ↓
Cascade Safety Router
   ├── Confident → Pure Model A resolution (Fast)
   └── Ambiguous / Suspicious / Boundary Margin → Model B
                                                  ↓
                                       Deep 1D CNN Block Encoder
                                                  ↓
                                       Evidence Fusion (Modal Gating)
                                                  ↓
                                        Final uncalibrated risk score
                                                  ↓
                                       Submission & Visualizations
```

The current saved calibrators are intentionally identity transforms so OOF
thresholds and inference scores remain on the same scale. The routing gate is a
heuristic built from OOF scores; this project does not claim split-conformal
coverage or calibrated probabilities.

---

## 2. Directory Structure

```text
sandisk_die_yield/
├── configs/
│   ├── base.yaml              # Master configuration
│   ├── cpu.yaml               # CPU development profile
│   └── gpu.yaml               # CUDA GPU acceleration profile
├── src/
│   └── sandisk_yield/
│       ├── schema.py          # Column constants, target & eligibility masking
│       ├── seed.py            # Reproducibility seeds (Python, NumPy, PyTorch)
│       ├── config.py          # Config loader supporting deep inheritance
│       ├── logging_utils.py   # Formatted logging handlers
│       ├── data/
│       │   ├── loader.py      # Autodetect CSV, Parquet, Pickle loaders
│       │   ├── validator.py   # Schema & constraint validation
│       │   └── splitter.py    # Grouped wafer-level CV & calibration splitters
│       ├── features/
│       │   ├── parametric.py  # Robust imputation & scaling
│       │   ├── spatial.py     # Geometry & old-failure neighborhood densities
│       │   ├── wafer.py       # Wafer context & local process z-scores
│       │   ├── blocks.py      # Block parser & explicit anomaly features
│       │   └── pipeline.py    # Unified feature transformer
│       ├── models/
│       │   ├── baseline.py    # LogisticRegression baseline
│       │   ├── model_a.py     # LightGBM screening classifier
│       │   ├── block_encoder.py # PyTorch 1D CNN with attention
│       │   ├── fusion.py      # Evidence fusion network with learned gating
│       │   ├── model_b.py     # End-to-end deep inspection model
│       │   └── calibration.py # Isotonic probability calibrator
│       ├── cascade/
│       │   ├── uncertainty.py # Entropy & boundary margin metrics
│       │   ├── conformal_gate.py # Split-conformal uncertainty gate
│       │   ├── router.py      # Safety routing policy
│       │   └── cascade.py     # End-to-end cascade orchestrator
│       ├── training/
│       │   ├── losses.py      # Imbalanced Focal Loss
│       │   ├── evaluation.py  # Die yield PR-AUC & threshold search
│       │   ├── trainer_a.py   # Grouped CV Model A trainer
│       │   ├── trainer_b.py   # PyTorch Model B trainer
│       │   └── trainer_cascade.py # Cascade calibration coordinator
│       ├── explainability/
│       │   ├── tree_explain.py # Feature importance extraction
│       │   ├── block_explain.py # Block saliency localization
│       │   └── risk_decomposition.py # Die Risk Card generator
│       ├── risk/
│       │   └── wafer_risk.py  # Wafer-level summary metrics
│       ├── visualization/
│       │   ├── wafer_map.py   # 2D spatial wafer risk maps
│       │   └── curves.py      # PR, ROC, Calibration curves
│       └── inference/
│           ├── predict.py     # High-level inference orchestrator
│           └── submission.py  # Strict submission validator & exporter
├── scripts/
│   ├── validate_data.py       # Dataset validation CLI
│   ├── build_features.py      # Feature engineering CLI
│   ├── run_all.py             # Master full execution pipeline
│   └── predict.py             # Submission prediction CLI
├── dashboard/
│   └── app.py                 # Interactive Streamlit dashboard
└── tests/
    ├── test_no_leakage.py     # Strict data-leakage boundary tests
    ├── test_cascade.py        # Conformal gate & routing tests
    ├── test_model_b.py        # PyTorch Model B forward pass tests
    ├── test_submission.py     # Submission validation tests
    ├── test_schema.py         # Schema tests
    └── test_validator.py      # Multi-level data validator tests
```

### Package-layout note

The active pipeline imports from `src/sandisk_yield/`. The smaller top-level
`src/data/` and `src/features/` packages are retained only because legacy schema
and validator tests import them. Top-level `src/models/` and `src/evaluation/`
are compatibility package markers; new application code should not import them.

---

## 3. Setup and judge walkthrough

From PowerShell on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps --no-build-isolation
```

Place the confidential files at exactly:

```text
input/train.csv
input/validation.csv
input/test.csv
```

They are ignored by Git. Model binaries and row-level prediction exports are
also intentionally excluded. A fresh clone therefore needs locally supplied
data and either the separately supplied frozen model artifacts or a deliberate
training run.

### Step 1: Run the offline test suite

```powershell
python -m pytest tests -v
```

### Step 2: Reproduce reports from existing frozen artifacts—no retraining

```powershell
python scripts/run_all.py --postprocess-only --threshold-objective cost_weighted --fn-fp-cost-ratio 8 --ensemble-mode none
python scripts/analyze_thresholds.py --test-labels input/test.csv --ratios 4 8 12
python scripts/generate_analysis_deliverables.py
```

The first two commands use saved OOF/test scores. The third performs feature
transformation and explanation with frozen models; it verifies model hashes did
not change. None of these commands fits a model or changes the submission.

### Step 3: Open the dashboard

```powershell
python -m streamlit run dashboard/app.py
```

Open `http://localhost:8501`. The judge-facing tabs include Benchmark
Comparison, Interpretability, and Imbalance Analysis.

### Optional: deliberately train the full pipeline

This is slow and replaces local model/output artifacts:

```powershell
python scripts/run_all.py --config configs/base.yaml --block-mode cnn --compare-block-modes
```

Do not run this merely to view the existing evidence.

### Recreate the final Cost 8:1 submission from saved probabilities

```powershell
python scripts/predict.py --from-probabilities outputs/predictions/prediction_probabilities.csv --threshold-objective cost_weighted --fn-fp-cost-ratio 8 --ensemble-mode none --output outputs/predictions/submission.csv
```

This command re-thresholds saved scores without model inference. The general CLI
default remains F1, so keep the explicit Cost 8:1 flags for the selected policy.

## 4. Current evidence

- `outputs/metrics/operating_point_comparison.csv`: Model A/Model B OOF metrics.
- `outputs/reports/final_report.md`: exact final submission metrics and caveat.
- `outputs/reports/model_a_per_die_shap.csv`: current per-die TreeSHAP evidence.
- `outputs/reports/model_a_spatial_contribution.png`: spatial SHAP map.
- `outputs/reports/model_b_block_pattern_analysis_current.csv`: CNN attention regions.
- `outputs/reports/imbalance_analysis/summary.md`: imbalance and overlap interpretation.

CNN attention is supporting evidence about regions emphasized by the encoder;
it is not presented as a causal explanation. TreeSHAP values are LightGBM
log-odds contributions.
