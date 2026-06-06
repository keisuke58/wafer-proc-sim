# Repository Tiers

## Tier 1 — Core (paper reproduction)

Required for the *Precision Engineering* submission. CI must pass.

```
validation/experimental_data.py
ml/train_from_experimental.py
pipeline/sic_dicing_pipeline.py
paper/sic_gp_quality/
tests/test_paper_metrics.py
scripts/reproduce_paper.sh
```

## Tier 2 — Validation & tooling

Supporting analysis, tested in CI via `tests/`.

```
ml/sensitivity_analysis.py
ml/anomaly_detection.py
optimization/process_capability.py
optimization/blade_wear.py
optimization/cost_per_die.py
tests/test_physics.py
tests/test_keyence.py
tests/test_tel_feol.py
```

## Tier 3 — Extensions (portfolio)

Optional demos for equipment vendors, market models, and interview portfolios. **Do not add new files to Tier 1 paths** — place new work here:

| Area | Location | Examples |
|------|----------|----------|
| Physics models | `fem/` | `keyence_*.py`, `tel_*.py`, `disco_*.py` |
| Pipelines | `pipeline/` | `run_full_pipeline.py`, `keyence_apc.py` |
| Demos | `demos/`, `app.py` | Streamlit dashboards, slides |
| Results | `results/` | Generated PNG/PKL (gitignored) |

## Dependency tiers

| Install | Command |
|---------|---------|
| Core | `pip install -e .` or `pip install -r requirements-core.txt` |
| Full | `pip install -e ".[full]"` or `pip install -r requirements-full.txt` |
| Dev | `pip install -e ".[dev,full]"` |