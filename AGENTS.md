# Agent Instructions — wafer-proc-sim

## Project goal

Reproduce and extend **4H-SiC blade dicing** surrogate modeling (heteroscedastic GP + EnKF + BO). The paper spine is documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Start here

| Task | Entry point |
|------|-------------|
| Reproduce paper | `./scripts/reproduce_paper.sh` |
| DISCO interview demo | `./scripts/demo_disco.sh` |
| Train GP | `python ml/train_from_experimental.py --loo --quality AB` |
| Run pipeline | `python pipeline/sic_dicing_pipeline.py --no-plots` |
| Run tests | `pytest tests/ -v` |
| Paper figures | `python paper/sic_gp_quality/gen_figures.py` |

## Do not touch

- `results/*.pkl`, `results/*.png` — regenerated artifacts (gitignored)
- `data/external/` — large downloaded datasets
- `runs/**` — simulation outputs (except `runs/examples/`)

## Where to add new code

- **Paper / core science** → `validation/`, `ml/`, `pipeline/sic_dicing_pipeline.py`
- **Portfolio / vendor demos** → `fem/`, `pipeline/` (non-core), `demos/`
- **Never** add industry models to `validation/experimental_data.py`

## Pre-approved commands (project `.grok/settings.json`)

- `python pipeline/run_full_pipeline.py --process stealth`
- `python fem/intel_model.py`

## Testing

- Always run `pytest tests/ -m "not slow"` after small changes
- Run `pytest tests/test_paper_metrics.py` after GP / data changes
- Paper metric targets: all grades LOO RMSE ≈ 3.12 µm; Grade A/B LOO RMSE ≈ 1.62 µm, R² ≈ 0.80

## Dependencies

```bash
pip install -e ".[dev]"    # core + pytest
pip install -e ".[full]"   # + streamlit, pennylane, etc.
```