# Career & Portfolio Context

This document covers DISCO-oriented portfolio mapping. It is **not** required for paper reproduction.

## DISCO engineering track coverage

| DISCO 職種 | カバレッジ | 主要ファイル |
|---|---|---|
| **メカ** (機械設計・構造FEM) | ◎ | `fem/disco_machine_struct.py`, `fem/dicing_blade_2d.py`, `fem/dicing_blade_3d.py`, `fem/stage_vibration_modal.py`, `fem/stage_cad_geometry.py` |
| **エレキ** (電気・制御・センサ) | ◎ | `fem/disco_elec_system.py`, `fem/spindle_motor_control.py`, `fem/wafer_thickness_sensor.py` |
| **プロセス** (プロセス研究・条件最適化) | ◎ | `fem/kabra_thermal_2d.py`, `fem/cmp_dry_polish_model.py`, `fem/stealth_dicing_crack_model.py`, `fem/plasma_etch_profile_model.py`, `fem/dbg_sdbg_model.py`, `optimization/bayesian_opt.py` |
| **R&D** (先行研究・代理モデル) | ◎ | `ml/train_from_experimental.py`, `ml/recipe_optimizer.py`, `ml/multifidelity_gp.py` |
| **ソフト** (機械制御ソフト・パイプライン) | ◎ | `pipeline/disco_sw_stack.py`, `pipeline/sic_dicing_pipeline.py`, `pipeline/keyence_apc.py` |

## Interview demo flow

1. **Core paper** — `./scripts/reproduce_paper.sh` (5 min): heteroscedastic GP LOO, EnKF pipeline
2. **Process depth** — `python pipeline/run_full_pipeline.py --process stealth`
3. **Software stack** — `python pipeline/disco_sw_stack.py` (state machine + recipe manager)
4. **Live monitor** — `streamlit run demos/paper_app.py` or `streamlit run streamlit_app.py`

## Related logs

- [DEVLOG.md](../DEVLOG.md) — development history (Keyence, TEL FEOL, etc.)