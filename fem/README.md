# FEM & Extension Models

106 physics / industry modules. Only a subset is required for the core paper — see [docs/TIERS.md](../docs/TIERS.md).

## Core (SiC dicing physics)

| Module | Physics basis |
|--------|---------------|
| `dicing_blade_2d.py` | 2D blade dicing FEM (fracture/chipping) |
| `dicing_blade_3d.py` | 3D blade dicing FEM |
| `crystal_anisotropy.py` | 4H-SiC K_Ic direction dependence |
| `grinding_warpage_2d.py` | Backside grinding → residual stress → warpage |
| `kabra_thermal_2d.py` | KABRA laser slicing thermo-mechanical |
| `stealth_dicing_crack_model.py` | Stealth dicing K_I / crack propagation |
| `dbg_sdbg_model.py` | DBG/SDBG Weibull die strength |

## Keyence stack (tested: `tests/test_keyence.py`)

| Module | Product |
|--------|---------|
| `keyence_metrology_model.py` | VK-X3100 confocal |
| `keyence_lj_profiler.py` | LJ-X8080 laser profiler |
| `keyence_iv3_vision.py` | IV3 machine vision |
| `keyence_business_model.py` | Business / pricing model |

## TEL FEOL stack (tested: `tests/test_tel_feol.py`)

| Module | Role |
|--------|------|
| `tel_feol_recipe.py` | Recipe → FOM forward model |
| `tel_process_model.py` | ALD / oxidation / RIE |
| `tel_cleaning_model.py` | Cleaning sequence |
| `tel_equipment_competitiveness.py` | Equipment spec → yield MC |

## DISCO portfolio

| Module | Role |
|--------|------|
| `disco_machine_struct.py` | Spindle housing / frame FEM |
| `disco_elec_system.py` | DC bus, inverter, linear motor |
| `disco_sw_stack.py` (in `pipeline/`) | IEC 61131-3 state machine |

## Industry / market models (30+)

`tsmc_model.py`, `nvidia_gpu_model.py`, `asml_model.py`, `lam_research_model.py`, etc. — competitive landscape sims for portfolio demos. Not paper-critical.

## Adding a new module

1. Create `fem/your_model.py` with a `main()` or `demo()` entry point
2. Add tests in `tests/test_your_model.py`
3. Document in this README under the appropriate section
4. Do **not** import from core `validation/` unless feeding experimental data