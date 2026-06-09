# PLC programs — IEC 61131-3 Structured Text

DISCO dicing saw controller logic written in IEC 61131-3 Structured Text (ST).
Mirrors the C++ simulation kernels but in the language that actually runs on
industrial PLCs (CODESYS, Beckhoff TwinCAT, Siemens S7, OpenPLC).

## Files

| File | Description |
|------|-------------|
| `SpindleFB.st`       | PMSM + PI-FOC current controller Function Block |
| `InterlockFB.st`     | Safety interlock monitor (IEC 61508 SIL-1 pattern) |
| `RecipeSeqFB.st`     | 12-step dicing recipe sequencer (CASE state machine) |
| `DicingController.st`| Top-level PROGRAM — wires all FBs to I/O |

## Compile & simulate

### Option A — OpenPLC (free, Linux/Windows/Raspberry Pi)
```bash
# Install: https://openplcproject.com/
# 1. Create new project in OpenPLC Editor
# 2. Add all .st files as POUs (Program Organisation Units)
# 3. Build → Run on soft-PLC
# 4. Monitor variables via Modbus TCP client
```

### Option B — CODESYS V3.5 (free tier available)
```
1. File → New Project → Standard Project
2. Add POU → paste each .st file
3. Build (F11) — should compile with 0 errors
4. Simulate → PLC_PRG calls DicingController
```

## Architecture

```
DicingController  (PROGRAM — scan cycle entry point)
    │
    ├── InterlockFB   — reads sensor inputs, latches E_STOP
    ├── RecipeSeqFB   — IDLE→RUNNING→DONE CASE state machine
    └── SpindleFB     — PMSM dq-axis Euler integration
```

This structure matches the standard DISCO machine controller pattern:
- Safety functions (interlock) evaluated **first** every scan
- Recipe sequencer drives machine-level state
- Low-level FB (spindle, axis) execute conditionally on recipe state
