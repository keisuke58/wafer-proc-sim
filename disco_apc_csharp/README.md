# DISCO APC (C#) — Run-to-Run recipe correction + EWMA anomaly monitor

A **C# / .NET** implementation of the two algorithms at the heart of the DISCO
"AI algorithm engineer" role: **dynamic recipe correction (Run-to-Run control)**
and **equipment-data anomaly detection** — the detect → diagnose → correct loop,
in the language the product software is written in.

This complements the Python/C++ process-control modules elsewhere in the repo
(grinding / CMP / laser APC) by porting the control + monitoring core to C#,
with idiomatic structure, XML doc comments, and a dependency-free test suite.

## What it does

- **`RunToRunController`** — EWMA Run-to-Run controller. Estimates a drifting
  process offset from lot-to-lot measurements and inverts the process model to
  pick the next recipe input: `u_next = (target − offsetEstimate) / gain`.
- **`EwmaAnomalyMonitor`** — EWMA control-chart monitor with time-varying limits;
  flags small sustained shifts (tool drift, consumable faults) that a raw 3σ
  check would miss. Starts from the in-control center to avoid startup false
  alarms.
- **`ProcessSimulator` / `DeterministicRandom`** — reproducible synthetic
  equipment process (linear drift + injected step fault + noise; splitmix64 RNG)
  for generating evaluation data — the "simulation for evaluation-environment
  automation" the job description calls out.
- **`Program` (demo)** — runs 200 lots open-loop vs closed-loop, injects a fault,
  prints an operator-console dashboard, and writes `disco_apc_results.json` and a
  self-contained `disco_apc_chart.svg`.

## Result (from the committed run)

| metric | open-loop | closed-loop (R2R) |
|---|---|---|
| RMSE to target | 4.61 | **0.45** |
| Cpk | −0.33 | **0.67** |
| in-spec yield | 17 % | **97 %** |

Anomaly monitor: flags the injected fault immediately at the fault lot, with
**0 false alarms** in the pre-fault region.

## Build & run

```bash
./build.sh            # auto-detects dotnet or Mono; runs tests then the demo
# or, explicitly with Mono:
mcs -target:library -out:AlgoLib.dll src/*.cs
mcs -r:AlgoLib.dll -out:Tests.exe tests/Tests.cs && mono Tests.exe
mcs -r:AlgoLib.dll -out:Demo.exe  demo/Program.cs && mono Demo.exe .
```

The library targets `netstandard2.0` (LangVersion 7.3) so it builds with the
.NET SDK (`dotnet build AlgoLib.csproj`) and with Mono `mcs` alike.

## Tests

`tests/Tests.cs` is a dependency-free harness (exit 0 = pass) so it runs without
NuGet/xUnit: controller inversion & EWMA-blend math, constructor guards,
closed-loop beats open-loop under drift, steady-state error bound, and anomaly
detection with a bounded pre-fault false-alarm count. **11 checks, all passing.**
