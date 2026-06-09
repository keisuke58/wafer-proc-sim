(*
 * DicingController.st — Top-level DISCO dicing saw PLC program
 * IEC 61131-3 Structured Text
 *
 * Integrates SpindleFB, InterlockFB, RecipeSeqFB into one scan cycle.
 * Mirrors the real machine controller architecture: each FB encapsulates
 * one subsystem; the main PROGRAM wires I/O and coordinates state.
 *
 * Compile with: OpenPLC Runtime  or  CODESYS V3.5
 *)

PROGRAM DicingController

VAR
    (* Function block instances *)
    Spindle     : SpindleFB;
    Interlock   : InterlockFB;
    Recipe      : RecipeSeqFB;

    (* HMI inputs (written by operator panel / SCADA) *)
    HMI_Start       : BOOL := FALSE;
    HMI_Abort       : BOOL := FALSE;
    HMI_Reset       : BOOL := FALSE;
    HMI_SpeedRef    : REAL := 30000.0;  (* [rpm] *)

    (* Sensor inputs (mapped to physical I/O in real machine) *)
    IN_CoolantFlow  : REAL := 2.0;      (* [L/min] *)
    IN_DoorClosed   : BOOL := TRUE;
    IN_BladeDepth   : REAL := 0.0;      (* [mm] *)
    IN_ChuckVac     : REAL := 80.0;     (* [kPa] *)
    IN_BladeWear    : REAL := 0.0;      (* [µm] — from APC/EnKF *)

    (* Internal *)
    StepDone_pulse  : BOOL := FALSE;
    dt_ms           : REAL := 1.0;      (* scan cycle [ms] *)

    (* Status outputs (to HMI / data logger) *)
    OUT_Speed_rpm   : REAL;
    OUT_Torque_Nm   : REAL;
    OUT_RecipeState : INT;
    OUT_StepName    : STRING;
    OUT_Progress    : REAL;
    OUT_EStop       : BOOL;
    OUT_FaultCode   : INT;
END_VAR

    (* 1. Interlock evaluation — always first *)
    Interlock(
        Spindle_rpm   := OUT_Speed_rpm,
        Coolant_lpm   := IN_CoolantFlow,
        Door_closed   := IN_DoorClosed,
        BladeDepth_mm := IN_BladeDepth,
        ChuckVac_kPa  := IN_ChuckVac,
        BladeWear_um  := IN_BladeWear,
        Reset         := HMI_Reset
    );

    (* 2. Recipe sequencer *)
    Recipe(
        Start        := HMI_Start,
        Abort        := HMI_Abort OR Interlock.E_Stop,
        StepDone     := StepDone_pulse,
        Interlock_OK := Interlock.System_OK,
        dt_ms        := dt_ms
    );

    (* 3. Spindle — only run when recipe is active and interlocks clear *)
    Spindle(
        Enable       := Interlock.System_OK AND (Recipe.State = 1),
        SpeedRef_rpm := HMI_SpeedRef,
        dt_us        := 10.0
    );

    (* 4. Propagate fault from spindle back to interlock *)
    IF Spindle.Fault THEN
        HMI_Abort := TRUE;
    END_IF

    (* 5. Collect outputs for HMI *)
    OUT_Speed_rpm   := Spindle.Speed_rpm;
    OUT_Torque_Nm   := Spindle.Torque_Nm;
    OUT_RecipeState := Recipe.State;
    OUT_StepName    := Recipe.StepName;
    OUT_Progress    := Recipe.Progress_pct;
    OUT_EStop       := Interlock.E_Stop;
    OUT_FaultCode   := Interlock.FaultCode;

END_PROGRAM
