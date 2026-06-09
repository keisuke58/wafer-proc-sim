(*
 * InterlockFB.st — Safety interlock monitor Function Block
 * IEC 61131-3 Structured Text  /  IEC 61508 SIL-1 inspired
 *
 * Each sensor has WARN / STOP / FAULT thresholds.
 * FAULT latches E_STOP until operator reset.
 *)

FUNCTION_BLOCK InterlockFB
VAR_INPUT
    Spindle_rpm     : REAL;     (* spindle speed readback *)
    Coolant_lpm     : REAL;     (* coolant flow [L/min] *)
    Door_closed     : BOOL;     (* safety door sensor *)
    BladeDepth_mm   : REAL;     (* blade Z position [mm, negative = into wafer] *)
    ChuckVac_kPa    : REAL;     (* chuck vacuum [kPa] *)
    BladeWear_um    : REAL;     (* estimated blade wear [µm] *)
    Reset           : BOOL;     (* operator reset pulse *)
END_VAR

VAR_OUTPUT
    System_OK       : BOOL;     (* TRUE = safe to run *)
    E_Stop          : BOOL;     (* latched emergency stop *)
    WarnActive      : BOOL;     (* at least one WARN *)
    FaultCode       : INT;      (* 0=OK 1=SpindleOvspd 2=NoCoolant
                                   3=DoorOpen 4=BladeDeep 5=VacLow
                                   6=WearHigh *)
END_VAR

VAR
    (* threshold constants *)
    SP_WARN_HI  : REAL := 32000.0;
    SP_FAULT_HI : REAL := 35000.0;
    CL_FAULT_LO : REAL := 0.5;
    BD_WARN_LO  : REAL := -0.35;
    BD_FAULT_LO : REAL := -0.50;
    VC_WARN_LO  : REAL := 60.0;
    WR_WARN_HI  : REAL := 30.0;
    WR_FAULT_HI : REAL := 50.0;

    _estop_latch : BOOL := FALSE;
END_VAR

    (* Reset latch on rising edge of Reset *)
    IF Reset AND _estop_latch THEN
        _estop_latch := FALSE;
    END_IF

    WarnActive := FALSE;
    FaultCode  := 0;

    (* Check spindle overspeed *)
    IF ABS(Spindle_rpm) > SP_FAULT_HI THEN
        _estop_latch := TRUE;
        FaultCode    := 1;
    ELSIF ABS(Spindle_rpm) > SP_WARN_HI THEN
        WarnActive := TRUE;
    END_IF

    (* Check coolant flow *)
    IF Coolant_lpm < CL_FAULT_LO THEN
        _estop_latch := TRUE;
        IF FaultCode = 0 THEN FaultCode := 2; END_IF
    END_IF

    (* Check door *)
    IF NOT Door_closed THEN
        _estop_latch := TRUE;
        IF FaultCode = 0 THEN FaultCode := 3; END_IF
    END_IF

    (* Check blade depth (more negative = deeper cut) *)
    IF BladeDepth_mm < BD_FAULT_LO THEN
        _estop_latch := TRUE;
        IF FaultCode = 0 THEN FaultCode := 4; END_IF
    ELSIF BladeDepth_mm < BD_WARN_LO THEN
        WarnActive := TRUE;
    END_IF

    (* Check chuck vacuum *)
    IF ChuckVac_kPa < VC_WARN_LO THEN
        WarnActive := TRUE;
        IF FaultCode = 0 THEN FaultCode := 5; END_IF
    END_IF

    (* Check blade wear *)
    IF BladeWear_um > WR_FAULT_HI THEN
        _estop_latch := TRUE;
        IF FaultCode = 0 THEN FaultCode := 6; END_IF
    ELSIF BladeWear_um > WR_WARN_HI THEN
        WarnActive := TRUE;
    END_IF

    E_Stop    := _estop_latch;
    System_OK := NOT _estop_latch;

END_FUNCTION_BLOCK
