(*
 * RecipeSeqFB.st — Dicing recipe sequencer Function Block
 * IEC 61131-3 Structured Text
 *
 * Drives the 12-step DISCO dicing recipe as a CASE state machine.
 * Interlocks are checked at every scan.
 *)

FUNCTION_BLOCK RecipeSeqFB
VAR_INPUT
    Start       : BOOL;     (* operator start command *)
    Abort       : BOOL;     (* operator abort *)
    StepDone    : BOOL;     (* rising edge = current step completed *)
    Interlock_OK: BOOL;     (* from InterlockFB.System_OK *)
    dt_ms       : REAL := 1.0;
END_VAR

VAR_OUTPUT
    State       : INT;      (* 0=IDLE 1=RUNNING 2=DONE 3=FAULT *)
    StepIndex   : INT;      (* 0..11 *)
    StepName    : STRING;
    Progress_pct: REAL;
    Fault       : BOOL;
END_VAR

VAR
    _elapsed_ms     : REAL := 0.0;
    _step_done_prev : BOOL := FALSE;
    _step_done_rise : BOOL;

    (* Step timeouts [ms] *)
    TIMEOUT : ARRAY[0..11] OF REAL := [
        5000.0,   (* 0  LOAD_WAFER     *)
        2000.0,   (* 1  VACUUM_ON      *)
        8000.0,   (* 2  ALIGN_WAFER    *)
        3000.0,   (* 3  PRIME_SPINDLE  *)
        1000.0,   (* 4  COOLANT_ON     *)
        60000.0,  (* 5  CUT_X_STREETS  *)
        2000.0,   (* 6  ROTATE_CHUCK   *)
        60000.0,  (* 7  CUT_Y_STREETS  *)
        2000.0,   (* 8  RETRACT_BLADE  *)
        1000.0,   (* 9  COOLANT_OFF    *)
        3000.0,   (* 10 SPINDLE_OFF    *)
        5000.0    (* 11 UNLOAD_WAFER   *)
    ];

    STEP_NAMES : ARRAY[0..11] OF STRING := [
        'LOAD_WAFER', 'VACUUM_ON', 'ALIGN_WAFER', 'PRIME_SPINDLE',
        'COOLANT_ON', 'CUT_X_STREETS', 'ROTATE_CHUCK', 'CUT_Y_STREETS',
        'RETRACT_BLADE', 'COOLANT_OFF', 'SPINDLE_OFF', 'UNLOAD_WAFER'
    ];

    TOTAL_STEPS : INT := 12;
END_VAR

    (* Detect rising edge on StepDone *)
    _step_done_rise := StepDone AND NOT _step_done_prev;
    _step_done_prev := StepDone;

    CASE State OF

        0: (* IDLE *)
            StepIndex   := 0;
            Progress_pct := 0.0;
            _elapsed_ms  := 0.0;
            Fault        := FALSE;
            IF Start AND Interlock_OK THEN
                State := 1;
            END_IF

        1: (* RUNNING *)
            StepName := STEP_NAMES[StepIndex];
            Progress_pct := REAL_TO_REAL(StepIndex) / REAL_TO_REAL(TOTAL_STEPS) * 100.0;

            IF NOT Interlock_OK OR Abort THEN
                State := 3;  (* → FAULT *)
            END_IF

            _elapsed_ms := _elapsed_ms + dt_ms;

            (* Timeout check *)
            IF _elapsed_ms > TIMEOUT[StepIndex] THEN
                State := 3;
                Fault := TRUE;
            END_IF

            (* Step complete *)
            IF _step_done_rise THEN
                _elapsed_ms := 0.0;
                IF StepIndex >= TOTAL_STEPS - 1 THEN
                    State        := 2;
                    Progress_pct := 100.0;
                ELSE
                    StepIndex := StepIndex + 1;
                END_IF
            END_IF

        2: (* DONE *)
            Fault := FALSE;

        3: (* FAULT *)
            Fault := TRUE;
            IF NOT Abort AND Start AND Interlock_OK THEN
                State     := 0;  (* reset to IDLE after operator clears *)
                StepIndex := 0;
            END_IF

    END_CASE

END_FUNCTION_BLOCK
