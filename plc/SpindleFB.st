(*
 * SpindleFB.st — PMSM spindle Function Block
 * IEC 61131-3 Structured Text
 *
 * Models the dq-axis dynamics of the DISCO dicing saw spindle motor.
 * Called every control cycle (dt_us = 10 µs).
 *)

FUNCTION_BLOCK SpindleFB
VAR_INPUT
    Enable      : BOOL;         (* TRUE = run spindle *)
    SpeedRef_rpm: REAL;         (* target speed [rpm] *)
    dt_us       : REAL := 10.0; (* timestep [µs] *)
END_VAR

VAR_OUTPUT
    Speed_rpm   : REAL;         (* actual speed [rpm] *)
    Torque_Nm   : REAL;         (* electromagnetic torque *)
    I_q         : REAL;         (* q-axis current [A] *)
    I_d         : REAL;         (* d-axis current [A] *)
    Fault       : BOOL;         (* overspeed / overcurrent *)
END_VAR

VAR
    (* Motor params — DISCO high-speed spindle *)
    Rs          : REAL := 0.5;      (* stator resistance [Ω] *)
    Ld          : REAL := 0.0015;   (* d-axis inductance [H] *)
    Lq          : REAL := 0.0015;   (* q-axis inductance [H] *)
    Psi_f       : REAL := 0.08;     (* flux linkage [Wb] *)
    Pn          : REAL := 4.0;      (* pole pairs *)
    J           : REAL := 0.0001;   (* rotor inertia [kg·m²] *)
    B_fric      : REAL := 0.01;     (* viscous friction [N·m·s] *)

    (* PI current controller *)
    Kp_i        : REAL := 10.0;
    Ki_i        : REAL := 50.0;
    Iq_ref      : REAL;
    Err_q       : REAL;
    Int_iq      : REAL := 0.0;
    V_d         : REAL;
    V_q         : REAL;

    (* State *)
    Omega_rad_s : REAL := 0.0;  (* mechanical angular velocity [rad/s] *)
    _I_d        : REAL := 0.0;
    _I_q        : REAL := 0.0;
    dt_s        : REAL;

    (* Limits *)
    MAX_SPEED_RPM  : REAL := 35000.0;
    MAX_CURRENT_A  : REAL := 25.0;
END_VAR

    IF NOT Enable THEN
        Omega_rad_s := Omega_rad_s * 0.99;  (* coast-down *)
        Speed_rpm   := Omega_rad_s * 60.0 / (2.0 * 3.14159);
        Fault       := FALSE;
        RETURN;
    END_IF

    dt_s := dt_us * 1.0E-6;

    (* Speed controller → Iq reference *)
    Iq_ref := (SpeedRef_rpm * 2.0 * 3.14159 / 60.0 - Omega_rad_s) * 5.0;
    Iq_ref := LIMIT(-20.0, Iq_ref, 20.0);

    (* PI current controller (q-axis) *)
    Err_q  := Iq_ref - _I_q;
    Int_iq := Int_iq + Err_q * dt_s;
    V_q    := Kp_i * Err_q + Ki_i * Int_iq
              + Pn * Omega_rad_s * (Ld * _I_d + Psi_f);   (* feed-forward *)
    V_d    := Kp_i * (0.0 - _I_d)
              - Pn * Omega_rad_s * Lq * _I_q;

    (* dq current integration *)
    _I_d := _I_d + (V_d - Rs * _I_d + Pn * Omega_rad_s * Lq * _I_q) / Ld * dt_s;
    _I_q := _I_q + (V_q - Rs * _I_q - Pn * Omega_rad_s * (Ld * _I_d + Psi_f)) / Lq * dt_s;

    (* Mechanical dynamics *)
    Torque_Nm   := 1.5 * Pn * (Psi_f * _I_q + (Ld - Lq) * _I_d * _I_q);
    Omega_rad_s := Omega_rad_s + (Torque_Nm - B_fric * Omega_rad_s) / J * dt_s;

    (* Outputs *)
    Speed_rpm := Omega_rad_s * 60.0 / (2.0 * 3.14159);
    I_q       := _I_q;
    I_d       := _I_d;

    (* Fault detection *)
    Fault := (ABS(Speed_rpm) > MAX_SPEED_RPM) OR (ABS(_I_q) > MAX_CURRENT_A);

END_FUNCTION_BLOCK
