try:
    from embedded._motor_ctrl_sm_kernel import MotorControlSM
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False
    MotorControlSM = None

__all__ = ["MotorControlSM", "_HAS_CPP"]
