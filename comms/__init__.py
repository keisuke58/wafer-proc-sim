try:
    from comms._modbus_rtu_kernel import (
        crc16_modbus, encode_read_holding, encode_write_single,
        parse_frame, ModbusRTUSlave,
    )
    _HAS_MODBUS = True
except ImportError:
    _HAS_MODBUS = False
    crc16_modbus = encode_read_holding = encode_write_single = None
    parse_frame = ModbusRTUSlave = None

try:
    from comms._ethercat_frame_kernel import (
        parse_ec_frame, build_ec_frame, cmd_name, EcSlave,
    )
    _HAS_ETHERCAT = True
except ImportError:
    _HAS_ETHERCAT = False
    parse_ec_frame = build_ec_frame = cmd_name = EcSlave = None

__all__ = [
    "crc16_modbus", "encode_read_holding", "encode_write_single",
    "parse_frame", "ModbusRTUSlave",
    "parse_ec_frame", "build_ec_frame", "cmd_name", "EcSlave",
    "_HAS_MODBUS", "_HAS_ETHERCAT",
]
