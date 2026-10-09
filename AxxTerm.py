# -*- coding: utf-8 -*-
import sys
import ast
import html
import math
import struct
import os
import json
import time
import ctypes
from collections import deque
from datetime import datetime

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtSerialPort import QSerialPort, QSerialPortInfo
from PyQt5.QtGui import QPixmap, QTextCursor, QIcon, QPainter, QColor
from PyQt5.QtWidgets import *
import numpy as np

# python-can (Kvaser/Ixxat/virtual backends) and pyqtgraph are imported on
# first use instead of at startup: cold imports cost ~4 s and ~2 s
# respectively, which made the window take several seconds to appear even in
# modes that never touch them. The module globals stay so existing
# `pycan.X` / `pg.X` call sites work unchanged once loaded.
pycan = None
pg = None


def _ensure_pycan():
    """Import python-can on first use. Raises ImportError if not installed."""
    global pycan
    if pycan is None:
        import can
        pycan = can
    return pycan


def _ensure_pg():
    """Import pyqtgraph on first use (plot or FFT view creation)."""
    global pg
    if pg is None:
        import pyqtgraph
        pg = pyqtgraph
    return pg


def _create_can_bus(interface, channel, bitrate):
    """Open a python-can bus. Runs on a worker thread: bus creation loads
    driver DLLs and talks to hardware, which can block for seconds."""
    kwargs = {}
    if interface == 'kvaser':
        # The Kvaser driver always installs virtual channels, and python-can
        # defaults accept_virtual=True - so with no hardware attached, Open
        # would "succeed" on a virtual channel and silently monitor nothing.
        # The app has its own explicit Virtual interface for that use case.
        kwargs['accept_virtual'] = False
    return _ensure_pycan().Bus(interface=interface, channel=channel,
                               bitrate=bitrate, **kwargs)

# --- Constants ---

DEFAULT_PLOT_LENGTH = 100
MAX_TEXT_LINES = 10000

PLOT_COLORS = [
    '#e6194b', '#3cb44b', '#0055d4', '#e67e00',
    '#911eb4', '#1a9bc7', '#f032e6', '#9A6324',
    '#800000', '#469990', '#7b68ee', '#000075',
]

# QSerialPort stop bit enum: OneStop=1, OneAndHalfStop=3, TwoStop=2
STOP_BIT_VALUES = [1, 3, 2]

# When frozen with PyInstaller, resolve paths next to the .exe, not the temp folder
if getattr(sys, 'frozen', False):
    SCRIPT_DIR = os.path.dirname(sys.executable)
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_MACRO_BUTTONS = 8
SETTINGS_FILE = os.path.join(SCRIPT_DIR, 'AxxTerm_settings.json')

DATA_TYPES = {
    'uint8':    ('B', 1),
    'int8':     ('b', 1),
    'uint16':   ('H', 2),
    'int16':    ('h', 2),
    'uint32':   ('I', 4),
    'int32':    ('i', 4),
    'float32':  ('f', 4),
    'double64': ('d', 8),
}

# numpy base dtype per data type, for vectorized (per-chunk) binary decoding
NUMPY_DTYPES = {
    'uint8':    np.uint8,
    'int8':     np.int8,
    'uint16':   np.uint16,
    'int16':    np.int16,
    'uint32':   np.uint32,
    'int32':    np.int32,
    'float32':  np.float32,
    'double64': np.float64,
}

# --- CAN mode constants ---

# Display name -> python-can interface (backend) name
CAN_INTERFACES = {
    'Kvaser': 'kvaser',
    'Ixxat': 'ixxat',
    'Virtual': 'virtual',
}
CAN_BITRATES = ['10', '20', '50', '100', '125', '250', '500', '800', '1000']  # kbit/s
CAN_MAX_SCROLL_ROWS = 10000  # scrolling trace ring-buffer cap


def j1939_pgn(can_id):
    """Extract the J1939 PGN from a 29-bit CAN ID.

    PDU1 (PF < 0xF0): the PS byte is a destination address, not part of the
    PGN, so it is cleared. PDU2 (PF >= 0xF0): PS is group extension and stays.
    Includes the Data Page / Extended Data Page bits.
    """
    pf = (can_id >> 16) & 0xFF
    if pf < 0xF0:
        return (can_id >> 8) & 0x3FF00
    return (can_id >> 8) & 0x3FFFF


def format_can_id(can_id, extended):
    """CAN ID as fixed-width uppercase hex (8 digits ext, 3 digits std)."""
    return f'{can_id:08X}' if extended else f'{can_id:03X}'


def format_can_log_line(msg):
    """One log-file line for a CAN message: ID, type, PGN, DLC, data, all hex."""
    can_id = getattr(msg, 'arbitration_id', 0)
    ext = bool(getattr(msg, 'is_extended_id', False))
    data = bytes(getattr(msg, 'data', b'') or b'')
    parts = ['CAN', format_can_id(can_id, ext), 'EXT' if ext else 'STD']
    if ext:
        parts.append(f'PGN {j1939_pgn(can_id):05X}')
    if getattr(msg, 'is_remote_frame', False):
        parts.append('RTR')
    if getattr(msg, 'is_error_frame', False):
        parts.append('ERR')
    parts.append(f'DLC {getattr(msg, "dlc", len(data))}')
    parts.append('DATA ' + (' '.join(f'{b:02X}' for b in data) if data else '-'))
    return ' '.join(parts)


# --- Kvaser controller state (direct canlib) --------------------------------
#
# python-can surfaces error *frames* but not the CAN controller's own state,
# and its Kvaser backend does not implement BusABC.state. Counting error
# frames is not the same thing: what matters is whether the controller has
# gone error-passive or, worse, BUS-OFF - at which point it has stopped
# taking part in traffic entirely and nothing else in the UI would say so.
#
# canReadStatus is a cheap, non-blocking local call against the handle the
# backend already opened, so the 1 Hz stats tick makes it directly. canlib is
# documented thread-safe, so sharing the handle with the reader thread's
# blocking recv() is fine. Any failure returns None and the UI shows nothing,
# never a guess.

# canstat.h status flags (not exported by python-can's constants module)
canSTAT_ERROR_PASSIVE = 0x00000001
canSTAT_BUS_OFF = 0x00000002
canSTAT_ERROR_WARNING = 0x00000004
canSTAT_ERROR_ACTIVE = 0x00000008

_canlib_dll = None  # ctypes handle, or False once a load attempt has failed


def _get_canlib():
    """Load canlib once. Returns the library, or None if it is unavailable."""
    global _canlib_dll
    if _canlib_dll is None:
        try:
            if os.name == 'nt':
                lib = ctypes.windll.LoadLibrary('canlib32')
            else:
                lib = ctypes.cdll.LoadLibrary('libcanlib.so')
            handle_t = ctypes.c_int
            dword_p = ctypes.POINTER(ctypes.c_ulong)
            lib.canRequestChipStatus.argtypes = [handle_t]
            lib.canRequestChipStatus.restype = ctypes.c_int
            lib.canReadStatus.argtypes = [handle_t, dword_p]
            lib.canReadStatus.restype = ctypes.c_int
            lib.canReadErrorCounters.argtypes = [handle_t, dword_p, dword_p, dword_p]
            lib.canReadErrorCounters.restype = ctypes.c_int
            _canlib_dll = lib
        except (OSError, AttributeError):
            # No Kvaser driver installed, or a build without these entry
            # points: the feature simply stays off.
            _canlib_dll = False
    return _canlib_dll or None


def kvaser_bus_state(bus):
    """Controller state of an open Kvaser bus as (state, tx_err, rx_err).

    Returns None for any other backend (python-can's virtual/Ixxat buses have
    no canlib handle), when canlib cannot be loaded, or when the driver call
    fails - the caller shows nothing in that case.
    """
    handle = getattr(bus, '_read_handle', None)
    if handle is None:
        return None
    lib = _get_canlib()
    if lib is None:
        return None
    flags = ctypes.c_ulong(0)
    tx = ctypes.c_ulong(0)
    rx = ctypes.c_ulong(0)
    ov = ctypes.c_ulong(0)
    try:
        # canReadStatus returns the last status the device reported;
        # canRequestChipStatus asks it for a fresh one first.
        lib.canRequestChipStatus(handle)
        if lib.canReadStatus(handle, ctypes.byref(flags)) != 0:
            return None
        lib.canReadErrorCounters(handle, ctypes.byref(tx), ctypes.byref(rx),
                                 ctypes.byref(ov))
    except Exception:
        return None
    return can_state_name(flags.value), tx.value, rx.value


def can_state_name(flags):
    """Controller state from canReadStatus flags, worst first."""
    if flags & canSTAT_BUS_OFF:
        return 'BUS-OFF'
    if flags & canSTAT_ERROR_PASSIVE:
        return 'PASSIVE'
    if flags & canSTAT_ERROR_WARNING:
        return 'WARNING'
    return 'ACTIVE'


# How each state is shown in the status bar. ACTIVE is the normal case and
# stays unstyled so the bar is not permanently shouting.
CAN_STATE_COLORS = {
    'BUS-OFF': '#c0453d',
    'PASSIVE': '#c0453d',
    'WARNING': '#d08b1e',
    'ACTIVE': '',
}


# --- Higher-layer protocol decoding -----------------------------------------
#
# Raw CAN carries no meaning: an ID is 11 or 29 bits and that is all the
# standard says. Everything below comes from a higher-layer protocol, and the
# same ID means different things under each - 0x7E8 is "OBD-II response from
# ECU 1" to one decoder and "CANopen heartbeat, node 104" to another. So the
# protocol is chosen in the UI rather than guessed from traffic; 'Auto' only
# applies the few rules that cannot collide (see decode_auto).
#
# Decoders take the already-unpacked frame fields and return a short label for
# the Decode column, or '' when the frame does not belong to that protocol.

# J1939-71/73 PGNs that show up on almost every vehicle bus.
J1939_PGN_NAMES = {
    0x0EA00: 'Request',
    0x0EB00: 'TP.DT',
    0x0EC00: 'TP.CM',
    0x0EE00: 'Address Claim',
    0x0EF00: 'Proprietary A',
    0x0F000: 'ERC1 Retarder',
    0x0F001: 'EBC1 Brakes',
    0x0F002: 'ETC1 Transmission',
    0x0F003: 'EEC2 Engine',
    0x0F004: 'EEC1 Engine',
    0x0F005: 'ETC2 Transmission',
    0x0FE6C: 'TCO1 Tachograph',
    0x0FEBF: 'EBC2 Wheel Speed',
    0x0FECA: 'DM1 Active DTCs',
    0x0FECB: 'DM2 Stored DTCs',
    0x0FECC: 'DM3 Clear DTCs',
    0x0FEE0: 'VD Vehicle Distance',
    0x0FEE4: 'SHUTDN Shutdown',
    0x0FEE5: 'HOURS Engine Hours',
    0x0FEE6: 'TD Time/Date',
    0x0FEE9: 'LFC Fuel Consumption',
    0x0FEEE: 'ET1 Engine Temp',
    0x0FEEF: 'EFL/P1 Fluid Level/Press',
    0x0FEF0: 'PTO',
    0x0FEF1: 'CCVS Speed/Cruise',
    0x0FEF2: 'LFE Fuel Economy',
    0x0FEF5: 'AMB Ambient',
    0x0FEF6: 'IC1 Inlet/Exhaust',
    0x0FEF7: 'VEP1 Electrical Power',
}

# TP.CM control bytes (first data byte of PGN 0xEC00)
J1939_TP_CM = {0x10: 'RTS', 0x11: 'CTS', 0x13: 'EndOfMsgAck',
               0x20: 'BAM', 0xFF: 'Abort'}


def decode_j1939(can_id, ext, data, rtr=False):
    """SAE J1939: PGN name, source address, destination and priority.

    J1939 is defined only over 29-bit IDs, so 11-bit frames get no label
    rather than a guess. NMEA 2000 shares the framing: its PGNs fall through
    to the numeric form.
    """
    if not ext:
        return ''
    prio = (can_id >> 26) & 0x7
    pf = (can_id >> 16) & 0xFF
    ps = (can_id >> 8) & 0xFF
    sa = can_id & 0xFF
    pgn = j1939_pgn(can_id)
    name = J1939_PGN_NAMES.get(pgn)
    if name is None:
        name = 'Proprietary B' if 0xFF00 <= pgn <= 0xFFFF else f'PGN {pgn:05X}'
    detail = ''
    if pgn == 0x0EA00 and len(data) >= 3:
        # A request carries the requested PGN as three bytes, LSB first
        req = data[0] | (data[1] << 8) | (data[2] << 16)
        req_name = J1939_PGN_NAMES.get(req)
        detail = f'  {req:05X}' + (f' {req_name.split()[0]}' if req_name else '')
    elif pgn == 0x0EC00 and data:
        detail = '  ' + J1939_TP_CM.get(data[0], f'ctrl {data[0]:02X}')
    elif pgn == 0x0EB00 and data:
        detail = f'  seq {data[0]}'
    # PDU1 (PF < 0xF0) is addressed: the PS byte is the destination address
    route = f'  SA {sa:02X} -> {ps:02X}' if pf < 0xF0 else f'  SA {sa:02X}'
    return f'{name}{detail}{route}  p{prio}'


# CANopen (CiA 301). Function code is bits 7..10, node-ID the low 7 bits.
CANOPEN_NMT_STATES = {0x00: 'Boot-up', 0x04: 'Stopped', 0x05: 'Operational',
                      0x7F: 'Pre-operational'}
# Phrased so the target reads on naturally: 'NMT: Reset node 3' / 'Reset all nodes'
CANOPEN_NMT_COMMANDS = {0x01: 'Start', 0x02: 'Stop', 0x80: 'Enter pre-op',
                        0x81: 'Reset', 0x82: 'Reset comm'}
CANOPEN_FUNCTIONS = {
    0x180: 'TPDO1', 0x200: 'RPDO1', 0x280: 'TPDO2', 0x300: 'RPDO2',
    0x380: 'TPDO3', 0x400: 'RPDO3', 0x480: 'TPDO4', 0x500: 'RPDO4',
    0x580: 'SDO tx', 0x600: 'SDO rx',
}


def decode_canopen(can_id, ext, data, rtr=False):
    """CANopen (CiA 301): the standardised 'heartbeat' lives here.

    Heartbeat is COB-ID 0x700 + node-ID with one data byte holding the NMT
    state - the only CAN heartbeat that is actually standardised. Node
    guarding reuses the same IDs as a remote frame.
    """
    if ext:
        return ''  # CiA 301 predefined connection set is 11-bit only
    node = can_id & 0x7F
    fc = can_id & 0x780
    if can_id == 0x000:
        if not data:
            return 'NMT'
        cmd = CANOPEN_NMT_COMMANDS.get(data[0], f'cmd {data[0]:02X}')
        if len(data) > 1:
            target = 'all nodes' if data[1] == 0 else f'node {data[1]}'
            return f'NMT: {cmd} {target}'
        return f'NMT: {cmd}'
    if can_id == 0x080:
        return 'SYNC'
    if can_id == 0x100:
        return 'TIME'
    if can_id in (0x7E4, 0x7E5):
        return 'LSS ' + ('slave' if can_id == 0x7E4 else 'master')
    if fc == 0x080 and node:
        if len(data) >= 2:
            return f'EMCY node {node}  code {data[0] | (data[1] << 8):04X}'
        return f'EMCY node {node}'
    if fc == 0x700 and node:
        if rtr:
            return f'Node guard request node {node}'
        if data:
            state = CANOPEN_NMT_STATES.get(data[0] & 0x7F, f'state {data[0]:02X}')
            return f'Heartbeat node {node}: {state}'
        return f'Heartbeat node {node}'
    name = CANOPEN_FUNCTIONS.get(fc)
    if name and node:
        if fc in (0x580, 0x600) and len(data) >= 4:
            # SDO: bytes 1-2 are the object index (LSB first), byte 3 the sub-index
            return f'{name} node {node}  {data[1] | (data[2] << 8):04X}:{data[3]:02X}'
        return f'{name} node {node}'
    return ''


# ISO-TP (ISO 15765-2) protocol control information, high nibble of byte 0
ISOTP_FLOW_STATUS = {0: 'CTS', 1: 'WAIT', 2: 'OVFLW'}
# OBD-II modes (ISO 15031-5) and UDS services (ISO 14229-1) share the SID byte
UDS_SERVICES = {
    0x01: 'OBD current data', 0x02: 'OBD freeze frame', 0x03: 'OBD show DTCs',
    0x04: 'OBD clear DTCs', 0x06: 'OBD test results', 0x07: 'OBD pending DTCs',
    0x09: 'OBD vehicle info', 0x0A: 'OBD permanent DTCs',
    0x10: 'DiagnosticSessionControl', 0x11: 'ECUReset',
    0x14: 'ClearDiagnosticInformation', 0x19: 'ReadDTCInformation',
    0x22: 'ReadDataByIdentifier', 0x23: 'ReadMemoryByAddress',
    0x27: 'SecurityAccess', 0x28: 'CommunicationControl',
    0x2E: 'WriteDataByIdentifier', 0x2F: 'InputOutputControl',
    0x31: 'RoutineControl', 0x34: 'RequestDownload', 0x35: 'RequestUpload',
    0x36: 'TransferData', 0x37: 'RequestTransferExit', 0x3E: 'TesterPresent',
    0x85: 'ControlDTCSetting',
}
UDS_NRC = {
    0x11: 'service not supported', 0x12: 'sub-function not supported',
    0x13: 'wrong length', 0x22: 'conditions not correct',
    0x31: 'request out of range', 0x33: 'security access denied',
    0x35: 'invalid key', 0x78: 'response pending',
    0x7E: 'service not supported in session',
}


def _obd_address(can_id, ext):
    """Who is talking, from the ID alone, or None if these are not OBD IDs."""
    if ext:
        # ISO 15765-4 29-bit addressing: 18DB33F1 functional, 18DAttss physical
        pf = (can_id >> 16) & 0xFF
        if pf not in (0xDA, 0xDB):
            return None
        dst = (can_id >> 8) & 0xFF
        src = can_id & 0xFF
        if pf == 0xDB:
            return 'Functional req'
        if src == 0xF1:
            return f'Req -> ECU {dst:02X}'
        if dst == 0xF1:
            return f'Resp ECU {src:02X}'
        return f'{src:02X} -> {dst:02X}'
    if can_id == 0x7DF:
        return 'Functional req (all ECUs)'
    if 0x7E0 <= can_id <= 0x7E7:
        return f'Req ECU{can_id - 0x7E0 + 1}'
    if 0x7E8 <= can_id <= 0x7EF:
        return f'Resp ECU{can_id - 0x7E8 + 1}'
    return None


def decode_obd2(can_id, ext, data, rtr=False):
    """OBD-II / UDS over ISO-TP: addressing, the PCI nibble and the service."""
    who = _obd_address(can_id, ext)
    if who is None:
        return ''
    if not data:
        return who
    pci = data[0] >> 4
    if pci == 2:
        return f'{who}  CF seq {data[0] & 0x0F}'
    if pci == 3:
        return f'{who}  FC {ISOTP_FLOW_STATUS.get(data[0] & 0x0F, "?")}'
    if pci == 0:
        sid_at, tail = 1, f'SF({data[0] & 0x0F})'
    elif pci == 1 and len(data) >= 2:
        sid_at = 2
        tail = f'FF(len {((data[0] & 0x0F) << 8) | data[1]})'
    else:
        return who
    if len(data) <= sid_at:
        return f'{who}  {tail}'
    sid = data[sid_at]
    if sid == 0x7F:  # negative response: echoed service + reason code
        svc = (UDS_SERVICES.get(data[sid_at + 1], f'{data[sid_at + 1]:02X}')
               if len(data) > sid_at + 1 else '?')
        nrc = (UDS_NRC.get(data[sid_at + 2], f'NRC {data[sid_at + 2]:02X}')
               if len(data) > sid_at + 2 else '')
        return f'{who}  {tail}  NegResp {svc}: {nrc}'.rstrip(': ')
    positive = sid >= 0x40 and (sid - 0x40) in UDS_SERVICES
    base = sid - 0x40 if positive else sid
    out = f'{who}  {tail}  ' + UDS_SERVICES.get(base, f'SID {sid:02X}')
    if positive:
        out += ' +'
    if base in (0x01, 0x02) and len(data) > sid_at + 1:
        out += f'  PID {data[sid_at + 1]:02X}'
    elif base == 0x22 and len(data) > sid_at + 2:
        out += f'  DID {data[sid_at + 1]:02X}{data[sid_at + 2]:02X}'
    return out


def decode_auto(can_id, ext, data, rtr=False):
    """Apply only the rules that cannot collide between protocols.

    29-bit: the OBD-II 18DA/18DB range first, then J1939 for everything else
    (J1939 owns 29-bit IDs by convention). 11-bit: the OBD-II 7DF/7E0-7EF
    range first, then CANopen, whose predefined IDs do not reach into it.
    Anything genuinely ambiguous is better resolved by picking the protocol
    explicitly in the combo.
    """
    return (decode_obd2(can_id, ext, data, rtr)
            or (decode_j1939 if ext else decode_canopen)(can_id, ext, data, rtr))


CAN_DECODE_MODES = ['Off', 'Auto', 'J1939', 'CANopen', 'OBD-II']
CAN_DECODERS = {
    'Auto': decode_auto,
    'J1939': decode_j1939,
    'CANopen': decode_canopen,
    'OBD-II': decode_obd2,
}


# Macro buttons are per-mode: the serial set and the CAN set are stored and
# restored independently, so switching Serial <-> CAN swaps the whole row of
# buttons instead of showing serial byte strings with no CAN ID attached.
DEFAULT_MACROS = [
    {"label": "0x7F",           "hex": "7F"},
    {"label": "FF",             "hex": "FF"},
    {"label": "FF",             "hex": "FF"},
    {"label": "0xBB",           "hex": "BB"},
    {"label": "__SHORTPRESS__", "hex": "5f5f53484f525450524553535f5f0a"},
    {"label": "__LONGPRESS__",  "hex": "5f5f4c4f4e4750524553535f5f0a"},
    {"label": "$$$",            "hex": "242424"},
    {"label": "__OTA__",        "hex": "5F5F4F54415F5F0A"},
]

DEFAULT_CAN_MACROS = [
    {"label": "100 zeros", "hex": "0000000000000000", "can_id": "100",      "can_ext": False},
    {"label": "100 FFs",   "hex": "FFFFFFFFFFFFFFFF", "can_id": "100",      "can_ext": False},
    {"label": "1 byte FF", "hex": "FF",               "can_id": "100",      "can_ext": False},
    # J1939 request (PGN 0xEA00, global destination) for EEC1: the data is
    # the requested PGN 0x00F004 as three bytes, least significant first.
    {"label": "Req EEC1",  "hex": "04F000",           "can_id": "18EAFFFE", "can_ext": True},
    {"label": "Macro 5",   "hex": "",                 "can_id": "",         "can_ext": False},
    {"label": "Macro 6",   "hex": "",                 "can_id": "",         "can_ext": False},
    {"label": "Macro 7",   "hex": "",                 "can_id": "",         "can_ext": False},
    {"label": "Macro 8",   "hex": "",                 "can_id": "",         "can_ext": False},
]

# --- Application stylesheet (light theme) ---
# One coherent design: neutral grays/whites with a single muted-blue accent
# (#3574b3) reserved for primary actions (Open / Send). QComboBox is styled
# with explicit ::drop-down sub-control; QSpinBox is intentionally left
# native so its up/down buttons keep rendering correctly on Windows.
ACCENT = '#3574b3'

# Connection-status colors for the DB-9 connector indicator (and window icon).
# Muted green/red picked to sit next to the neutral gray theme rather than
# the former traffic-light #22bb22 / #cc2222.
CONNECTED_COLOR = '#3d9950'
DISCONNECTED_COLOR = '#c0453d'

LIGHT_QSS = """
QToolTip {
    color: #24292e; background-color: #f9fafb;
    border: 1px solid #b7bcc2; padding: 3px 6px;
}

QToolBar {
    background: #f4f5f6;
    border-bottom: 1px solid #d5d8db;
    spacing: 4px;
    padding: 3px 4px;
}

QMenuBar { background: #f4f5f6; }
QMenuBar::item { background: transparent; padding: 4px 10px; }
QMenuBar::item:selected { background: #e2e8ee; border-radius: 3px; }
QMenu { background: #ffffff; border: 1px solid #c3c8cd; }
QMenu::item { padding: 4px 24px; }
QMenu::item:selected { background: #dce8f5; color: #24292e; }

QStatusBar { background: #f4f5f6; border-top: 1px solid #d5d8db; }
QStatusBar::item { border: none; }

/* Secondary buttons: neutral, subtle border, clear hover/pressed states */
QPushButton {
    background-color: #fbfbfc;
    border: 1px solid #c3c8cd;
    border-radius: 3px;
    padding: 3px 12px;
    color: #24292e;
}
QPushButton:hover { background-color: #f0f4f8; border-color: #a8b3bd; }
QPushButton:pressed { background-color: #e2e8ee; }
QPushButton:checked { background-color: #e2e8ee; border-color: #a8b3bd; }
QPushButton:focus { border-color: #3574b3; }
QPushButton:disabled {
    color: #9aa0a6; background-color: #f4f5f6; border-color: #dcdfe2;
}

/* Primary actions (Open / Send): the one accent color, used sparingly */
QPushButton#primaryButton {
    background-color: #3574b3;
    border: 1px solid #2c619b;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#primaryButton:hover { background-color: #3f81c4; border-color: #2c619b; }
QPushButton#primaryButton:pressed { background-color: #2a5c92; }
QPushButton#primaryButton:checked { background-color: #2a5c92; border-color: #234e7c; }
QPushButton#primaryButton:focus { border-color: #1d4570; }
QPushButton#primaryButton:disabled {
    background-color: #9db8d2; border-color: #8fa9c2; color: #f0f4f8;
}

/* Record button lives in the status bar: keep it compact. While recording
   (button is checkable) it turns the same muted red as the disconnected
   connector icon, instead of an ad-hoc inline style. No font-weight change:
   Qt sizes the button from its QFont, so QSS-only bold renders wider than
   the computed width and clips the text. */
QPushButton#recordButton { padding: 1px 12px; }
QPushButton#recordButton:checked {
    background-color: #c0453d;
    border-color: #a93b34;
    color: #ffffff;
}
QPushButton#recordButton:checked:hover { background-color: #cb524a; }
QPushButton#recordButton:checked:focus { border-color: #8f312b; }

QComboBox {
    background-color: #ffffff;
    border: 1px solid #c3c8cd;
    border-radius: 3px;
    padding: 2px 6px 2px 8px;
    color: #24292e;
}
QComboBox:hover { border-color: #a8b3bd; }
QComboBox:focus { border-color: #3574b3; }
QComboBox:disabled { color: #9aa0a6; background-color: #f4f5f6; }
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 20px;
    border: none;
}
QComboBox::down-arrow {
    image: url(%COMBO_ARROW%);
    width: 10px;
    height: 10px;
}
QComboBox QAbstractItemView {
    background: #ffffff;
    border: 1px solid #c3c8cd;
    selection-background-color: #dce8f5;
    selection-color: #24292e;
    outline: none;
}

QLineEdit, QTextEdit {
    background-color: #ffffff;
    border: 1px solid #c9cdd2;
    border-radius: 3px;
    selection-background-color: #b8d4f0;
    selection-color: #1a1d20;
}
QLineEdit:focus, QTextEdit:focus { border: 1px solid #3574b3; }
QLineEdit:disabled, QTextEdit:disabled { background-color: #f4f5f6; color: #9aa0a6; }

/* Section headers: "Data: ASCII", "Data: HEX", "Converter" */
QLabel#sectionLabel { color: #5a6570; font-weight: 600; }
"""

# Dark-mode counterpart. Note: before this stylesheet existed, the native
# Windows style ignored the dark palette for buttons/combos (white boxes with
# white text); these rules make dark mode actually readable.
DARK_QSS = """
QToolTip { color: #ffffff; background-color: #2a2a2a; border: 1px solid #666666; padding: 3px 6px; }

QToolBar { background: #2f2f2f; border-bottom: 1px solid #1f1f1f; spacing: 4px; padding: 3px 4px; }
QStatusBar { background: #2f2f2f; border-top: 1px solid #1f1f1f; }
QStatusBar::item { border: none; }

QMenuBar { background: #2f2f2f; color: #e8e8e8; }
QMenuBar::item { background: transparent; padding: 4px 10px; }
QMenuBar::item:selected { background: #44484b; border-radius: 3px; }
QMenu { background: #2b2b2b; color: #e8e8e8; border: 1px solid #555555; }
QMenu::item { padding: 4px 24px; }
QMenu::item:selected { background: #3574b3; color: #ffffff; }

QPushButton {
    background-color: #3c3f41;
    border: 1px solid #5a5d5f;
    border-radius: 3px;
    padding: 3px 12px;
    color: #e8e8e8;
}
QPushButton:hover { background-color: #46494b; border-color: #6f7375; }
QPushButton:pressed { background-color: #2f3234; }
QPushButton:checked { background-color: #2f3234; border-color: #6f7375; }
QPushButton:focus { border-color: #2a82da; }
QPushButton:disabled { color: #7d8184; background-color: #333537; border-color: #4a4d4f; }

QPushButton#primaryButton {
    background-color: #3574b3;
    border: 1px solid #2c619b;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#primaryButton:hover { background-color: #3f81c4; border-color: #2c619b; }
QPushButton#primaryButton:pressed { background-color: #2a5c92; }
QPushButton#primaryButton:checked { background-color: #2a5c92; border-color: #234e7c; }
QPushButton#primaryButton:disabled {
    background-color: #3e4c5c; border-color: #37424f; color: #8d99a5;
}

QPushButton#recordButton { padding: 1px 12px; }
QPushButton#recordButton:checked {
    background-color: #c0453d;
    border-color: #a93b34;
    color: #ffffff;
}
QPushButton#recordButton:checked:hover { background-color: #cb524a; }
QPushButton#recordButton:checked:focus { border-color: #8f312b; }

QComboBox {
    background-color: #2f3234;
    border: 1px solid #5a5d5f;
    border-radius: 3px;
    padding: 2px 6px 2px 8px;
    color: #e8e8e8;
}
QComboBox:hover { border-color: #6f7375; }
QComboBox:focus { border-color: #2a82da; }
QComboBox:disabled { color: #7d8184; background-color: #333537; }
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 20px;
    border: none;
}
QComboBox::down-arrow {
    image: url(%COMBO_ARROW%);
    width: 10px;
    height: 10px;
}
QComboBox QAbstractItemView {
    background: #2b2b2b;
    color: #e8e8e8;
    border: 1px solid #555555;
    selection-background-color: #3574b3;
    selection-color: #ffffff;
    outline: none;
}

QLineEdit, QTextEdit {
    background-color: #232323;
    border: 1px solid #4a4d4f;
    border-radius: 3px;
    color: #e8e8e8;
    selection-background-color: #2a5c92;
    selection-color: #ffffff;
}
QLineEdit:focus, QTextEdit:focus { border: 1px solid #2a82da; }
QLineEdit:disabled, QTextEdit:disabled { background-color: #2c2e30; color: #7d8184; }

/* Spin boxes: the native style paints them white-on-white in dark mode */
QSpinBox {
    background-color: #232323;
    border: 1px solid #4a4d4f;
    border-radius: 3px;
    color: #e8e8e8;
}
QSpinBox:focus { border-color: #2a82da; }
QSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 16px;
    background: #3c3f41;
    border-left: 1px solid #4a4d4f;
    border-bottom: 1px solid #4a4d4f;
    border-top-right-radius: 3px;
}
QSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 16px;
    background: #3c3f41;
    border-left: 1px solid #4a4d4f;
    border-bottom-right-radius: 3px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover { background: #46494b; }
QSpinBox::up-button:pressed, QSpinBox::down-button:pressed { background: #2f3234; }
QSpinBox::up-arrow { image: url(%ARROW_UP%); width: 10px; height: 10px; }
QSpinBox::down-arrow { image: url(%ARROW_DOWN%); width: 10px; height: 10px; }

QLabel#sectionLabel { color: #9aa5b0; font-weight: 600; }

/* Table headers: the native header stays white in dark mode, which made the
   CAN column titles white-on-white (invisible) */
QHeaderView::section {
    background-color: #2f3234;
    color: #e8e8e8;
    border: none;
    border-right: 1px solid #4a4d4f;
    border-bottom: 1px solid #4a4d4f;
    padding: 2px 6px;
}
QTableView {
    background-color: #232323;
    gridline-color: #4a4d4f;
    border: 1px solid #4a4d4f;
}
QTableCornerButton::section { background-color: #2f3234; border: none; }
"""

CONVERTERS = {
    'HEX --> ASCII': lambda v: bytes.fromhex(v).decode('ISO-8859-1'),
    'HEX --> DECIMAL': lambda v: str(int(v.replace(' ', ''), 16)),
    'HEX --> BINARY': lambda v: format(int(v.replace(' ', ''), 16),
                                       '0{}b'.format(len(v.replace(' ', '')) * 4)),
    'ASCII --> HEX': lambda v: '0x' + v.encode('ISO-8859-1').hex(),
    'ASCII --> DECIMAL': lambda v: ' '.join(str(b) for b in v.encode('ISO-8859-1')),
    'ASCII --> BINARY': lambda v: ' '.join(format(b, '08b') for b in v.encode('ISO-8859-1')),
    'DECIMAL --> HEX': lambda v: hex(int(v)),
    'DECIMAL --> ASCII': lambda v: chr(int(v)),
    'DECIMAL --> BINARY': lambda v: format(int(v), '08b'),
    'BINARY --> HEX': lambda v: hex(int(v.replace(' ', ''), 2)),
    'BINARY --> ASCII': lambda v: chr(int(v.replace(' ', ''), 2)),
    'BINARY --> DECIMAL': lambda v: str(int(v.replace(' ', ''), 2)),
}


class BinaryStreamReader:
    """Decodes a continuous binary byte stream into channel values."""

    def __init__(self):
        self.buffer = bytearray()
        self.data_type = 'float32'
        self.endianness = 'little'
        self.num_channels = 4

    def feed(self, data):
        """Feed raw bytes. Returns list of tuples, one per sample."""
        self.buffer.extend(data)
        fmt_char, type_size = DATA_TYPES[self.data_type]
        package_size = self.num_channels * type_size
        if package_size == 0:
            return []
        prefix = '<' if self.endianness == 'little' else '>'
        fmt = prefix + fmt_char * self.num_channels
        n_samples = len(self.buffer) // package_size
        if n_samples == 0:
            return []
        chunk = bytes(self.buffer[:n_samples * package_size])
        del self.buffer[:n_samples * package_size]
        return list(struct.iter_unpack(fmt, chunk))

    def feed_np(self, data):
        """Feed raw bytes; return a complete-samples numpy array shaped
        (n_samples, num_channels) as float64, or None if no full sample yet.

        Fully vectorized: np.frombuffer over the whole chunk, no per-sample
        Python. This is the hot path for high-rate binary streams.
        """
        self.buffer.extend(data)
        if self.num_channels == 0:
            return None
        base = NUMPY_DTYPES[self.data_type]
        type_size = np.dtype(base).itemsize
        package_size = self.num_channels * type_size
        n_samples = len(self.buffer) // package_size
        if n_samples == 0:
            return None
        nbytes = n_samples * package_size
        raw = bytes(self.buffer[:nbytes])
        del self.buffer[:nbytes]
        order = '<' if self.endianness == 'little' else '>'
        dt = np.dtype(base).newbyteorder(order)
        arr = np.frombuffer(raw, dtype=dt, count=n_samples * self.num_channels)
        # Garbage/uninitialized bytes can decode to inf/NaN float32; the cast is
        # still correct, so silence the cosmetic warning (flush_plot turns any
        # non-finite value into a plot gap).
        with np.errstate(invalid='ignore'):
            return arr.astype(np.float64).reshape(n_samples, self.num_channels)

    def sync(self):
        """Clear buffer to re-align stream."""
        self.buffer.clear()


class FrameReader:
    """Decodes framed binary packets with sync word, optional size, and optional checksum.

    Buffer-based: scans for the sync word with bytes.find (handles sync words
    with internal prefix repetition), and re-scans from the byte after a false
    sync when a size check or checksum fails, so one corrupt byte never
    swallows subsequent valid frames.
    """

    MAX_BUFFER = 1 << 20  # 1 MB cap against garbage input with no sync words

    def __init__(self):
        self.data_type = 'float32'
        self.endianness = 'little'
        self.num_channels = 4
        self.sync_word = bytes([0xAA])
        self.size_field = 'fixed'
        self.frame_size = 12
        self.checksum_enabled = False
        self.buffer = bytearray()

    def reset(self):
        """Clear buffered bytes."""
        self.buffer.clear()

    def feed(self, data):
        """Feed raw bytes. Returns list of tuples, one per sample."""
        self.buffer.extend(data)
        results = []
        fmt_char, type_size = DATA_TYPES[self.data_type]
        prefix = '<' if self.endianness == 'little' else '>'
        sample_size = self.num_channels * type_size
        if sample_size == 0:
            return []
        fmt = prefix + fmt_char * self.num_channels
        sync = self.sync_word
        size_len = {'fixed': 0, '1-byte': 1, '2-byte': 2}.get(self.size_field, 0)
        checksum_len = 1 if self.checksum_enabled else 0

        pos = 0
        buf = self.buffer
        while True:
            start = buf.find(sync, pos)
            if start < 0:
                # Keep a potential partial sync word at the tail
                pos = max(pos, len(buf) - (len(sync) - 1))
                break
            header_end = start + len(sync) + size_len
            if len(buf) < header_end:
                pos = start
                break  # wait for size field bytes
            if size_len == 0:
                payload_size = self.frame_size
            elif size_len == 1:
                payload_size = buf[start + len(sync)]
            else:
                size_fmt = '<H' if self.endianness == 'little' else '>H'
                payload_size = struct.unpack_from(size_fmt, buf, start + len(sync))[0]
            if payload_size == 0 or (size_len > 0 and payload_size % sample_size != 0):
                pos = start + 1  # false sync: re-scan from the next byte
                continue
            frame_end = header_end + payload_size + checksum_len
            if len(buf) < frame_end:
                pos = start
                break  # wait for full frame
            payload = bytes(buf[header_end:header_end + payload_size])
            if checksum_len and buf[header_end + payload_size] != (sum(payload) & 0xFF):
                pos = start + 1  # bad checksum: re-scan from the next byte
                continue
            offset = 0
            while offset + sample_size <= len(payload):
                results.append(struct.unpack_from(fmt, payload, offset))
                offset += sample_size
            pos = frame_end

        if pos > 0:
            del buf[:pos]
        if len(buf) > self.MAX_BUFFER:
            del buf[:len(buf) - self.MAX_BUFFER]
        return results


_ARROW_ICON_PATHS = {}


def arrow_icon_url(color='#5a6570', direction='down'):
    """Render a small arrow PNG once per (color, direction) -- QSS needs an
    image once QComboBox/QSpinBox are styled -- and return its path in QSS
    url() form (forward slashes)."""
    key = (color, direction)
    if key not in _ARROW_ICON_PATHS:
        import tempfile
        pixmap = QPixmap(10, 10)
        pixmap.fill(QtCore.Qt.transparent)
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QColor(color))
        if direction == 'down':
            points = [(2.0, 3.5), (8.0, 3.5), (5.0, 7.0)]
        else:
            points = [(2.0, 6.5), (8.0, 6.5), (5.0, 3.0)]
        p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in points]))
        p.end()
        path = os.path.join(
            tempfile.gettempdir(),
            f'axxterm_arrow_{direction}_{color.lstrip("#")}.png')
        pixmap.save(path, 'PNG')
        _ARROW_ICON_PATHS[key] = path.replace('\\', '/')
    return _ARROW_ICON_PATHS[key]


def create_connector_pixmap(color, width=71, height=30):
    """Draw a DB-9 connector icon programmatically (no external PNG needed).

    Flat, antialiased rendering in the app's neutral grays; only the D-shell
    face carries the status color (green = connected, red = disconnected).
    """
    pixmap = QPixmap(width, height)
    pixmap.fill(QtCore.Qt.transparent)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.Antialiasing)

    cy = height / 2.0
    status = QColor(color)
    status_edge = status.darker(120)

    # 1. Outer rounded rectangle (metal shell): soft gray, subtle border
    p.setPen(QtGui.QPen(QColor('#a8b3bd'), 1.0))
    p.setBrush(QColor('#eef0f2'))
    p.drawRoundedRect(QtCore.QRectF(0.5, 0.5, width - 1.0, height - 1.0), 5, 5)

    # 2. Inner D-shaped colored area (trapezoid: wider at top, narrower at bottom)
    d_left = 15.0
    d_right = width - 15.0
    d_top = 4.5
    d_bot = height - 4.5
    taper = 1.5
    cr = 3.0
    d_path = QtGui.QPainterPath()
    d_path.moveTo(d_left + cr, d_top)
    d_path.lineTo(d_right - cr, d_top)
    d_path.quadTo(d_right, d_top, d_right, d_top + cr)
    d_path.lineTo(d_right - taper, d_bot - cr)
    d_path.quadTo(d_right - taper, d_bot, d_right - taper - cr, d_bot)
    d_path.lineTo(d_left + taper + cr, d_bot)
    d_path.quadTo(d_left + taper, d_bot, d_left + taper, d_bot - cr)
    d_path.lineTo(d_left, d_top + cr)
    d_path.quadTo(d_left, d_top, d_left + cr, d_top)
    d_path.closeSubpath()
    p.setPen(QtGui.QPen(status_edge, 1.0))
    p.setBrush(status)
    p.drawPath(d_path)

    # 3. Mounting screws: small flat circles (no cross-head clutter)
    screw_r = 3.5
    screw_lx = 8.0
    screw_rx = width - 8.0
    p.setPen(QtGui.QPen(QColor('#a8b3bd'), 1.0))
    p.setBrush(QColor('#dcdfe2'))
    p.drawEllipse(QtCore.QPointF(screw_lx, cy), screw_r, screw_r)
    p.drawEllipse(QtCore.QPointF(screw_rx, cy), screw_r, screw_r)

    # 4. Pin holes: 5 top row, 4 bottom row, soft white on the status color
    d_cx = (d_left + d_right) / 2.0
    pin_r = 1.5
    pin_spacing = 7.0
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(QColor(255, 255, 255, 225))
    for i in range(5):
        p.drawEllipse(QtCore.QPointF(d_cx + (i - 2) * pin_spacing, cy - 3.5), pin_r, pin_r)
    for i in range(4):
        p.drawEllipse(QtCore.QPointF(d_cx + (i - 1.5) * pin_spacing, cy + 3.5), pin_r, pin_r)

    p.end()
    return pixmap


class CanReaderThread(QtCore.QThread):
    """Blocking-recv reader for a python-can bus.

    Appends ('RX', msg) tuples to a deque (thread-safe append) that the GUI
    drains on its ~30 fps display timer -- same batching pattern as the serial
    RX path, so a saturated bus never floods the event loop.
    """

    errorOccurred = QtCore.pyqtSignal(str)

    def __init__(self, bus, queue, parent=None):
        super().__init__(parent)
        self._bus = bus
        self._queue = queue
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        while not self._stop:
            try:
                msg = self._bus.recv(timeout=0.2)
            except Exception as e:
                # Backend exceptions vary per driver (Kvaser/Ixxat DLLs);
                # anything raised here means the bus is gone.
                if not self._stop:
                    self.errorOccurred.emit(str(e))
                return
            if msg is not None:
                self._queue.append(('RX', msg))


# Threads abandoned to a hung driver call at window close. Detaching them from
# the window (plus keeping a reference here) prevents Qt from destroying a
# running QThread during window teardown, which aborts the process.
_ORPHANED_THREADS = []


def _orphan_thread(t):
    try:
        t.setParent(None)
    except RuntimeError:
        return  # C++ object already gone
    _ORPHANED_THREADS.append(t)


class CanBusOpener(QtCore.QThread):
    """Opens a python-can bus off the GUI thread.

    Bus creation imports python-can (~4 s cold), loads driver DLLs and talks
    to hardware (Kvaser/Ixxat), any of which can block for seconds. Joins the
    given closer threads first so reopening a channel never races its own
    shutdown.
    """

    succeeded = QtCore.pyqtSignal(object)  # the opened bus
    failed = QtCore.pyqtSignal(str)

    def __init__(self, interface, channel, bitrate, wait_for=(), parent=None):
        super().__init__(parent)
        self._interface = interface
        self._channel = channel
        self._bitrate = bitrate
        self._wait_for = list(wait_for)
        self.bus = None  # read directly at app exit, when the queued
        #                  succeeded signal can no longer be delivered

    def run(self):
        for closer in self._wait_for:
            if not closer.wait(8000):
                # A hung driver shutdown would otherwise wedge this open
                # forever in a silent 'Opening CAN...' state.
                self.failed.emit('previous CAN close has not finished - '
                                 'driver may be hung; try again')
                return
        try:
            bus = _create_can_bus(self._interface, self._channel, self._bitrate)
        except ImportError:
            self.failed.emit(
                'python-can is not installed - run: pip install python-can')
            return
        except Exception as e:
            # Backend exceptions vary per driver (missing DLL, no device,
            # channel in use); report the message instead of crashing.
            self.failed.emit(str(e))
            return
        self.bus = bus
        self.succeeded.emit(bus)


class CanBusCloser(QtCore.QThread):
    """Stops a reader thread and shuts a bus down off the GUI thread, so
    closing (or switching UI mode) never stalls the UI on driver teardown."""

    def __init__(self, reader, bus, parent=None):
        super().__init__(parent)
        self.reader = reader
        self._bus = bus

    def run(self):
        if self.reader is not None:
            self.reader.stop()
            self.reader.wait(1000)
        if self._bus is not None:
            try:
                self._bus.shutdown()
            except Exception:
                pass
        # shutdown() often unblocks a reader stuck in recv(); one last chance
        # to join it so it isn't left running when the window is destroyed
        if self.reader is not None:
            self.reader.wait(2000)


class CanFrameModel(QtCore.QAbstractTableModel):
    """Table model for CAN traffic with two display modes (like CANKing).

    Scrolling: every frame appends a row (ring-buffered at
    CAN_MAX_SCROLL_ROWS). Fixed: one row per (direction, ID); the row is
    overwritten in place and Count increments.

    Rows are stored as pre-formatted string tuples so data() is a plain list
    lookup -- no per-cell formatting while the view repaints under load.
    Delta-time and count bookkeeping is shared between modes, so switching
    view mode never loses the per-ID timing state.

    In fixed mode a refreshed row is nothing but changed text, which is easy
    to miss on a busy bus - so the Time cell of an updated row is tinted and
    the tint fades out over FLASH_MS. Only that one column lights up: a whole
    row blinking across the table is distracting to read against, while a
    column of blips at the left edge scans like an activity strip. The fade
    is derived in data() from a per-row monotonic stamp (no timer state per
    row) and repainted by CanView's fade timer.
    """

    COLUMNS = ['Time [s]', 'Δt [s]', 'Count', 'Dir', 'Type', 'ID [hex]',
               'PGN', 'DLC', 'Data [hex]', 'Decode']
    _DECODE_COLUMN = 9
    _RIGHT_ALIGNED = {0, 1, 2}      # Time, dt, Count
    # DLC is centred rather than right-aligned: hard against the right edge
    # it sat one pixel from the first data byte and read as part of it.
    _CENTERED = {3, 4, 7}           # Dir, Type, DLC

    TX_COLOR = QtGui.QColor('#2a82da')  # readable on light and dark themes

    FLASH_MS = 700.0        # fade-out duration of the "row just updated" tint
    _FLASH_COLUMN = 0       # Time [s]: the only cell that lights up
    _FLASH_STEPS = 12       # alpha is quantised to this many levels and cached
    # Base tints (alpha is applied per step). RX reads as the accent blue, TX
    # as a warmer tone so a frame we sent ourselves stays distinguishable.
    _FLASH_RGB_LIGHT = {False: (53, 116, 179), True: (196, 120, 40)}
    _FLASH_RGB_DARK = {False: (90, 160, 230), True: (230, 160, 70)}
    # Peak alpha. Higher than a full-width band would need: one 90 px cell
    # has to carry the signal on its own.
    _FLASH_PEAK_LIGHT = 165
    _FLASH_PEAK_DARK = 175

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []           # list of (9 display strings,) tuples
        self._row_is_tx = []     # parallel: True for TX rows (foreground color)
        self.fixed_mode = False
        self._fixed_index = {}   # {key: row} in fixed mode
        self._last_ts = {}       # {key: last timestamp} for delta-time
        self._counts = {}        # {key: frames seen}
        self._t0 = None          # timestamp of the first frame (relative time base)
        # Row-update flash (fixed mode only)
        self.flash_enabled = True
        self._row_flash = []     # parallel: monotonic ms when the row last changed
        self._flash_now = 0.0    # cached clock for one repaint pass
        self._flash_active = False
        self._dark_mode = False
        self._flash_cache = {}   # {(is_tx, step): QColor} -- no QColor per cell
        # Higher-layer decode. _row_src keeps the few numbers each decoder
        # needs, so switching protocol re-labels the existing trace in place
        # instead of throwing the capture away.
        self.decode_mode = 'Off'
        self._decoder = None
        self._row_src = []       # parallel: (can_id, ext, data, rtr)

    # --- Qt model interface ---

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.COLUMNS)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            return self.rows[index.row()][index.column()]
        if role == QtCore.Qt.ForegroundRole and self._row_is_tx[index.row()]:
            return self.TX_COLOR
        if (role == QtCore.Qt.BackgroundRole and self._flash_active
                and index.column() == self._FLASH_COLUMN):
            return self._flash_brush(index.row())
        if role == QtCore.Qt.TextAlignmentRole:
            col = index.column()
            if col in self._RIGHT_ALIGNED:
                return QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
            if col in self._CENTERED:
                return QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter
            return QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole and orientation == QtCore.Qt.Horizontal:
            return self.COLUMNS[section]
        return None

    # --- row-update flash ---

    def _flash_brush(self, row):
        """Translucent tint for a row updated less than FLASH_MS ago."""
        stamp = self._row_flash[row]
        if not stamp:
            return None
        age = self._flash_now - stamp
        if age < 0 or age >= self.FLASH_MS:
            return None
        # Quantise the remaining fraction so repeated repaints reuse a handful
        # of cached QColors instead of allocating one per visible cell.
        step = int((1.0 - age / self.FLASH_MS) * self._FLASH_STEPS)
        if step <= 0:
            return None
        is_tx = self._row_is_tx[row]
        key = (is_tx, step)
        color = self._flash_cache.get(key)
        if color is None:
            rgb = (self._FLASH_RGB_DARK if self._dark_mode
                   else self._FLASH_RGB_LIGHT)[is_tx]
            peak = (self._FLASH_PEAK_DARK if self._dark_mode
                    else self._FLASH_PEAK_LIGHT)
            color = QtGui.QColor(*rgb)
            color.setAlpha(int(peak * step / self._FLASH_STEPS))
            self._flash_cache[key] = color
        return color

    @property
    def flash_active(self):
        """True while at least one row is still lit (drives the fade timer)."""
        return self._flash_active

    def set_dark_mode(self, dark):
        """Theme switch: the tint needs more lift on a dark base."""
        dark = bool(dark)
        if dark != self._dark_mode:
            self._dark_mode = dark
            self._flash_cache.clear()

    def set_flash_enabled(self, enabled):
        """Turn the row-update tint on/off (view preference)."""
        self.flash_enabled = bool(enabled)
        if not self.flash_enabled and self._flash_active:
            self._flash_active = False
            self._row_flash = [0.0] * len(self.rows)
            self._repaint_all()

    def refresh_flash(self):
        """Advance the fade clock. Returns True while any row is still lit.

        Called from CanView's fade timer; the timer stops as soon as this
        returns False, so an idle bus costs nothing.
        """
        if not self._flash_active:
            return False
        self._flash_now = time.monotonic() * 1000.0
        cutoff = self._flash_now - self.FLASH_MS
        still_lit = any(st > cutoff for st in self._row_flash)
        if not still_lit:
            self._flash_active = False
            self._repaint_all()  # final repaint clears the last faint tint
        return still_lit

    def _repaint_all(self):
        if self.rows:
            self.dataChanged.emit(
                self.index(0, self._FLASH_COLUMN),
                self.index(len(self.rows) - 1, self._FLASH_COLUMN),
                [QtCore.Qt.BackgroundRole])

    # --- higher-layer decode ---

    def _decode(self, can_id, ext, data, rtr):
        if self._decoder is None:
            return ''
        try:
            return self._decoder(can_id, ext, data, rtr)
        except Exception:
            # A malformed frame must never take the table down: the decoders
            # index into data that a real bus can truncate in any way.
            return ''

    def set_decode_mode(self, mode):
        """Switch protocol and re-label every row already in the table."""
        if mode == self.decode_mode:
            return
        self.decode_mode = mode
        self._decoder = CAN_DECODERS.get(mode)
        col = self._DECODE_COLUMN
        for i, src in enumerate(self._row_src):
            row = self.rows[i]
            self.rows[i] = row[:col] + (self._decode(*src),) + row[col + 1:]
        if self.rows:
            self.dataChanged.emit(self.index(0, col),
                                  self.index(len(self.rows) - 1, col))

    # --- ingest ---

    def _make_row(self, direction, msg):
        """Format one message into (key, display tuple, is_tx)."""
        ts = getattr(msg, 'timestamp', None)
        if ts is None:
            ts = time.time()
        if self._t0 is None:
            self._t0 = ts
        can_id = getattr(msg, 'arbitration_id', 0)
        ext = bool(getattr(msg, 'is_extended_id', False))
        key = (direction, ext, can_id)

        prev = self._last_ts.get(key)
        self._last_ts[key] = ts
        count = self._counts.get(key, 0) + 1
        self._counts[key] = count
        # Pathological ID churn (random 29-bit IDs / fuzzing) would grow these
        # dicts without bound over hours of monitoring. Losing Δt/Count state
        # in that regime is fine - it was meaningless churn anyway.
        if len(self._last_ts) > 50000:
            self._last_ts.clear()
            self._counts.clear()

        data = bytes(getattr(msg, 'data', b'') or b'')
        type_str = 'EXT' if ext else 'STD'
        if getattr(msg, 'is_remote_frame', False):
            type_str += ' RTR'
        if getattr(msg, 'is_error_frame', False):
            type_str += ' ERR'
        rtr = bool(getattr(msg, 'is_remote_frame', False))
        row = (
            f'{ts - self._t0:.4f}',
            f'{ts - prev:.4f}' if prev is not None else '',
            str(count),
            direction,
            type_str,
            format_can_id(can_id, ext),
            f'{j1939_pgn(can_id):05X}' if ext else '',
            str(getattr(msg, 'dlc', len(data))),
            ' '.join(f'{b:02X}' for b in data),
            self._decode(can_id, ext, data, rtr),
        )
        return key, row, direction == 'TX', (can_id, ext, data, rtr)

    def add_frames(self, batch):
        """Ingest a batch of (direction, msg) tuples (called on the GUI thread)."""
        if not batch:
            return
        if self.fixed_mode:
            self._add_fixed(batch)
        else:
            self._add_scrolling(batch)

    def _add_scrolling(self, batch):
        made = [self._make_row(d, m) for d, m in batch]
        start = len(self.rows)
        self.beginInsertRows(QtCore.QModelIndex(), start, start + len(made) - 1)
        for _key, row, is_tx, src in made:
            self.rows.append(row)
            self._row_is_tx.append(is_tx)
            self._row_src.append(src)
            # No flash in scrolling mode: every row is new, so a tint on all
            # of them says nothing. The stamp is still kept parallel so a
            # mid-session mode switch finds consistent lists.
            self._row_flash.append(0.0)
        self.endInsertRows()
        excess = len(self.rows) - CAN_MAX_SCROLL_ROWS
        if excess > 0:
            self.beginRemoveRows(QtCore.QModelIndex(), 0, excess - 1)
            del self.rows[:excess]
            del self._row_is_tx[:excess]
            del self._row_flash[:excess]
            del self._row_src[:excess]
            self.endRemoveRows()

    def _add_fixed(self, batch):
        changed_min = None
        changed_max = None
        pending = []  # rows for IDs not seen before, appended at the end
        # One clock read per batch: every frame in it arrived within the same
        # flush tick, so they are lit (and fade) together.
        now = time.monotonic() * 1000.0 if self.flash_enabled else 0.0
        for direction, msg in batch:
            key, row, is_tx, src = self._make_row(direction, msg)
            r = self._fixed_index.get(key)
            if r is not None:
                if r < len(self.rows):
                    self.rows[r] = row
                    self._row_src[r] = src
                    self._row_flash[r] = now
                    changed_min = r if changed_min is None else min(changed_min, r)
                    changed_max = r if changed_max is None else max(changed_max, r)
                else:
                    # Row is still in this batch's pending block
                    pending[r - len(self.rows)] = (row, is_tx, src)
            else:
                if len(self.rows) + len(pending) >= CAN_MAX_SCROLL_ROWS:
                    continue  # fixed view full (ID churn/fuzz): drop new IDs
                self._fixed_index[key] = len(self.rows) + len(pending)
                pending.append((row, is_tx, src))
        if pending:
            start = len(self.rows)
            self.beginInsertRows(QtCore.QModelIndex(), start, start + len(pending) - 1)
            for row, is_tx, src in pending:
                self.rows.append(row)
                self._row_is_tx.append(is_tx)
                self._row_src.append(src)
                self._row_flash.append(now)
            self.endInsertRows()
        if changed_min is not None:
            self.dataChanged.emit(
                self.index(changed_min, 0),
                self.index(changed_max, len(self.COLUMNS) - 1))
        if now and (pending or changed_min is not None):
            self._flash_now = now
            self._flash_active = True

    # --- control ---

    def reset_timing(self):
        """Forget per-ID timing/count state (fresh bus session): the first Δt
        and Count of a new session must not be measured against the last
        frames of the previous one (possibly hours old, different hardware)."""
        self._last_ts = {}
        self._counts = {}
        self._t0 = None

    def set_fixed_mode(self, fixed):
        """Switch view mode. Rows restart, but per-ID timing/count state stays."""
        if fixed == self.fixed_mode:
            return
        self.beginResetModel()
        self.fixed_mode = fixed
        self.rows = []
        self._row_is_tx = []
        self._row_flash = []
        self._row_src = []
        self._flash_active = False
        self._fixed_index = {}
        self.endResetModel()

    def clear(self):
        self.beginResetModel()
        self.rows = []
        self._row_is_tx = []
        self._row_flash = []
        self._row_src = []
        self._flash_active = False
        self._fixed_index = {}
        self._last_ts = {}
        self._counts = {}
        self._t0 = None
        self.endResetModel()


class CanView(QtWidgets.QWidget):
    """CAN traffic panel: view-mode selector plus the frame table."""

    def __init__(self, parent):
        super().__init__(parent)
        self.model = CanFrameModel(self)

        view_font = QtGui.QFont('Segoe UI', 10)

        self.view_mode_combo = QtWidgets.QComboBox()
        self.view_mode_combo.addItems(['Scrolling', 'Fixed'])
        self.view_mode_combo.setFont(view_font)
        self.view_mode_combo.setFixedHeight(30)
        self.view_mode_combo.setToolTip(
            'Scrolling: every frame is a new line\n'
            'Fixed: one line per CAN ID, overwritten in place')
        self.view_mode_combo.currentTextChanged.connect(self._on_view_mode_changed)

        # Higher-layer protocol for the Decode column. Explicit rather than
        # sniffed: the same 11-bit ID means different things under CANopen
        # and OBD-II, and a confidently wrong label is worse than none.
        self.decode_combo = QtWidgets.QComboBox()
        self.decode_combo.addItems(CAN_DECODE_MODES)
        self.decode_combo.setCurrentText('Auto')
        self.decode_combo.setFont(view_font)
        self.decode_combo.setFixedHeight(30)
        self.decode_combo.setToolTip(
            'Name well-known frames in the Decode column.\n'
            'J1939: PGN name, source address, destination, priority (29-bit)\n'
            'CANopen: heartbeat/NMT/SYNC/EMCY/PDO/SDO (11-bit)\n'
            'OBD-II: ISO-TP addressing, PCI and UDS service\n'
            'Auto: OBD-II ID ranges first, then J1939 for 29-bit and\n'
            'CANopen for 11-bit. Pick a protocol if Auto mislabels your bus.')
        self.decode_combo.currentTextChanged.connect(self._on_decode_changed)

        # Highlight rows as they refresh. In fixed mode the only visible sign
        # that a frame arrived is text changing in place, which is easy to
        # miss - the tint makes live IDs obvious at a glance.
        self.flash_check = QtWidgets.QCheckBox('Flash')
        self.flash_check.setFont(view_font)
        self.flash_check.setChecked(True)
        self.flash_check.setFixedHeight(30)
        self.flash_check.setToolTip(
            'Fixed mode: light up the Time cell when a frame with that ID\n'
            'arrives, then fade it out. Shows at a glance which IDs are live.')
        self.flash_check.toggled.connect(self._on_flash_toggled)
        # Only meaningful in fixed mode; the view starts in scrolling mode.
        self.flash_check.setEnabled(False)

        self.clear_button = QtWidgets.QPushButton('Clear')
        self.clear_button.setFont(view_font)
        self.clear_button.setFixedHeight(30)
        self.clear_button.clicked.connect(self.model.clear)

        # Repaint ticker for the fade; runs only while something is lit.
        self._fade_timer = QtCore.QTimer(self)
        self._fade_timer.setInterval(40)  # ~25 fps, enough for a 700 ms fade
        self._fade_timer.timeout.connect(self._fade_tick)

        label = QtWidgets.QLabel('CAN frames')
        label.setObjectName('sectionLabel')
        label.setFont(view_font)
        label.setIndent(5)

        # View-level ID filter (log recording still captures everything):
        # pass-list IDs show only those; '!'-prefixed IDs are blocked.
        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setFont(view_font)
        self.filter_edit.setFixedHeight(30)
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.setPlaceholderText('Filter IDs (hex): 123, 18FEF100 or block: !0CF00400')
        self.filter_edit.setToolTip(
            'Show only matching CAN IDs (comma/space separated hex).\n'
            'Prefix an ID with ! to hide it instead.\n'
            'Affects the view only - recording still captures all frames.')
        self.filter_edit.textChanged.connect(self._on_filter_changed)
        self._filter_pass = set()   # arbitration IDs to show exclusively
        self._filter_block = set()  # arbitration IDs to hide

        controls = QtWidgets.QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(4)
        controls.addWidget(label)
        controls.addWidget(self.view_mode_combo)
        controls.addWidget(self.decode_combo)
        controls.addWidget(self.flash_check)
        controls.addWidget(self.filter_edit, stretch=1)
        controls.addWidget(self.clear_button)

        self.table = QtWidgets.QTableView(self)
        self.table.setModel(self.model)
        self.table.setFont(QtGui.QFont('Consolas', 10))
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        vh = self.table.verticalHeader()
        vh.setVisible(False)
        vh.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        vh.setDefaultSectionSize(20)
        hh = self.table.horizontalHeader()
        hh.setStretchLastSection(True)
        hh.setHighlightSections(False)
        for col, width in enumerate([90, 90, 60, 44, 70, 90, 60, 52, 240, 260]):
            self.table.setColumnWidth(col, width)

        # Connection indicator, same DB-9 icon as the serial view
        self.indicator = QtWidgets.QLabel(self)
        self.indicator.setPixmap(create_connector_pixmap(DISCONNECTED_COLOR))

        bottom = QtWidgets.QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.addWidget(self.indicator)
        bottom.addStretch()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addLayout(controls)
        layout.addWidget(self.table, stretch=1)
        layout.addLayout(bottom)

        # The combo was populated before the signal was connected, so push its
        # starting value into the model explicitly.
        self.model.set_decode_mode(self.decode_combo.currentText())

    def _on_view_mode_changed(self, text):
        fixed = (text == 'Fixed')
        self.model.set_fixed_mode(fixed)
        # Scrolling mode has nothing to flash (every row is new), so the
        # toggle is greyed out there rather than silently doing nothing.
        self.flash_check.setEnabled(fixed)
        if not fixed:
            self._fade_timer.stop()
        monitor = self.window()
        if hasattr(monitor, 'schedule_save'):
            monitor.schedule_save()

    def _on_decode_changed(self, text):
        self.model.set_decode_mode(text)
        monitor = self.window()
        if hasattr(monitor, 'schedule_save'):
            monitor.schedule_save()

    def _on_flash_toggled(self, checked):
        self.model.set_flash_enabled(checked)
        if not checked:
            self._fade_timer.stop()
        monitor = self.window()
        if hasattr(monitor, 'schedule_save'):
            monitor.schedule_save()

    def _fade_tick(self):
        """Repaint the fading rows; stop the timer once all are dark."""
        if self.model.refresh_flash():
            self.table.viewport().update()
        else:
            self._fade_timer.stop()

    def _on_filter_changed(self, text):
        """Parse the filter field into pass/block ID sets."""
        passes, blocks = set(), set()
        invalid = False
        for token in text.replace(',', ' ').split():
            target = blocks if token.startswith('!') else passes
            try:
                target.add(int(token.lstrip('!'), 16))
            except ValueError:
                invalid = True
        self._filter_pass = passes
        self._filter_block = blocks
        # Red border = a token isn't valid hex (filter still applies the rest)
        self.filter_edit.setStyleSheet(
            'border: 1px solid #c0453d;' if invalid else '')

    def _frame_passes(self, msg):
        can_id = getattr(msg, 'arbitration_id', 0)
        if can_id in self._filter_block:
            return False
        return not self._filter_pass or can_id in self._filter_pass

    def ingest(self, batch):
        """Add a batch of (direction, msg) tuples; keep autoscroll at bottom."""
        if self._filter_pass or self._filter_block:
            batch = [(d, m) for d, m in batch if self._frame_passes(m)]
            if not batch:
                return
        sb = self.table.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        self.model.add_frames(batch)
        if not self.model.fixed_mode and at_bottom:
            self.table.scrollToBottom()
        if self.model.flash_active and not self._fade_timer.isActive():
            self._fade_timer.start()

    def set_connected(self, connected):
        self.indicator.setPixmap(create_connector_pixmap(
            CONNECTED_COLOR if connected else DISCONNECTED_COLOR))


class SerialMonitor(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.port = QSerialPort()
        self.serialDataView = SerialDataView(self)
        self.canView = CanView(self)
        self.canView.setVisible(False)
        self.serialSendView = SerialSendView(self)

        self.setCentralWidget(QtWidgets.QWidget(self))
        self.layout = QtWidgets.QVBoxLayout(self.centralWidget())
        self.layout.addWidget(self.serialDataView)
        self.layout.addWidget(self.canView)
        self.layout.addWidget(self.serialSendView)
        self.layout.setContentsMargins(3, 3, 3, 3)

        ### CAN bus state ###
        self._ui_mode = 'Serial'
        self._can_bus = None
        self._can_reader = None
        self._can_opener = None      # in-flight CanBusOpener, else None
        self._can_opener_info = ('', 0, 0)  # (iface display name, channel, bitrate)
        self._can_open_cancelled = False
        self._can_reopen_requested = False  # open clicked while a cancelled open resolves
        self._can_closers = []       # running CanBusCloser threads
        self._closing = False        # window closeEvent has started
        self._can_rx_queue = deque()  # ('RX'|'TX', can.Message), thread-safe append
        self._can_rx_frames = 0       # frames this second (stats)
        self._can_tx_frames = 0
        self._can_rx_total = 0
        self._can_tx_total = 0
        self._can_bits = 0            # wire bits this second (bus-load estimate)
        self._can_err_total = 0       # error frames seen this session
        self._can_state = ''          # last controller state (Kvaser only)

        self.setWindowTitle('AxxTerm')
        self.setWindowIcon(QIcon(create_connector_pixmap(CONNECTED_COLOR)))

        ### Menu Bar ###
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

        save_settings_action = file_menu.addAction('Save Settings...')
        save_settings_action.setShortcut('Ctrl+S')
        save_settings_action.triggered.connect(self._menu_save_settings)

        load_settings_action = file_menu.addAction('Load Settings...')
        load_settings_action.setShortcut('Ctrl+O')
        load_settings_action.triggered.connect(self._menu_load_settings)

        file_menu.addSeparator()

        self._record_action = file_menu.addAction('Start Recording')
        self._record_action.setShortcut('Ctrl+R')
        self._record_action.triggered.connect(self._toggle_recording)

        file_menu.addSeparator()

        export_csv_action = file_menu.addAction('Export CSV...')
        export_csv_action.triggered.connect(self._menu_export_csv)

        export_png_action = file_menu.addAction('Export PNG...')
        export_png_action.triggered.connect(self._menu_export_png)

        file_menu.addSeparator()

        quit_action = file_menu.addAction('Quit')
        quit_action.setShortcut('Ctrl+Q')
        quit_action.triggered.connect(self.close)

        ### View Menu ###
        view_menu = menubar.addMenu('View')
        self._dark_mode_action = view_menu.addAction('Dark Mode')
        self._dark_mode_action.setCheckable(True)
        self._dark_mode_action.setShortcut('Ctrl+D')
        self._dark_mode_action.triggered.connect(self._toggle_dark_mode)
        self._dark_mode = False

        self._auto_reconnect_action = view_menu.addAction('Auto-Reconnect')
        self._auto_reconnect_action.setCheckable(True)
        self._auto_reconnect_action.setChecked(True)
        self._auto_reconnect_action.triggered.connect(self._toggle_auto_reconnect)
        self._auto_reconnect = True

        self._timestamps_action = view_menu.addAction('Timestamps')
        self._timestamps_action.setCheckable(True)
        self._timestamps_action.setShortcut('Ctrl+T')
        self._timestamps_action.setToolTip(
            'Prefix each received line with the arrival time (ASCII view)')
        self._timestamps_action.triggered.connect(self._toggle_timestamps)

        ### Edit Menu ###
        edit_menu = menubar.addMenu('Edit')
        find_action = edit_menu.addAction('Find')
        find_action.setShortcut('Ctrl+F')
        find_action.triggered.connect(lambda: self.serialDataView.toggle_search())

        ### Tool Bar ###
        self.toolBar = ToolBar(self)
        self.addToolBar(self.toolBar)

        ### Status Bar ###
        self.setStatusBar(QtWidgets.QStatusBar(self))
        self._record_btn = QtWidgets.QPushButton('Record', self)
        self._record_btn.setObjectName('recordButton')
        self._record_btn.setCheckable(True)
        self._record_btn.setFixedHeight(22)
        # Fixed width that fits the longer 'Recording...' label (plus QSS
        # padding/border), so the text never clips and the status bar does
        # not jump when recording starts.
        self._record_btn.setFixedWidth(
            self._record_btn.fontMetrics().horizontalAdvance('Recording...') + 32)
        self._record_btn.clicked.connect(self._toggle_recording)
        self.statusBar().addWidget(self._record_btn)
        self.statusText = QtWidgets.QLabel(self)
        self.statusBar().addWidget(self.statusText)
        self.statsLabel = QtWidgets.QLabel(self)
        self.statsLabel.setFont(QtGui.QFont('Segoe UI', 10))
        self.statusBar().addPermanentWidget(self.statsLabel)

        ### Recording state ###
        self._log_file = None
        self._recording = False
        self._rx_log_pending = ''  # partial RX line awaiting its newline
        self._asc_writer = None    # Vector .asc CAN log (interoperable format)

        ### Window state restore ###
        self._geometry_restored = False
        self._splitter_sizes_to_restore = None

        ### Debounced settings save (coalesce rapid changes into one disk write) ###
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(750)
        self._save_timer.timeout.connect(lambda: self.save_all_settings())

        ### Throughput tracking ###
        self._rx_bytes = 0
        self._tx_bytes = 0
        self._rx_total = 0
        self._tx_total = 0
        self._stats_timer = QtCore.QTimer(self)
        self._stats_timer.timeout.connect(self._update_stats)
        self._stats_timer.start(1000)

        ### Display throttle (~30 fps) ###
        self._rx_buffer = bytearray()
        self._rx_frozen_dropped = False
        self._display_timer = QtCore.QTimer(self)
        self._display_timer.timeout.connect(self._flush_display)
        self._display_timer.start(33)  # ~30fps

        ### Auto-reconnect state ###
        self._reconnect_port_name = ''
        self._reconnect_timer = QtCore.QTimer(self)
        self._reconnect_timer.timeout.connect(self._try_reconnect)

        ### Signal Connect ###
        self.toolBar.portOpenButton.clicked.connect(self.portOpen)
        self.serialSendView.serialSendSignal.connect(self.sendFromPort)
        self.serialSendView.canSendSignal.connect(self.sendCanFrame)
        self.port.readyRead.connect(self.readFromPort)
        self.port.errorOccurred.connect(self._on_port_error)
        self.toolBar.modeCombo.currentTextChanged.connect(self._on_ui_mode_changed)

        # Line controls (work live on the open port)
        self.toolBar.dtrCheck.toggled.connect(self._on_dtr_toggled)
        self.toolBar.rtsCheck.toggled.connect(self._on_rts_toggled)
        self.toolBar.breakButton.clicked.connect(self._send_break)

        # Save when serial port settings change (debounced).
        # baudRates uses currentTextChanged so typed custom rates save too.
        self.toolBar.baudRates.currentTextChanged.connect(lambda: self.schedule_save())
        self.toolBar.dataBits.currentIndexChanged.connect(lambda: self.schedule_save())
        self.toolBar._parity.currentIndexChanged.connect(lambda: self.schedule_save())
        self.toolBar.stopBits.currentIndexChanged.connect(lambda: self.schedule_save())
        self.toolBar._flowControl.currentIndexChanged.connect(lambda: self.schedule_save())
        self.toolBar.canInterfaces.currentIndexChanged.connect(lambda: self.schedule_save())
        self.toolBar.canChannels.currentIndexChanged.connect(lambda: self.schedule_save())
        self.toolBar.canBitrates.currentIndexChanged.connect(lambda: self.schedule_save())

        ### Apply default (light) theme, then load all settings ###
        # (load_all_settings switches to the dark palette if the user saved it,
        # but returns early when no settings file exists yet)
        self._apply_light_palette()
        self.load_all_settings()

    def portOpen(self, flag):
        # Manual open/close always cancels any pending auto-reconnect
        self._reconnect_timer.stop()
        self._reconnect_port_name = ''

        if self._ui_mode == 'CAN':
            if flag:
                self._open_can_bus()
            else:
                self._close_can_bus()
            return

        if flag:
            if not self.toolBar.baudRates.currentText().strip():
                # Empty custom-baud field: show the fallback instead of
                # silently opening at a rate the (blank) combo doesn't show
                self.toolBar.baudRates.setCurrentText('115200')
            self.port.setBaudRate(self.toolBar.baudRate())
            self.port.setPortName(self.toolBar.portName())
            self.port.setDataBits(self.toolBar.dataBit())
            self.port.setParity(self.toolBar.parity())
            self.port.setStopBits(self.toolBar.stopBit())
            self.port.setFlowControl(self.toolBar.flowControl())

            r = self.port.open(QtCore.QIODevice.ReadWrite)
            if not r:
                # errorString distinguishes "in use by another program" from
                # "device not found" etc. - a bare 'Port open error' doesn't.
                self.statusText.setText(
                    f'Port open error: {self.port.errorString()}')
                self.toolBar.portOpenButton.setChecked(False)
                self.toolBar.serialControlEnable(True)
            else:
                self.statusText.setText('Port opened')
                # Apply the line-control checkboxes to the fresh connection
                self.port.setDataTerminalReady(self.toolBar.dtrCheck.isChecked())
                if self.toolBar.flowControl() != QSerialPort.HardwareControl:
                    self.port.setRequestToSend(self.toolBar.rtsCheck.isChecked())
                self._rx_bytes = 0
                self._tx_bytes = 0
                self._rx_total = 0
                self._tx_total = 0
                self._reset_stream_state()
                self.toolBar.serialControlEnable(False)
                self.serialDataView.label.setPixmap(create_connector_pixmap(CONNECTED_COLOR))
        else:
            self.port.close()
            self.serialSendView.stop_repeat()
            self._flush_rx_log_pending()  # last partial line reaches the log
            self.statusText.setText('Port closed')
            self.toolBar.serialControlEnable(True)
            self.serialDataView.label.setPixmap(create_connector_pixmap(DISCONNECTED_COLOR))

    # --- CAN mode -----------------------------------------------------------

    def _on_ui_mode_changed(self, mode):
        """Switch between Serial and CAN mode (toolbar combo)."""
        # Close whichever connection is open before switching. The serial port
        # is closed directly (not via portOpen, which dispatches on the mode),
        # and any pending auto-reconnect is cancelled so it can't reopen the
        # port behind the CAN UI.
        self._reconnect_timer.stop()
        self._reconnect_port_name = ''
        self.serialSendView.stop_repeat()
        if self.port.isOpen():
            self.port.close()
            self._flush_rx_log_pending()  # same as the regular close path
            self.statusText.setText('Port closed')
            self.serialDataView.label.setPixmap(create_connector_pixmap(DISCONNECTED_COLOR))
        if self._can_bus is not None or self._can_opener is not None:
            self._close_can_bus()
        self.toolBar.serialControlEnable(True)
        self.toolBar.portOpenButton.setChecked(False)

        self._ui_mode = mode
        is_can = (mode == 'CAN')
        self.toolBar.set_can_mode(is_can)
        self.serialDataView.setVisible(not is_can)
        self.canView.setVisible(is_can)
        self.serialSendView.set_can_mode(is_can)
        self.schedule_save()

    def _open_can_bus(self):
        if self._can_bus is not None:
            return  # already open
        if self._can_opener is not None:
            if self._can_open_cancelled:
                # A cancelled open is still resolving; reopen (with the
                # current toolbar settings) as soon as it does.
                self._can_reopen_requested = True
                self.toolBar.serialControlEnable(False)
                self.statusText.setText('Reopening CAN...')
            return  # an open is already in progress
        iface_name = self.toolBar.canInterfaces.currentText()
        iface = CAN_INTERFACES.get(iface_name, 'virtual')
        channel = self.toolBar.canChannel()
        bitrate = self.toolBar.canBitrate()
        # Open on a worker thread; driver init can block for seconds and
        # would freeze the whole UI if done here.
        self._can_open_cancelled = False
        self._can_closers = [c for c in self._can_closers if c.isRunning()]
        self._can_opener = CanBusOpener(iface, channel, bitrate,
                                        wait_for=self._can_closers, parent=self)
        self._can_opener_info = (iface_name, channel, bitrate)
        self._can_opener.succeeded.connect(self._on_can_opened)
        self._can_opener.failed.connect(self._on_can_open_failed)
        self._can_opener.start()
        self.toolBar.serialControlEnable(False)
        self.statusText.setText(f'Opening CAN: {iface_name} ch {channel}...')

    def _maybe_reopen_can_bus(self):
        """Start the open that was requested while a cancelled one resolved."""
        if self._can_reopen_requested:
            self._can_reopen_requested = False
            if self._ui_mode == 'CAN':
                self._open_can_bus()

    def _on_can_opened(self, bus):
        self._can_opener = None
        if self._closing:
            # Delivered between closeEvent and actual destruction: shut the
            # bus down inline - a new closer thread would be created on a
            # dying window after the teardown waits already ran.
            try:
                bus.shutdown()
            except Exception:
                pass
            return
        if self._can_open_cancelled:
            # User closed / switched mode while the bus was still opening:
            # the freshly opened bus just gets shut down again.
            closer = CanBusCloser(None, bus, self)
            self._can_closers.append(closer)
            closer.start()
            self._maybe_reopen_can_bus()
            return
        iface_name, channel, bitrate = self._can_opener_info
        self._can_bus = bus
        self._can_rx_queue.clear()
        self._can_rx_frames = 0
        self._can_tx_frames = 0
        self._can_rx_total = 0
        self._can_tx_total = 0
        self._can_bits = 0
        self._can_err_total = 0
        self._can_state = ''  # controller state is per session, like the counters
        self.canView.model.reset_timing()  # Δt/Count restart per session
        self._can_reader = CanReaderThread(bus, self._can_rx_queue, self)
        self._can_reader.errorOccurred.connect(self._on_can_error)
        self._can_reader.start()
        self.canView.set_connected(True)
        self.statusText.setText(
            f'CAN open: {iface_name} ch {channel} @ {bitrate // 1000} kbit/s')

    def _on_can_open_failed(self, message):
        self._can_opener = None
        if self._can_open_cancelled:
            # The UI was already reset by _close_can_bus, and the user may
            # have moved on (e.g. opened a serial port) - leave it alone.
            self._maybe_reopen_can_bus()
            return
        self.statusText.setText(f'CAN open error: {message}')
        self.toolBar.portOpenButton.setChecked(False)
        self.toolBar.serialControlEnable(True)

    def _close_can_bus(self):
        self.serialSendView.stop_repeat()
        self._can_reopen_requested = False
        if self._can_opener is not None:
            # Still opening: flag it; _on_can_opened shuts the bus down on
            # arrival. The opener outcome handlers restore the UI state.
            self._can_open_cancelled = True
        reader, self._can_reader = self._can_reader, None
        bus, self._can_bus = self._can_bus, None
        if reader is not None:
            # A bus error that fired right around this close must not reach
            # _on_can_error anymore - it would tear down the *next* session.
            try:
                reader.errorOccurred.disconnect(self._on_can_error)
            except TypeError:
                pass
        if reader is not None or bus is not None:
            # Teardown (reader join + driver shutdown) runs off the GUI
            # thread. A subsequent open waits for these closers, so the
            # channel is never reopened while it is still closing.
            self._can_closers = [c for c in self._can_closers if c.isRunning()]
            closer = CanBusCloser(reader, bus, self)
            self._can_closers.append(closer)
            closer.start()
        self.toolBar.serialControlEnable(True)
        self.canView.set_connected(False)
        self.statusText.setText('CAN closed')

    def _on_can_error(self, message):
        """Bus died under the reader thread (device unplugged, driver error)."""
        self._close_can_bus()
        self.toolBar.portOpenButton.setChecked(False)
        self.statusText.setText(f'CAN bus error: {message}')

    def sendCanFrame(self, id_text, extended, data_text):
        """Send one CAN frame (from the send row or a macro button)."""
        if self._send_can_frame(id_text, extended, data_text):
            self.serialSendView.note_send_result(True)
        else:
            # Repeat must stop on persistent failure: with a non-ACKing bus
            # every attempt blocks the GUI ~300 ms waiting for confirmation.
            self.serialSendView.note_send_result(False)

    def _send_can_frame(self, id_text, extended, data_text):
        if self._can_bus is None:
            self.statusText.setText('CAN bus is not open')
            return False
        try:
            can_id = int(id_text, 16)
        except (TypeError, ValueError):
            self.statusText.setText(f'Invalid CAN ID (hex): {id_text!r}')
            return False
        max_id = 0x1FFFFFFF if extended else 0x7FF
        if not 0 <= can_id <= max_id:
            self.statusText.setText(
                f'CAN ID {can_id:X} out of range (max {max_id:X} for '
                f'{"extended" if extended else "standard"})')
            return False
        try:
            data = bytes.fromhex(data_text.replace(' ', ''))
        except ValueError:
            self.statusText.setText('CAN data is not a valid HEX string')
            return False
        if len(data) > 8:
            self.statusText.setText(f'CAN data is max 8 bytes (got {len(data)})')
            return False
        msg = pycan.Message(arbitration_id=can_id, is_extended_id=extended,
                            data=data, timestamp=time.time())
        try:
            # A timeout makes the Kvaser backend wait for on-wire transmit
            # confirmation (canWriteSync) and raise on failure. Without it,
            # send() only queues in the driver: with a wrong bitrate or no
            # ACKing node, frames were shown/logged as TX but never sent.
            self._can_bus.send(msg, timeout=0.3)
        except Exception as e:
            self.statusText.setText(f'CAN send failed: {e}')
            return False
        self.statusText.setText('')
        # Show the sent frame in the table via the same drain path as RX
        self._can_rx_queue.append(('TX', msg))
        return True

    def _flush_can_queue(self):
        """Drain buffered CAN frames into the table (and the log)."""
        q = self._can_rx_queue
        if not q:
            return
        batch = []
        while q:
            batch.append(q.popleft())
        for direction, msg in batch:
            if direction == 'TX':
                self._can_tx_frames += 1
                self._can_tx_total += 1
            else:
                self._can_rx_frames += 1
                self._can_rx_total += 1
            # Bus-load estimate: frame overhead (47 bits std / 67 ext incl.
            # interframe space) + data, before bit stuffing (~+10% applied
            # in the stats display). The same approach PCAN-View uses.
            data_len = len(getattr(msg, 'data', b'') or b'')
            self._can_bits += (67 if getattr(msg, 'is_extended_id', False)
                               else 47) + 8 * data_len
            if getattr(msg, 'is_error_frame', False):
                self._can_err_total += 1
            if self._recording and self._log_file is not None:
                self._log_data(direction, format_can_log_line(msg),
                               when=getattr(msg, 'timestamp', None))
            if self._asc_writer is not None:
                self._write_asc(direction, msg)
        self.canView.ingest(batch)

    def _flush_rx_log_pending(self):
        """Log the held partial RX line now, instead of dropping it or letting
        it surface out of order at recording stop."""
        if self._rx_log_pending:
            pending, self._rx_log_pending = self._rx_log_pending, ''
            self._log_data('RX', pending)

    def _reset_stream_state(self):
        """Drop buffered/partial stream state so a new connection starts clean."""
        self._rx_buffer.clear()
        # Flush (not drop) any partial RX line held for the log, so it isn't
        # lost - and isn't merged with post-reconnect data into one line.
        self._flush_rx_log_pending()
        self.serialDataView.reset_stream_state()

    def readFromPort(self):
        data = self.port.readAll()
        if len(data) > 0:
            raw_bytes = bytes(data.data())
            self._rx_bytes += len(raw_bytes)
            self._rx_total += len(raw_bytes)
            # Log immediately for accurate timestamps
            if self._recording and self._log_file is not None:
                mode = self.serialDataView.data_mode.currentText()
                if mode == 'ASCII':
                    # Buffer partial lines so each logged line is one device line
                    text = self._rx_log_pending + raw_bytes.decode('ISO-8859-1')
                    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
                    self._rx_log_pending = lines[-1][-4096:]
                    complete = '\n'.join(lines[:-1])
                    if complete:
                        self._log_data('RX', complete)
                else:
                    hex_str = raw_bytes.hex().upper()
                    self._log_data('RX', f'[HEX] {hex_str}')
            # Buffer for throttled display update
            self._rx_buffer.extend(raw_bytes)

    # Backlog cap while the display is frozen: logging keeps the full stream
    # (readFromPort logs immediately); this only bounds what re-renders later.
    FROZEN_BUFFER_CAP = 4 * 1024 * 1024
    # Max bytes rendered per tick: a frozen multi-MB backlog drains over a few
    # ticks instead of one seconds-long QTextEdit insert that stalls the UI.
    FLUSH_CHUNK = 128 * 1024

    def _flush_display(self):
        """Process buffered RX data and update the display (~30 fps)."""
        if self.serialDataView.freeze_btn.isChecked():
            if len(self._rx_buffer) > self.FROZEN_BUFFER_CAP:
                del self._rx_buffer[:len(self._rx_buffer) - self.FROZEN_BUFFER_CAP]
                self._rx_frozen_dropped = True
            self._flush_can_queue()  # CAN table is a separate view; keep it live
            return
        if self._rx_frozen_dropped:
            self._rx_frozen_dropped = False
            self.serialDataView._insert_colored_text(
                self.serialDataView.serialData,
                '\n[... oldest data not shown (frozen too long) - the log file has everything ...]\n',
                QtGui.QColor(128, 128, 128))
        if self._rx_buffer:
            data = bytes(self._rx_buffer[:self.FLUSH_CHUNK])
            del self._rx_buffer[:self.FLUSH_CHUNK]
            self.serialDataView.handleReceivedData(data)
        # Push any pending plot samples to the curves (one setData per channel)
        self.serialDataView.flush_plot()
        self._flush_can_queue()

    def _write_serial(self, tx):
        """Write bytes to the open port, counting only accepted bytes.

        QSerialPort.write returns -1 on immediate error (previously ignored,
        so failed TX was counted, logged, and echoed as if it had been sent).
        """
        n = self.port.write(tx)
        if n != len(tx):
            self.statusText.setText(f'Write failed: {self.port.errorString()}')
            return False
        self._tx_bytes += n
        self._tx_total += n
        self.statusText.setText('')
        return True

    def sendFromPort(self, text):
        if not self.port.isOpen():
            self.statusText.setText('Port is not open')
            self.serialSendView.note_send_result(False)
            return
        sent = False
        if self.serialSendView.charMode.currentText() == 'HEX':
            if self.serialSendView.lineEnding.currentIndex() == 1:
                text = text + '0A'
            elif self.serialSendView.lineEnding.currentIndex() == 2:
                text = text + '0D'
            elif self.serialSendView.lineEnding.currentIndex() == 3:
                text = text + '0D0A'
            try:
                tx = bytes.fromhex(text)
                sent = self._write_serial(tx)
            except ValueError:
                self.statusText.setText('Not a valid HEX string')

        elif self.serialSendView.charMode.currentText() == 'ASCII':
            if self.serialSendView.lineEnding.currentIndex() == 1:
                text = text + '\n'
            elif self.serialSendView.lineEnding.currentIndex() == 2:
                text = text + '\r'
            elif self.serialSendView.lineEnding.currentIndex() == 3:
                text = text + '\r\n'
            try:
                # Match the RX/display encoding (ISO-8859-1); fall back to
                # UTF-8 only for characters outside Latin-1.
                try:
                    tx = text.encode('ISO-8859-1')
                except UnicodeEncodeError:
                    tx = text.encode('utf-8')
                sent = self._write_serial(tx)
            except (UnicodeEncodeError, ValueError):
                self.statusText.setText('Not a valid ASCII string')

        elif self.serialSendView.charMode.currentText() == 'BINARY':
            try:
                bits = text.replace(' ', '')
                value = int(bits, 2)
                # Width from the typed bit string, so leading zero bytes survive
                num_bytes = max(1, (len(bits) + 7) // 8)
                tx = value.to_bytes(num_bytes, byteorder='big')
                ending = [b'', b'\n', b'\r', b'\r\n'][self.serialSendView.lineEnding.currentIndex()]
                tx += ending
                sent = self._write_serial(tx)
            except (ValueError, OverflowError):
                self.statusText.setText('Not a valid BINARY string')

        self.serialSendView.note_send_result(sent)
        if sent:
            mode = self.serialSendView.charMode.currentText()
            if mode == 'HEX':
                self._log_data('TX', f'[HEX] {text.upper()}')
            elif mode == 'BINARY':
                self._log_data('TX', f'[BIN] {text}')
            else:
                self._log_data('TX', text)
            self.serialDataView.appendSerialText(text, "send", mode)

    def _update_stats(self):
        """Update status bar with throughput and totals (called every second)."""
        def fmt(n):
            if n >= 1_000_000:
                return f'{n / 1_000_000:.1f} MB'
            if n >= 1_000:
                return f'{n / 1_000:.1f} kB'
            return f'{n} B'

        rx_rate = self._rx_bytes
        tx_rate = self._tx_bytes
        self._rx_bytes = 0
        self._tx_bytes = 0

        # Periodic log flush (writes are buffered on the RX hot path).
        if self._recording and self._log_file is not None:
            try:
                self._log_file.flush()
            except (OSError, ValueError):
                pass

        if self._can_bus is not None:
            can_rx = self._can_rx_frames
            can_tx = self._can_tx_frames
            self._can_rx_frames = 0
            self._can_tx_frames = 0
            bits, self._can_bits = self._can_bits, 0
            bitrate = max(1, self.toolBar.canBitrate())
            load = min(100.0, bits * 1.1 / bitrate * 100)  # ~10% stuff bits
            err = f'  Err: {self._can_err_total}' if self._can_err_total else ''
            text = (f'Load: {load:.1f}%{err}  |  '
                    f'RX: {can_rx} msg/s  TX: {can_tx} msg/s  |  '
                    f'RX total: {self._can_rx_total}  TX total: {self._can_tx_total}  |  '
                    f'{bitrate // 1000} kbit/s')
            # Controller state (Kvaser only; None on other backends). BUS-OFF
            # means the controller has dropped off the bus entirely - it has
            # to be impossible to miss, so it is coloured and carries the
            # error counters that explain it.
            state = kvaser_bus_state(self._can_bus)
            if state is not None:
                name, tx_err, rx_err = state
                if name != self._can_state and name != 'ACTIVE':
                    # Degrading is an event, not just a readout: say it once
                    # in the status line so a glance away does not miss it.
                    self.statusText.setText(
                        f'CAN controller {name} (TX errors {tx_err}, '
                        f'RX errors {rx_err})' +
                        (' - not transmitting or receiving'
                         if name == 'BUS-OFF' else ''))
                self._can_state = name
                color = CAN_STATE_COLORS.get(name, '')
                counters = (f' (TEC {tx_err} REC {rx_err})'
                            if name != 'ACTIVE' else '')
                if color:
                    text = (html.escape(text) +
                            f'  |  <b><span style="color:{color}">'
                            f'{name}</span></b>{counters}')
                else:
                    text += f'  |  {name}'
            self.statsLabel.setText(text)
        elif self.port.isOpen():
            baud = self.toolBar.baudRate()
            self.statsLabel.setText(
                f'RX: {fmt(rx_rate)}/s  TX: {fmt(tx_rate)}/s  |  '
                f'RX total: {fmt(self._rx_total)}  TX total: {fmt(self._tx_total)}  |  '
                f'{baud} baud')
        else:
            self.statsLabel.setText('')

    def _apply_dark_palette(self):
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QtGui.QPalette.WindowText, QColor(255, 255, 255))
        palette.setColor(QtGui.QPalette.Base, QColor(35, 35, 35))
        palette.setColor(QtGui.QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QtGui.QPalette.ToolTipBase, QColor(25, 25, 25))
        palette.setColor(QtGui.QPalette.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QtGui.QPalette.Text, QColor(255, 255, 255))
        palette.setColor(QtGui.QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QtGui.QPalette.ButtonText, QColor(255, 255, 255))
        palette.setColor(QtGui.QPalette.BrightText, QColor(255, 0, 0))
        palette.setColor(QtGui.QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QtGui.QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QtGui.QPalette.HighlightedText, QColor(35, 35, 35))
        QtWidgets.QApplication.instance().setPalette(palette)
        QtWidgets.QApplication.instance().setStyleSheet(
            DARK_QSS
            .replace('%COMBO_ARROW%', arrow_icon_url('#b8bfc6', 'down'))
            .replace('%ARROW_UP%', arrow_icon_url('#b8bfc6', 'up'))
            .replace('%ARROW_DOWN%', arrow_icon_url('#b8bfc6', 'down')))
        self.canView.model.set_dark_mode(True)

    def _apply_light_palette(self):
        QtWidgets.QApplication.instance().setPalette(
            QtWidgets.QApplication.style().standardPalette())
        QtWidgets.QApplication.instance().setStyleSheet(
            LIGHT_QSS.replace('%COMBO_ARROW%', arrow_icon_url('#5a6570', 'down')))
        self.canView.model.set_dark_mode(False)

    def _toggle_dark_mode(self):
        self._dark_mode = self._dark_mode_action.isChecked()
        if self._dark_mode:
            self._apply_dark_palette()
        else:
            self._apply_light_palette()
        self.serialDataView._update_graph_theme()
        self.save_all_settings()

    def _toggle_auto_reconnect(self):
        self._auto_reconnect = self._auto_reconnect_action.isChecked()
        if not self._auto_reconnect:
            self._reconnect_timer.stop()
            if self._reconnect_port_name:
                self._reconnect_port_name = ''
                self.statusText.setText('Auto-reconnect disabled')
        self.save_all_settings()

    def _toggle_timestamps(self):
        self.serialDataView.show_timestamps = self._timestamps_action.isChecked()
        self.schedule_save()

    # --- Serial line controls (DTR / RTS / Break) ---------------------------

    def _on_dtr_toggled(self, checked):
        if self.port.isOpen():
            self.port.setDataTerminalReady(checked)

    def _on_rts_toggled(self, checked):
        if not self.port.isOpen():
            return
        if self.port.flowControl() == QSerialPort.HardwareControl:
            self.statusText.setText('RTS is managed by hardware flow control')
            return
        self.port.setRequestToSend(checked)

    def _send_break(self):
        if not self.port.isOpen():
            self.statusText.setText('Port is not open')
            return
        self.port.setBreakEnabled(True)
        QtCore.QTimer.singleShot(300, self._end_break)
        self.statusText.setText('Break sent (300 ms)')

    def _end_break(self):
        if self.port.isOpen():
            self.port.setBreakEnabled(False)

    def _on_port_error(self, error):
        """Surface serial port errors. On ResourceError (device unplugged), close
        gracefully and optionally start scanning for reconnection."""
        if error == QSerialPort.NoError:
            return
        if error != QSerialPort.ResourceError:
            # WriteError, ReadError, PermissionError, ... were previously
            # swallowed entirely: failed I/O looked like success and a dead
            # RX path kept showing "connected". Show the driver's reason.
            self.statusText.setText(f'Serial error: {self.port.errorString()}')
            return
        if error == QSerialPort.ResourceError and self.port.isOpen():
            self._reconnect_port_name = self.port.portName()
            self.port.close()
            self.serialSendView.stop_repeat()
            self.toolBar.portOpenButton.setChecked(False)
            self.toolBar.serialControlEnable(True)
            self.serialDataView.label.setPixmap(create_connector_pixmap(DISCONNECTED_COLOR))
            if self._auto_reconnect:
                self.statusText.setText(
                    f'Port disconnected - reconnecting to {self._reconnect_port_name}...')
                self._reconnect_timer.start(1000)
            else:
                self.statusText.setText('Port disconnected')
                self._reconnect_port_name = ''

    def _try_reconnect(self):
        """Poll available ports for the previously connected port name."""
        if self._ui_mode != 'Serial':  # never reopen serial behind the CAN UI
            self._reconnect_timer.stop()
            self._reconnect_port_name = ''
            return
        available = [p.portName() for p in QSerialPortInfo.availablePorts()]
        if self._reconnect_port_name not in available:
            return
        # Port reappeared -- try to open with the current toolbar settings
        self.port.setPortName(self._reconnect_port_name)
        self.port.setBaudRate(self.toolBar.baudRate())
        self.port.setDataBits(self.toolBar.dataBit())
        self.port.setParity(self.toolBar.parity())
        self.port.setStopBits(self.toolBar.stopBit())
        self.port.setFlowControl(self.toolBar.flowControl())
        if self.port.open(QtCore.QIODevice.ReadWrite):
            self._reconnect_timer.stop()
            self._reconnect_port_name = ''
            self.toolBar.portOpenButton.setChecked(True)
            self.toolBar.serialControlEnable(False)
            self.serialDataView.label.setPixmap(create_connector_pixmap(CONNECTED_COLOR))
            self._rx_bytes = 0
            self._tx_bytes = 0
            self._rx_total = 0
            self._tx_total = 0
            # Drop partial frames/lines from before the disconnect so the
            # decoders don't permanently misalign on the reconnected stream
            self._reset_stream_state()
            self.statusText.setText('Port reconnected')

    # --- Recording / Logging ---------------------------------------------------

    def _log_data(self, direction, text, when=None):
        """Write a timestamped line to the log file if recording is active.

        when: optional epoch timestamp for the event (e.g. the CAN frame's
        hardware timestamp); defaults to now. CAN frames are drained on the
        ~30 fps display timer, so stamping at drain time would collapse every
        frame of a tick onto one wall-clock time up to 33 ms late.
        """
        if not self._recording or self._log_file is None:
            return
        try:
            now = datetime.fromtimestamp(when) if when else datetime.now()
        except (OverflowError, OSError, ValueError):
            now = datetime.now()
        ts = now.strftime('%Y-%m-%d %H:%M:%S.') + f'{now.microsecond // 1000:03d}'
        try:
            for line in text.splitlines():
                if line:
                    self._log_file.write(f'[{ts}] {direction}: {line}\n')
            # Flushing here on every readyRead would issue many fsync-adjacent
            # syscalls per second on the GUI thread at high baud. Timestamps are
            # already captured above; the actual flush is piggybacked on the 1s
            # stats timer (and forced on stop/close).
        except (OSError, ValueError):
            # Disk full / file gone: stop recording instead of crashing the RX path
            try:
                self._log_file.close()
            except (OSError, ValueError):
                pass
            self._log_file = None
            self._recording = False
            self._rx_log_pending = ''
            self._record_btn.setChecked(False)
            self._record_btn.setText('Record')
            self._record_action.setText('Start Recording')
            self.statusText.setText('Recording stopped: log file write failed')

    def _write_asc(self, direction, msg):
        """Mirror one CAN frame into the .asc log (standard Vector format
        readable by CANalyzer/SavvyCAN/asammdf, unlike the .txt log)."""
        try:
            try:
                msg.is_rx = (direction != 'TX')  # .asc direction column
            except AttributeError:
                pass  # python-can < 4.0 Message has no is_rx; log without it
            self._asc_writer(msg)
        except Exception:
            self._stop_asc_writer()
            self.statusText.setText('.asc log write failed - .asc recording stopped')

    def _stop_asc_writer(self):
        if self._asc_writer is not None:
            try:
                self._asc_writer.stop()
            except Exception:
                pass
            self._asc_writer = None

    def _toggle_recording(self):
        """Start or stop recording serial data to a log file."""
        if self._recording:
            # Stop recording: flush queued CAN frames and any partial RX line
            # first so nothing buffered in the last display tick is lost
            self._flush_can_queue()
            self._stop_asc_writer()
            if self._rx_log_pending:
                pending, self._rx_log_pending = self._rx_log_pending, ''
                self._log_data('RX', pending)
            if self._log_file is not None:
                try:
                    self._log_file.close()
                except OSError:
                    pass
                self._log_file = None
            self._recording = False
            self._record_btn.setChecked(False)
            self._record_btn.setText('Record')
            self._record_action.setText('Start Recording')
            self.statusText.setText('Recording stopped')
        else:
            # Start recording
            now = datetime.now()
            ts = now.strftime('%Y-%m-%d_%H-%M-%S_') + f'{now.microsecond // 1000:03d}'
            filename = f'AxxTerm_log_{ts}.txt'
            filepath = os.path.join(SCRIPT_DIR, filename)
            try:
                self._log_file = open(filepath, 'w', encoding='utf-8')
            except OSError as e:
                self.statusText.setText(f'Cannot create log file: {e}')
                self._record_btn.setChecked(False)
                return
            # In CAN mode, additionally record a Vector .asc file: the .txt
            # log is human-readable but a dead end for tooling, while .asc
            # loads into CANalyzer/SavvyCAN/asammdf for real analysis.
            extra = ''
            if self._ui_mode == 'CAN' and pycan is not None:
                try:
                    self._asc_writer = pycan.ASCWriter(filepath[:-4] + '.asc')
                    extra = ' (+.asc)'
                except Exception:
                    self._asc_writer = None
            self._recording = True
            self._record_btn.setChecked(True)
            self._record_btn.setText('Recording...')
            self._record_action.setText('Stop Recording')
            self.statusText.setText(f'Recording to {filename}{extra}')

    def closeEvent(self, event):
        """Flush pending state and close the log file when the application exits."""
        self._closing = True
        self._save_timer.stop()
        # Stop the remaining timers so nothing fires against half-torn-down widgets.
        self._stats_timer.stop()
        self._display_timer.stop()
        self._reconnect_timer.stop()
        if self._can_bus is not None or self._can_opener is not None:
            self._close_can_bus()
        # Give the teardown threads a bounded window to finish. A thread that
        # does not finish (hung driver call) is detached from the window:
        # destroying a running QThread aborts the process.
        if self._can_opener is not None:
            if self._can_opener.wait(5000):
                # The queued succeeded signal can't be delivered anymore, so
                # shut down a bus that finished opening during exit directly.
                bus = self._can_opener.bus
                if bus is not None:
                    try:
                        bus.shutdown()
                    except Exception:
                        pass
            else:
                _orphan_thread(self._can_opener)
        for closer in self._can_closers:
            if not closer.wait(3000):
                _orphan_thread(closer)
            elif closer.reader is not None and closer.reader.isRunning():
                _orphan_thread(closer.reader)  # stuck in a driver recv()
        # Readers are stopped now; log whatever they queued during the last tick
        self._flush_can_queue()
        self._stop_asc_writer()
        self.save_all_settings()  # also captures final window/splitter geometry
        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None
        self._recording = False
        super().closeEvent(event)

    def schedule_save(self):
        """Request a debounced settings save (rapid changes coalesce into one write)."""
        # Child widgets fire change signals during construction, before the
        # timer exists; there is nothing worth saving at that point anyway.
        if hasattr(self, '_save_timer'):
            self._save_timer.start()

    def save_all_settings(self, path=None):
        """Save all settings (plot, serial port, macros) to one JSON file."""
        settings = {
            'dark_mode': self._dark_mode,
            'auto_reconnect': self._auto_reconnect,
            'show_timestamps': self.serialDataView.show_timestamps,
            'ui_mode': self._ui_mode,
            'can': {
                'interface': self.toolBar.canInterfaces.currentText(),
                'channel': self.toolBar.canChannels.currentIndex(),
                'bitrate': self.toolBar.canBitrates.currentText(),
                'view_mode': self.canView.view_mode_combo.currentText(),
                'flash': self.canView.flash_check.isChecked(),
                'decode': self.canView.decode_combo.currentText(),
            },
            'window': {
                'geometry': bytes(self.saveGeometry().toHex()).decode('ascii'),
                'splitter': self.serialDataView.splitter.sizes(),
            },
            'plot': self.serialDataView._get_settings_dict(),
            'serial': {
                'baud_rate': self.toolBar.baudRates.currentText(),
                'data_bits': self.toolBar.dataBits.currentIndex(),
                'parity': self.toolBar._parity.currentIndex(),
                'stop_bits': self.toolBar.stopBits.currentIndex(),
                'flow_control': self.toolBar._flowControl.currentIndex(),
            },
            'macros': self.serialSendView._get_macros_list('serial'),
            'macros_can': self.serialSendView._get_macros_list('can'),
        }
        target = path or SETTINGS_FILE
        # Write to a temp file in the same directory, then atomically replace the
        # target. A crash / full disk mid-write leaves the old settings intact
        # instead of truncating them to an unparseable file.
        tmp = target + '.tmp'
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2)
            os.replace(tmp, target)
            return True
        except OSError as e:
            try:
                os.remove(tmp)
            except OSError:
                pass
            # A full disk / read-only folder previously failed silently - the
            # user lost settings with no indication anywhere.
            self.statusText.setText(f'Settings save failed: {e}')
            return False

    @staticmethod
    def _restore_combo_text(combo, text):
        """Set a combo to text only if it is an existing item (setCurrentText
        on a non-editable combo is a silent no-op for unknown values)."""
        idx = combo.findText(str(text))
        if idx >= 0:
            combo.setCurrentIndex(idx)
            return True
        if combo.isEditable():
            combo.setCurrentText(str(text))
            return True
        return False

    def load_all_settings(self, path=None):
        """Load all settings from one JSON file. Returns False on failure."""
        try:
            with open(path or SETTINGS_FILE, 'r') as f:
                s = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError, UnicodeDecodeError):
            return False
        if not isinstance(s, dict):
            return False

        # Window geometry / splitter sizes. We restore the size/position but
        # never reopen maximized or full-screen -- the app should always start
        # as a normal resizable window.
        win = s.get('window', {})
        try:
            geo = win.get('geometry', '')
            if geo:
                ok = self.restoreGeometry(QtCore.QByteArray.fromHex(geo.encode('ascii')))
                if ok:
                    self.setWindowState(self.windowState() & ~(
                        QtCore.Qt.WindowMaximized | QtCore.Qt.WindowFullScreen))
                    # Only mark restored on success so main() still applies the
                    # default size when the stored geometry is corrupt.
                    self._geometry_restored = True
            sizes = win.get('splitter')
            if sizes:
                self._splitter_sizes_to_restore = [int(x) for x in sizes]
        except (TypeError, ValueError):
            pass

        # Dark mode
        self._dark_mode = s.get('dark_mode', False)
        self._dark_mode_action.setChecked(self._dark_mode)
        if self._dark_mode:
            self._apply_dark_palette()
        else:
            self._apply_light_palette()

        # Auto-reconnect
        self._auto_reconnect = s.get('auto_reconnect', True)
        self._auto_reconnect_action.setChecked(self._auto_reconnect)

        # Display timestamps
        self.serialDataView.show_timestamps = bool(s.get('show_timestamps', False))
        self._timestamps_action.setChecked(self.serialDataView.show_timestamps)

        # Plot/decode settings
        plot = s.get('plot', s)  # fallback: old format had plot keys at top level
        self.serialDataView._load_plot_settings(plot)

        # Connection parameters are only applied while disconnected: the open
        # port/bus keeps running with its original parameters, so rewriting
        # the combos would make the toolbar (and stats bar) lie about the
        # actual connection.
        connected = self.port.isOpen() or self._can_bus is not None \
            or self._can_opener is not None
        if connected:
            self.statusText.setText(
                'Connection settings not applied while connected')

        # Serial port settings (block signals to avoid cascading saves)
        serial = s.get('serial', {})
        if serial and not connected:
            for w in [self.toolBar.baudRates, self.toolBar.dataBits,
                      self.toolBar._parity, self.toolBar.stopBits, self.toolBar._flowControl]:
                w.blockSignals(True)
            self._restore_combo_text(self.toolBar.baudRates, serial.get('baud_rate', '115200'))
            self.toolBar.dataBits.setCurrentIndex(serial.get('data_bits', 3))
            self.toolBar._parity.setCurrentIndex(serial.get('parity', 0))
            self.toolBar.stopBits.setCurrentIndex(serial.get('stop_bits', 0))
            self.toolBar._flowControl.setCurrentIndex(serial.get('flow_control', 0))
            for w in [self.toolBar.baudRates, self.toolBar.dataBits,
                      self.toolBar._parity, self.toolBar.stopBits, self.toolBar._flowControl]:
                w.blockSignals(False)

        # CAN settings
        can_cfg = s.get('can', {})
        if isinstance(can_cfg, dict) and can_cfg and not connected:
            for w in [self.toolBar.canInterfaces, self.toolBar.canChannels,
                      self.toolBar.canBitrates]:
                w.blockSignals(True)
            self._restore_combo_text(self.toolBar.canInterfaces,
                                     can_cfg.get('interface', 'Kvaser'))
            try:
                idx = int(can_cfg.get('channel', 0))
                idx = max(0, min(idx, self.toolBar.canChannels.count() - 1))
                self.toolBar.canChannels.setCurrentIndex(idx)
            except (TypeError, ValueError):
                pass
            self._restore_combo_text(self.toolBar.canBitrates,
                                     str(can_cfg.get('bitrate', '500 kbit/s')))
            for w in [self.toolBar.canInterfaces, self.toolBar.canChannels,
                      self.toolBar.canBitrates]:
                w.blockSignals(False)
        if isinstance(can_cfg, dict):
            view_mode = can_cfg.get('view_mode', 'Scrolling')
            if view_mode in ('Scrolling', 'Fixed'):
                self.canView.view_mode_combo.setCurrentText(view_mode)
            self.canView.flash_check.setChecked(bool(can_cfg.get('flash', True)))
            decode = can_cfg.get('decode', 'Auto')
            if decode in CAN_DECODE_MODES:
                self.canView.decode_combo.setCurrentText(decode)

        # UI mode last, so switching applies over the loaded CAN settings
        # (never while connected: the switch would close the live connection)
        ui_mode = s.get('ui_mode', 'Serial')
        if ui_mode in ('Serial', 'CAN') and not connected:
            # setCurrentText triggers _on_ui_mode_changed, which updates all
            # visibility; setting it to the current value triggers nothing,
            # which is fine since 'Serial' is the initial state.
            self.toolBar.modeCombo.setCurrentText(ui_mode)

        # Macros (one set per mode; loaded after the UI mode switch above so
        # whichever set is on screen is the one that gets refreshed)
        macros = s.get('macros', None)
        if macros:
            self.serialSendView._load_macros_from_list(macros, 'serial')
        macros_can = s.get('macros_can', None)
        if macros_can:
            self.serialSendView._load_macros_from_list(macros_can, 'can')

        # Apply splitter sizes after the plot widget (if any) has been created
        if self._splitter_sizes_to_restore:
            sizes = self._splitter_sizes_to_restore
            self._splitter_sizes_to_restore = None
            if len(sizes) == self.serialDataView.splitter.count():
                self.serialDataView.splitter.setSizes(sizes)
        return True

    def _menu_save_settings(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Settings', '', 'JSON Files (*.json);;All Files (*)')
        if path:
            if self.save_all_settings(path):
                self.statusText.setText(f'Settings saved to {os.path.basename(path)}')
            # on failure save_all_settings already shows the reason

    def _menu_load_settings(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Load Settings', '', 'JSON Files (*.json);;All Files (*)')
        if not path:
            return
        if not self.load_all_settings(path):
            self.statusText.setText(
                f'Failed to load settings from {os.path.basename(path)} '
                '(unreadable or not a settings file)')
            return
        note = ''
        if self.port.isOpen() or self._can_bus is not None:
            note = ' (connection settings skipped while connected)'
        self.statusText.setText(
            f'Settings loaded from {os.path.basename(path)}{note}')

    def _menu_export_csv(self):
        dv = self.serialDataView
        if not dv.plot_data:
            self.statusText.setText('No plot data to export')
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export CSV', '', 'CSV Files (*.csv);;All Files (*)')
        if not path:
            return
        try:
            def quote(name):
                if any(c in name for c in ',"\n'):
                    return '"' + name.replace('"', '""') + '"'
                return name

            dv._update_math_channels()  # make math values current, not up to a frame stale
            n_channels = len(dv.plot_data)
            n_points = len(dv.plot_data[0])
            # Skip the NaN-prefilled head: export only rows that hold real samples
            start = max(0, n_points - dv._plot_fill)
            # Build header: regular channels + math channels
            headers = [quote(dv._channel_name(i)) for i in range(n_channels)]
            for mch in dv._math_channels:
                headers.append(quote(mch.get('name', 'Math')))
            header = ','.join(headers)
            lines = [header]
            for row in range(start, n_points):
                values = [str(dv.plot_data[ch][row]) for ch in range(n_channels)]
                for arr in dv._math_data:
                    values.append(str(arr[row]) if row < len(arr) else '')
                lines.append(','.join(values))
            with open(path, 'w', encoding='utf-8', newline='') as f:
                f.write('\n'.join(lines) + '\n')
            self.statusText.setText(f'CSV exported to {os.path.basename(path)}')
        except (OSError, UnicodeError) as e:
            self.statusText.setText(f'Export failed: {e}')

    def _menu_export_png(self):
        dv = self.serialDataView
        if dv.graphWidget is None:
            self.statusText.setText('No plot to export (enable Show Plot)')
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export PNG', '', 'PNG Files (*.png);;All Files (*)')
        if not path:
            return
        try:
            from pyqtgraph.exporters import ImageExporter
            exporter = ImageExporter(dv.graphWidget.plotItem)
            exporter.export(path)
            self.statusText.setText(f'PNG exported to {os.path.basename(path)}')
        except Exception as e:
            self.statusText.setText(f'Export failed: {e}')


class SerialDataView(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.plot_lines = []
        self.plot_data = []
        self._plot_fill = 0        # how many trailing samples in plot_data are real
        self._pending_rows = []    # parsed sample rows (ASCII/frame) awaiting display flush
        self._pending_blocks = []  # numpy (n, nch) blocks (binary) awaiting display flush
        self._ascii_line_buffer = ''  # partial ASCII line awaiting its newline
        self.graphWidget = None
        self.channel_names = {}   # {index: 'custom name'} for renamed channels
        self.channel_colors = {}  # {index: '#hex'} for custom channel colors
        self.channel_axes = {}    # {index: 1 or 2} axis assignment (1=left, 2=right)
        self.channel_scale = {}   # {index: float} gain applied to plotted value
        self.channel_offset = {}  # {index: float} offset added after gain
        self.channel_units = {}   # {index: 'unit'} shown in legend/crosshair
        self.hidden_channels = set()  # set of channel indices toggled off
        self._scale_vec = None    # cached (nch,) scale array; None until built
        self._offset_vec = None
        self._graph_container = None
        self._channel_toggle_bar = None
        self._y2_viewbox = None
        self._y2_plot_lines = {}  # {channel_index: PlotDataItem} for Y2 channels
        self._y_auto_scale = True
        self._y_min = -1.0
        self._y_max = 1.0
        # Tracks whether the previous received chunk ended in '\r'. Used to
        # suppress a '\n' that begins the next chunk when CRLF is split across
        # serial reads (otherwise Qt renders a blank line between messages).
        self._pending_cr = False
        self._hex_col = 0  # tracks hex pairs on current line (0..15)

        # Display timestamps (View > Timestamps): prefix each received line
        # in the ASCII view with its arrival time. _ts_at_line_start tracks
        # line boundaries across arbitrarily-chunked reads.
        self.show_timestamps = False
        self._ts_at_line_start = True

        # FFT view state
        self._fft_widget = None
        self._fft_lines = []
        self._fft_update_counter = 0
        self._fft_window = None
        self._fft_window_n = 0
        self._fft_scale = 1.0

        # Math/computed channels
        self._math_channels = []   # list of {'name': str, 'expression': str}
        self._math_lines = []      # PlotDataItem list
        self._math_data = []       # numpy array list
        self._math_errors = set()  # indices of channels with eval errors
        self._math_expr_cache = {}  # {expression: compiled code or None if rejected}

        self._binary_reader = BinaryStreamReader()
        self._frame_reader = FrameReader()

        # X-axis time mode: when enabled, the bottom axis is relabeled in
        # seconds using a measured sample rate (samples are still stored by
        # index; only the axis tick scale changes, so nothing in the data path
        # or crosshair logic has to change).
        self._x_time_mode = False
        self._x_sample_total = 0      # samples since the last clear/start
        self._x_start_time = None     # monotonic time of first sample
        self._x_rate = 0.0            # measured samples/sec

        # Pause / Trigger state
        self._plot_paused = False
        self._trigger_enabled = False
        self._trigger_armed = False
        self._trigger_channel = 0
        self._trigger_level = 0.0
        self._trigger_edge = 'rising'
        self._trigger_countdown = -1
        self._trigger_prev_value = None  # previous value on trigger channel

        self.serialData = QtWidgets.QTextEdit(self)
        self.serialData.setReadOnly(True)
        self.serialData.setUndoRedoEnabled(False)
        self.serialData.setFontFamily('Segoe UI')
        self.serialData.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.serialDataHex = QtWidgets.QTextEdit(self)
        self.serialDataHex.setReadOnly(True)
        self.serialDataHex.setUndoRedoEnabled(False)
        self.serialDataHex.setFontFamily('Segoe UI')
        self.serialDataHex.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.label_data_flow = QtWidgets.QLabel('Data: HEX')
        self.label_data_flow.setObjectName('sectionLabel')
        self.label_data_flow.setFont(QtGui.QFont('Segoe UI', 10))
        self.label_data_flow.setIndent(5)

        self.label_sent_data = QtWidgets.QLabel('Data: ASCII')
        self.label_sent_data.setObjectName('sectionLabel')
        self.label_sent_data.setFont(QtGui.QFont('Segoe UI', 10))
        self.label_sent_data.setIndent(5)

        self.graph_mode = QCheckBox("Show Plot")
        self.graph_mode.setFont(QtGui.QFont('Segoe UI', 10))
        self.graph_mode.stateChanged.connect(self.graph_state_changed)

        self._fft_check = QCheckBox("Show FFT")
        self._fft_check.setFont(QtGui.QFont('Segoe UI', 10))
        self._fft_check.stateChanged.connect(self._fft_state_changed)
        self._fft_check.stateChanged.connect(lambda: self._save_settings())

        self._time_axis_check = QCheckBox("Time X")
        self._time_axis_check.setFont(QtGui.QFont('Segoe UI', 10))
        self._time_axis_check.setToolTip(
            'Label the X axis in seconds using the measured sample rate\n'
            '(instead of the sample index)')
        self._time_axis_check.toggled.connect(self.set_x_time_mode)

        self.graph_channels = QSpinBox(minimum=1, maximum=12, value=4, prefix="Ch: ")
        self.graph_channels.setFont(QtGui.QFont('Segoe UI', 10))
        self.graph_channels.valueChanged.connect(self._on_channels_changed)

        self.data_mode = QtWidgets.QComboBox()
        self.data_mode.addItems(['ASCII', 'Binary Stream', 'Custom Frame'])
        self.data_mode.setFont(QtGui.QFont('Segoe UI', 10))
        self.data_mode.setMinimumWidth(130)
        self.data_mode.currentIndexChanged.connect(self._on_mode_changed)

        self._pts_label = QtWidgets.QLabel('Pts:')
        self._pts_label.setFont(QtGui.QFont('Segoe UI', 10))
        self.plot_length_spin = QSpinBox(minimum=10, maximum=10000, value=DEFAULT_PLOT_LENGTH, singleStep=50)
        self.plot_length_spin.setFont(QtGui.QFont('Segoe UI', 10))
        self.plot_length_spin.valueChanged.connect(self._on_pts_changed)

        self.clear_button = QtWidgets.QPushButton('Clear ALL')
        self.clear_button.setFont(QtGui.QFont('Segoe UI', 10))
        self.clear_button.clicked.connect(self.clear_button_Clicked)
        self.clear_button.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)

        self.label = QLabel(self)
        self.label.setPixmap(create_connector_pixmap(DISCONNECTED_COLOR))

        self.converter_label = QtWidgets.QLabel('Converter')
        self.converter_label.setObjectName('sectionLabel')
        self.converter_label.setFont(QtGui.QFont('Segoe UI', 10))
        self.converter_label.setIndent(5)

        self.convert_A_type = QtWidgets.QComboBox(self)
        self.convert_A_type.addItems(list(CONVERTERS.keys()))
        self.convert_A_type.setCurrentIndex(0)
        self.convert_A_type.setMinimumHeight(30)
        self.convert_A_type.setFont(QtGui.QFont('Segoe UI', 10))
        self.convert_A_type.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.convert_A_type.currentIndexChanged.connect(self.translate_data)

        self.convert_A_text = QtWidgets.QTextEdit(self)
        self.convert_A_text.setMaximumHeight(31)
        self.convert_A_text.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.convert_A_text.textChanged.connect(self.translate_data)
        self.convert_A_text.setFont(QtGui.QFont('Segoe UI', 10))

        self.convert_arrow = QtWidgets.QLabel('\u2192')
        self.convert_arrow.setFont(QtGui.QFont('Segoe UI', 14))
        self.convert_arrow.setAlignment(QtCore.Qt.AlignCenter)
        self.convert_arrow.setFixedWidth(20)

        self.convert_B_text = QtWidgets.QTextEdit(self)
        self.convert_B_text.setMaximumHeight(31)
        self.convert_B_text.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.convert_B_text.setFont(QtGui.QFont('Segoe UI', 10))

        # --- All decode/plot widgets in one row ---

        # Type combo (binary/frame modes)
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItems(list(DATA_TYPES.keys()))
        self.type_combo.setCurrentText('float32')
        self.type_combo.setMinimumWidth(100)
        self.type_combo.currentTextChanged.connect(self._on_setting_changed)

        # Delimiter (ASCII mode)
        self.delimiter_combo = QtWidgets.QComboBox()
        self.delimiter_combo.addItems(['Auto', 'Comma', 'Semicolon', 'Space', 'Tab', 'Other'])
        self.delimiter_combo.setMinimumWidth(100)
        self.delimiter_combo.currentIndexChanged.connect(self._on_delimiter_changed)

        self.delimiter_custom = QtWidgets.QLineEdit()
        self.delimiter_custom.setMaximumWidth(50)
        self.delimiter_custom.setPlaceholderText('...')
        self.delimiter_custom.setVisible(False)
        self.delimiter_custom.editingFinished.connect(self._on_setting_changed)

        # Endianness (binary/frame)
        self.endian_combo = QtWidgets.QComboBox()
        self.endian_combo.addItems(['Little Endian', 'Big Endian'])
        self.endian_combo.setMinimumWidth(110)
        self.endian_combo.currentTextChanged.connect(self._on_setting_changed)

        # Binary-only: sync button
        self.sync_button = QtWidgets.QPushButton('Sync')
        self.sync_button.setToolTip('Clear byte buffer to re-align stream')
        self.sync_button.clicked.connect(self._on_sync_clicked)

        # Frame-only: frame start
        self._frame_start_label = QtWidgets.QLabel('Start byte [hex]:')
        self.sync_word_edit = QtWidgets.QLineEdit('AA')
        self.sync_word_edit.setMaximumWidth(80)
        self.sync_word_edit.setPlaceholderText('AA BB')
        self.sync_word_edit.editingFinished.connect(self._on_setting_changed)

        # Frame-only: payload size
        self._payload_size_label = QtWidgets.QLabel('Payload Size:')
        self.size_field_combo = QtWidgets.QComboBox()
        self.size_field_combo.addItems(['Fixed', '1-byte size field', '2-byte size field'])
        self.size_field_combo.setMinimumWidth(70)
        self.size_field_combo.currentTextChanged.connect(self._on_size_field_changed)

        self.frame_size_spin = QtWidgets.QSpinBox(minimum=1, maximum=65535, value=12)
        self.frame_size_spin.valueChanged.connect(self._on_setting_changed)

        # Frame-only: checksum
        self.checksum_check = QtWidgets.QCheckBox('Checksum')
        self.checksum_check.stateChanged.connect(self._on_setting_changed)

        self._frame_only_widgets = [
            self._frame_start_label, self.sync_word_edit,
            self._payload_size_label, self.size_field_combo,
            self.frame_size_spin, self.checksum_check,
        ]
        self._binary_frame_widgets = [
            self.type_combo, self.endian_combo,
        ]

        # Single controls row: left = decoding, right = Pts + Show Plot
        controls = QtWidgets.QWidget()
        cl = QtWidgets.QHBoxLayout(controls)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(4)
        row_widgets = [
            self.data_mode, self.type_combo, self.delimiter_combo,
            self.delimiter_custom, self.graph_channels, self.endian_combo,
            self.sync_button, self.sync_word_edit, self.size_field_combo,
            self.frame_size_spin, self.checksum_check, self._pts_label,
            self.plot_length_spin, self.graph_mode, self._fft_check,
            self._frame_start_label, self._payload_size_label,
        ]
        row_font = QtGui.QFont('Segoe UI', 10)
        for w in row_widgets:
            w.setFixedHeight(30)
            w.setFont(row_font)
        # Left: decoding group
        cl.addWidget(self.data_mode)
        cl.addWidget(self.type_combo)
        cl.addWidget(self.delimiter_combo)
        cl.addWidget(self.delimiter_custom)
        cl.addWidget(self.graph_channels)
        cl.addWidget(self.endian_combo)
        cl.addWidget(self.sync_button)
        cl.addWidget(self._frame_start_label)
        cl.addWidget(self.sync_word_edit)
        cl.addWidget(self._payload_size_label)
        cl.addWidget(self.size_field_combo)
        cl.addWidget(self.frame_size_spin)
        cl.addWidget(self.checksum_check)
        # Middle: trigger controls
        cl.addStretch()

        self._trigger_check = QCheckBox("Trigger")
        self._trigger_check.setFont(row_font)
        self._trigger_check.setFixedHeight(30)
        self._trigger_check.stateChanged.connect(self._on_trigger_toggled)

        self._trigger_ch_spin = QSpinBox(minimum=0, maximum=11, value=0, prefix="Ch: ")
        self._trigger_ch_spin.setFont(row_font)
        self._trigger_ch_spin.setFixedHeight(30)
        self._trigger_ch_spin.valueChanged.connect(self._on_trigger_setting_changed)

        self._trigger_level_edit = QtWidgets.QLineEdit("0.0")
        self._trigger_level_edit.setFont(row_font)
        self._trigger_level_edit.setFixedHeight(30)
        self._trigger_level_edit.setFixedWidth(70)
        self._trigger_level_edit.setPlaceholderText("Level")
        self._trigger_level_edit.editingFinished.connect(self._on_trigger_setting_changed)

        self._trigger_edge_combo = QtWidgets.QComboBox()
        self._trigger_edge_combo.addItems(['Rising', 'Falling'])
        self._trigger_edge_combo.setFont(row_font)
        self._trigger_edge_combo.setFixedHeight(30)
        self._trigger_edge_combo.currentIndexChanged.connect(self._on_trigger_setting_changed)

        self._trigger_rearm_btn = QtWidgets.QPushButton("Re-arm")
        self._trigger_rearm_btn.setFont(row_font)
        self._trigger_rearm_btn.setFixedHeight(30)
        self._trigger_rearm_btn.clicked.connect(self._on_trigger_rearm)

        cl.addWidget(self._trigger_check)
        cl.addWidget(self._trigger_ch_spin)
        cl.addWidget(self._trigger_level_edit)
        cl.addWidget(self._trigger_edge_combo)
        cl.addWidget(self._trigger_rearm_btn)

        # Initially hide trigger detail widgets
        self._trigger_detail_widgets = [
            self._trigger_ch_spin, self._trigger_level_edit,
            self._trigger_edge_combo, self._trigger_rearm_btn,
        ]
        for w in self._trigger_detail_widgets:
            w.setVisible(False)

        # Separator before plot controls
        _trig_sep = QtWidgets.QFrame()
        _trig_sep.setFrameShape(QtWidgets.QFrame.VLine)
        _trig_sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        cl.addWidget(_trig_sep)

        # Math channels button
        self._math_btn = QtWidgets.QPushButton("Math")
        self._math_btn.setFont(row_font)
        self._math_btn.setFixedHeight(30)
        self._math_btn.setToolTip("Configure math/computed channels")
        self._math_btn.clicked.connect(self._open_math_dialog)
        cl.addWidget(self._math_btn)

        # Freeze display: scroll back / read / search while RX capture,
        # logging and stats keep running (buffered data renders on resume)
        self.freeze_btn = QtWidgets.QPushButton('Freeze')
        self.freeze_btn.setCheckable(True)
        self.freeze_btn.setFont(row_font)
        self.freeze_btn.setFixedHeight(30)
        self.freeze_btn.setToolTip(
            'Freeze the data views to scroll back and read.\n'
            'Capture, logging and stats keep running; buffered\n'
            'data is shown when unfrozen (display timestamps on\n'
            'that backlog show the unfreeze time - the log file\n'
            'keeps the true arrival times).')
        cl.addWidget(self.freeze_btn)

        # Right: plot controls
        cl.addWidget(self._pts_label)
        cl.addWidget(self.plot_length_spin)
        cl.addWidget(self.graph_mode)
        cl.addWidget(self._fft_check)
        cl.addWidget(self._time_axis_check)

        # Vertical splitter: graph (top) | data views (bottom)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        data_panel = QtWidgets.QWidget()
        dp_layout = QtWidgets.QGridLayout(data_panel)
        dp_layout.setContentsMargins(0, 0, 0, 0)
        dp_layout.addWidget(self.label_sent_data,   0, 0, 1, 3)
        dp_layout.addWidget(self.label_data_flow,   0, 3, 1, 3)
        dp_layout.addWidget(self.serialData,         1, 0, 1, 3)
        dp_layout.addWidget(self.serialDataHex,      1, 3, 1, 3)

        self.splitter.addWidget(data_panel)

        # Search bar (toggled via Ctrl+F)
        self.search_bar = QtWidgets.QWidget()
        self.search_bar.setVisible(False)
        sb_layout = QtWidgets.QHBoxLayout(self.search_bar)
        sb_layout.setContentsMargins(0, 2, 0, 2)
        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText('Search...')
        self.search_input.setFont(QtGui.QFont('Segoe UI', 10))
        self.search_input.returnPressed.connect(self._do_search)
        self.search_count = QtWidgets.QLabel('')
        self.search_count.setFont(QtGui.QFont('Segoe UI', 10))
        find_btn = QtWidgets.QPushButton('Find')
        find_btn.setFont(QtGui.QFont('Segoe UI', 10))
        find_btn.clicked.connect(self._do_search)
        clear_search_btn = QtWidgets.QPushButton('Clear')
        clear_search_btn.setFont(QtGui.QFont('Segoe UI', 10))
        clear_search_btn.clicked.connect(self._clear_search)
        sb_layout.addWidget(self.search_input)
        sb_layout.addWidget(find_btn)
        sb_layout.addWidget(clear_search_btn)
        sb_layout.addWidget(self.search_count)
        sb_layout.addStretch()

        self.setLayout(QtWidgets.QGridLayout())
        self.layout().addWidget(controls,               0, 0, 1, 7)
        self.layout().addWidget(self.search_bar,        1, 0, 1, 7)
        self.layout().addWidget(self.splitter,          2, 0, 1, 7)
        self.layout().addWidget(self.converter_label,   3, 1, 1, 1)
        self.layout().addWidget(self.label,             4, 0, 1, 1)
        self.layout().addWidget(self.convert_A_type,    4, 1, 1, 1)
        self.layout().addWidget(self.convert_A_text,    4, 2, 1, 1)
        self.layout().addWidget(self.convert_arrow,     4, 3, 1, 1)
        self.layout().addWidget(self.convert_B_text,    4, 4, 1, 2)
        self.layout().addWidget(self.clear_button,      4, 6, 1, 1, alignment=QtCore.Qt.AlignRight)
        self.layout().setRowStretch(2, 1)
        self.layout().setContentsMargins(2, 2, 2, 2)

        # Apply initial mode visibility (first run has no settings file, so
        # nothing else triggers it - binary/frame widgets showed in ASCII mode)
        self._on_mode_changed()

    def _channel_name(self, index):
        """Return custom name for a channel, or default 'Ch N'."""
        return self.channel_names.get(index, f'Ch {index}')

    def _channel_display(self, index):
        """Channel name with its unit suffix, e.g. 'Voltage (V)'. Used for
        legend and crosshair; the bare name is kept for CSV headers/renames."""
        name = self._channel_name(index)
        unit = self.channel_units.get(index, '')
        return f'{name} ({unit})' if unit else name

    def _channel_color(self, index):
        """Return custom color for a channel, or default from PLOT_COLORS."""
        return self.channel_colors.get(index, PLOT_COLORS[index % len(PLOT_COLORS)])

    def _invalidate_scale_cache(self):
        """Force the scale/offset vectors to rebuild on the next flush."""
        self._scale_vec = None
        self._offset_vec = None

    def _build_scale_vectors(self, nch):
        """Build (or reuse) the per-channel scale/offset arrays for nch channels."""
        if (self._scale_vec is not None and len(self._scale_vec) == nch):
            return self._scale_vec, self._offset_vec
        if not self.channel_scale and not self.channel_offset:
            self._scale_vec = None
            self._offset_vec = None
            return None, None
        self._scale_vec = np.array(
            [self.channel_scale.get(i, 1.0) for i in range(nch)], dtype=np.float64)
        self._offset_vec = np.array(
            [self.channel_offset.get(i, 0.0) for i in range(nch)], dtype=np.float64)
        return self._scale_vec, self._offset_vec

    def _create_channel_toggle_bar(self):
        """Create a horizontal bar of channel toggle checkboxes below the graph."""
        self._channel_toggle_bar = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(self._channel_toggle_bar)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(10)
        self._populate_channel_toggles(layout)

    def _rebuild_channel_toggles(self):
        """Rebuild channel toggle checkboxes when channel count or names change."""
        if self._channel_toggle_bar is None:
            return
        layout = self._channel_toggle_bar.layout()
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._populate_channel_toggles(layout)

    def _populate_channel_toggles(self, layout):
        """Fill a layout with one toggle checkbox per channel."""
        dark = getattr(self.window(), '_dark_mode', False)
        label_color = '#ffffff' if dark else '#000000'
        n = self.graph_channels.value()
        for i in range(n):
            name = self._channel_name(i)
            color = self._channel_color(i)
            cb = QCheckBox(name)
            cb.setFont(QtGui.QFont('Segoe UI', 9, QtGui.QFont.Bold))
            cb.setStyleSheet(
                f'QCheckBox {{ color: {label_color}; }}'
                f'QCheckBox::indicator:checked {{ background-color: {color}; border: 1px solid #888; }}'
                f'QCheckBox::indicator:unchecked {{ background-color: #ffffff; border: 1px solid #888; }}')
            cb.setChecked(i not in self.hidden_channels)
            cb.toggled.connect(lambda checked, idx=i: self._on_channel_toggled(idx, checked))
            layout.addWidget(cb)
        layout.addStretch()

    def _on_channel_toggled(self, index, checked):
        """Show or hide a channel when its toggle checkbox changes."""
        if checked:
            self.hidden_channels.discard(index)
        else:
            self.hidden_channels.add(index)
        # Update plot line visibility
        if index < len(self.plot_lines):
            self.plot_lines[index].setVisible(checked)
            if checked and index < len(self.plot_data):
                # Hidden channels skip setData during streaming; refresh on show
                self.plot_lines[index].setData(self.plot_data[index])
        # Update Y2 line visibility
        if index in self._y2_plot_lines:
            self._y2_plot_lines[index].setVisible(checked)
        # Rebuild legend to only show visible channels
        if self.graphWidget is not None and self.graphWidget.plotItem.legend is not None:
            legend = self.graphWidget.plotItem.legend
            legend.clear()
            for i, line in enumerate(self.plot_lines):
                if i not in self.hidden_channels:
                    legend.addItem(line, self._channel_display(i))
            for i, mch in enumerate(self._math_channels):
                if i < len(self._math_lines):
                    legend.addItem(self._math_lines[i], mch.get('name', f'Math {i}'))
        # Update FFT line visibility
        if index < len(self._fft_lines):
            self._fft_lines[index].setVisible(checked)
        self._save_settings()

    def _setup_y2_axis(self):
        """Create a second Y-axis viewbox linked to the main plot."""
        if self._y2_viewbox is not None:
            return
        self._y2_viewbox = pg.ViewBox()
        self.graphWidget.plotItem.showAxis('right')
        self.graphWidget.plotItem.scene().addItem(self._y2_viewbox)
        self.graphWidget.plotItem.getAxis('right').linkToView(self._y2_viewbox)
        self._y2_viewbox.setXLink(self.graphWidget.plotItem)
        # Sync geometry now and on resize
        self._sync_y2_viewbox()
        self.graphWidget.plotItem.vb.sigResized.connect(self._sync_y2_viewbox)
        # Style right axis to match theme
        dark = getattr(self.window(), '_dark_mode', False)
        axis_color = '#ffffff' if dark else '#000000'
        self.graphWidget.plotItem.getAxis('right').setPen(pg.mkPen(color=axis_color))
        self.graphWidget.plotItem.getAxis('right').setTextPen(pg.mkPen(color=axis_color))
        self.graphWidget.plotItem.getAxis('right').setLabel('Y2')
        self._y2_viewbox.enableAutoRange(axis='y')

    def _sync_y2_viewbox(self):
        """Keep Y2 viewbox geometry in sync with the main viewbox."""
        if self._y2_viewbox and self.graphWidget:
            self._y2_viewbox.setGeometry(self.graphWidget.plotItem.vb.sceneBoundingRect())

    def _remove_y2_axis(self):
        """Remove the second Y-axis viewbox and hide the right axis."""
        if self._y2_viewbox and self.graphWidget:
            try:
                self.graphWidget.plotItem.vb.sigResized.disconnect(self._sync_y2_viewbox)
            except (TypeError, RuntimeError):
                pass
            self.graphWidget.plotItem.scene().removeItem(self._y2_viewbox)
            self.graphWidget.plotItem.hideAxis('right')
        self._y2_viewbox = None
        self._y2_plot_lines = {}

    def _has_y2_channels(self):
        """Check if any current channel is assigned to Y2."""
        n = self.graph_channels.value()
        return any(self.channel_axes.get(i, 1) == 2 for i in range(n))

    def _create_plot_lines(self):
        """Create plot lines based on the current channel count spinbox."""
        n = self.graph_channels.value()
        plot_length = self.plot_length_spin.value()
        self.plot_lines = []
        self.plot_data = []
        # Remove old Y2 lines
        if self._y2_viewbox:
            for line in self._y2_plot_lines.values():
                self._y2_viewbox.removeItem(line)
        self._y2_plot_lines = {}
        # Setup or remove Y2 axis as needed
        if self._has_y2_channels():
            self._setup_y2_axis()
        else:
            self._remove_y2_axis()
        self._plot_fill = 0
        self._pending_rows = []
        self._pending_blocks = []
        self._invalidate_scale_cache()
        self._reset_sample_rate()
        for i in range(n):
            color = self._channel_color(i)
            # NaN-prefilled: curves start empty instead of as a flat zero line
            arr = np.full(plot_length, np.nan)
            axis = self.channel_axes.get(i, 1)
            if axis == 2 and self._y2_viewbox is not None:
                # Dashed pen for Y2 channels
                pen = pg.mkPen(color, width=2, style=QtCore.Qt.DashLine)
                line = pg.PlotDataItem(pen=pen, connect='finite')
                self._y2_viewbox.addItem(line)
                self._y2_plot_lines[i] = line
            else:
                pen = pg.mkPen(color, width=2)
                line = pg.PlotDataItem(pen=pen, connect='finite')
                self.graphWidget.plotItem.addItem(line)
            # Draw at most ~2 points per screen pixel while keeping spikes visible
            line.setDownsampling(auto=True, method='peak')
            # Add visible channels to legend; apply visibility
            visible = i not in self.hidden_channels
            if self.graphWidget.plotItem.legend is not None and visible:
                self.graphWidget.plotItem.legend.addItem(line, self._channel_display(i))
            line.setVisible(visible)
            line.setData(arr)
            self.plot_lines.append(line)
            self.plot_data.append(arr)
        # Rebuild math channel lines too
        self._rebuild_math_lines()

    def _on_channels_changed(self):
        """Rebuild plot lines when channel count changes while graph is active."""
        # Remove hidden state for channels beyond the new count
        n = self.graph_channels.value()
        self.hidden_channels = {i for i in self.hidden_channels if i < n}
        if self.graphWidget is not None:
            for line in self.plot_lines:
                self.graphWidget.plotItem.removeItem(line)
            # Also remove Y2 lines from their viewbox
            if self._y2_viewbox:
                for line in self._y2_plot_lines.values():
                    self._y2_viewbox.removeItem(line)
                self._y2_plot_lines = {}
            if self.graphWidget.plotItem.legend is not None:
                self.graphWidget.plotItem.legend.clear()
            self._create_plot_lines()
        # Rebuild FFT lines if FFT widget exists
        if self._fft_widget is not None:
            self._create_fft_lines()
        self._rebuild_channel_toggles()
        self._apply_reader_settings()
        self._save_settings()

    def graph_state_changed(self):
        if self.graph_mode.isChecked():
            _ensure_pg()
            self.graphWidget = pg.PlotWidget()
            self.graphWidget.setBackground('#FFFFFFFF')
            self.graphWidget.setMinimumHeight(150)
            self.graphWidget.plotItem.getAxis('bottom').setPen(pg.mkPen(color='#000000'))
            self.graphWidget.plotItem.getAxis('left').setPen(pg.mkPen(color='#000000'))
            self.graphWidget.plotItem.showGrid(True, True, 0.3)
            # Force 3-button mouse mode (left=pan, right=menu)
            self.graphWidget.plotItem.vb.setMouseMode(pg.ViewBox.PanMode)
            self.graphWidget.plotItem.vb.setMouseEnabled(x=True, y=False)
            # Disable axis-edge hover zoom
            self.graphWidget.plotItem.getAxis('bottom').setStyle(tickLength=5)
            self.graphWidget.plotItem.getAxis('left').setStyle(tickLength=5)
            self.graphWidget.plotItem.setClipToView(True)
            for axis_name in ('left', 'bottom', 'right', 'top'):
                ax = self.graphWidget.plotItem.getAxis(axis_name)
                ax.setAcceptedMouseButtons(QtCore.Qt.NoButton)
                ax.setAcceptHoverEvents(False)
            # Customize context menus
            for action in self.graphWidget.plotItem.ctrlMenu.actions():
                if action.text() in ('Average', 'Downsample', 'Alpha'):
                    action.setVisible(False)
            for action in self.graphWidget.plotItem.vb.menu.actions():
                if action.text() == 'View All':
                    action.setText('Reset Zoom')
            self.graphWidget.setXRange(0, self.plot_length_spin.value())
            # Restore Y-axis settings
            if self._y_auto_scale:
                self.graphWidget.enableAutoRange(axis='y')
            else:
                self.graphWidget.setYRange(self._y_min, self._y_max)
            self.graphWidget.addLegend()
            self.graphWidget.scene().sigMouseClicked.connect(self._on_plot_mouse_clicked)
            self._create_plot_lines()
            # Connect range signal AFTER setup to avoid overwriting restored Y-axis
            self.graphWidget.sigRangeChanged.connect(self._on_range_changed)
            self._ascii_line_buffer = ''
            # Crosshair cursor
            self._crosshair = pg.InfiniteLine(angle=90, movable=False,
                                               pen=pg.mkPen('#888888', width=1, style=QtCore.Qt.DashLine))
            self._crosshair.setVisible(False)
            self.graphWidget.addItem(self._crosshair)
            self._cursor_label = pg.TextItem(anchor=(0, 1), color='#000000')
            self._cursor_label.setVisible(False)
            self.graphWidget.addItem(self._cursor_label)
            self.graphWidget.scene().sigMouseMoved.connect(self._on_mouse_moved)
            # Pause button overlaid in lower-right corner (left of Clear Plot)
            self._pause_btn = QtWidgets.QPushButton('Pause', self.graphWidget)
            self._pause_btn.setStyleSheet(
                'background-color: #ffffff; border: 1px solid #aaa; padding: 2px 8px;')
            self._pause_btn.clicked.connect(self._toggle_pause)
            self._pause_btn.adjustSize()
            # Clear Plot button overlaid in lower-right corner
            self._clear_graph_btn = QtWidgets.QPushButton('Clear Plot', self.graphWidget)
            self._clear_graph_btn.setStyleSheet(
                'background-color: #ffffff; border: 1px solid #aaa; padding: 2px 8px;')
            self._clear_graph_btn.clicked.connect(self._clear_graph)
            self._clear_graph_btn.adjustSize()
            self.graphWidget.installEventFilter(self)
            # Wrap graph + channel toggle bar in a container
            self._graph_container = QtWidgets.QWidget()
            _gc_layout = QtWidgets.QVBoxLayout(self._graph_container)
            _gc_layout.setContentsMargins(0, 0, 0, 0)
            _gc_layout.setSpacing(0)
            _gc_layout.addWidget(self.graphWidget, stretch=1)
            self._create_channel_toggle_bar()
            _gc_layout.addWidget(self._channel_toggle_bar)
            self.splitter.insertWidget(0, self._graph_container)
            self._position_overlay_buttons()
            self._update_graph_theme()
            self._apply_x_axis_scale()  # honor saved time-axis mode
            # Restore the FFT view if its checkbox is on (e.g. from saved settings)
            if self._fft_check.isChecked() and self._fft_widget is None:
                self._create_fft_widget()
        else:
            # Destroy FFT widget first if it exists
            self._destroy_fft_widget()
            # Clean up Y2 axis before destroying graph (disconnects sigResized
            # and removes the ViewBox from the scene while graphWidget is valid)
            self._remove_y2_axis()
            self.graphWidget.removeEventFilter(self)
            self._channel_toggle_bar = None
            self._graph_container.setParent(None)
            self._graph_container.deleteLater()
            self._graph_container = None
            self.graphWidget = None
            self._clear_graph_btn = None
            self._pause_btn = None
            self._crosshair = None
            self._cursor_label = None
            self.plot_lines = []
            self.plot_data = []
            self._plot_fill = 0
            self._pending_rows = []
            self._pending_blocks = []
            self._math_lines = []
            self._math_data = []
            self._math_errors = set()
            self._plot_paused = False

    def _clear_graph(self):
        """Reset all plot data to empty (NaN)."""
        self._plot_fill = 0
        self._pending_rows = []
        self._pending_blocks = []
        self._reset_sample_rate()
        for arr, line in zip(self.plot_data, self.plot_lines):
            arr[:] = np.nan
            line.setData(arr)
        for arr, line in zip(self._math_data, self._math_lines):
            arr[:] = np.nan
            line.setData(arr)
        if self._y2_viewbox:
            self._y2_viewbox.enableAutoRange(axis='y')

    def _toggle_pause(self):
        """Toggle plot pause/resume."""
        self._plot_paused = not self._plot_paused
        self._update_pause_btn_style()
        self._position_overlay_buttons()

    def _update_pause_btn_style(self):
        """Update the pause button text and color to reflect current state."""
        if not hasattr(self, '_pause_btn') or self._pause_btn is None:
            return
        dark = getattr(self.window(), '_dark_mode', False)
        if self._plot_paused:
            self._pause_btn.setText('Resume')
            # No font-weight here: adjustSize() uses the QFont, so QSS-only
            # bold text would render wider than the button and clip.
            self._pause_btn.setStyleSheet(
                'background-color: #cc6600; color: #ffffff; border: 1px solid #995500; '
                'padding: 2px 8px;')
        else:
            self._pause_btn.setText('Pause')
            if dark:
                self._pause_btn.setStyleSheet(
                    'background-color: #353535; color: #ffffff; border: 1px solid #666; padding: 2px 8px;')
            else:
                self._pause_btn.setStyleSheet(
                    'background-color: #ffffff; border: 1px solid #aaa; padding: 2px 8px;')

    def _on_trigger_toggled(self):
        """Enable or disable trigger mode."""
        self._trigger_enabled = self._trigger_check.isChecked()
        for w in self._trigger_detail_widgets:
            w.setVisible(self._trigger_enabled)
        if self._trigger_enabled:
            self._trigger_armed = True
            self._trigger_countdown = -1
            self._trigger_prev_value = None
            self._on_trigger_setting_changed()
        else:
            self._trigger_armed = False
            self._trigger_countdown = -1

    def _on_trigger_setting_changed(self):
        """Update trigger parameters from UI."""
        self._trigger_channel = self._trigger_ch_spin.value()
        try:
            self._trigger_level = float(self._trigger_level_edit.text())
        except ValueError:
            self._trigger_level = 0.0
        self._trigger_edge = 'rising' if self._trigger_edge_combo.currentIndex() == 0 else 'falling'
        self._trigger_prev_value = None

    def _on_trigger_rearm(self):
        """Re-arm the trigger: resume plotting and reset trigger state."""
        self._plot_paused = False
        self._trigger_armed = True
        self._trigger_countdown = -1
        self._trigger_prev_value = None
        self._update_pause_btn_style()
        self._position_overlay_buttons()

    def _update_graph_theme(self):
        """Update graph background and axis colors based on dark mode state."""
        monitor = self.window()
        dark = getattr(monitor, '_dark_mode', False)
        if self.graphWidget is not None:
            if dark:
                self.graphWidget.setBackground('#2b2b2b')
                axis_color = '#ffffff'
            else:
                self.graphWidget.setBackground('#FFFFFF')
                axis_color = '#000000'
            self.graphWidget.plotItem.getAxis('bottom').setPen(pg.mkPen(color=axis_color))
            self.graphWidget.plotItem.getAxis('left').setPen(pg.mkPen(color=axis_color))
            self.graphWidget.plotItem.getAxis('bottom').setTextPen(pg.mkPen(color=axis_color))
            self.graphWidget.plotItem.getAxis('left').setTextPen(pg.mkPen(color=axis_color))
            if self._y2_viewbox is not None:
                self.graphWidget.plotItem.getAxis('right').setPen(pg.mkPen(color=axis_color))
                self.graphWidget.plotItem.getAxis('right').setTextPen(pg.mkPen(color=axis_color))
            if hasattr(self, '_cursor_label') and self._cursor_label is not None:
                self._cursor_label.setColor(axis_color)
            if hasattr(self, '_clear_graph_btn') and self._clear_graph_btn is not None:
                if dark:
                    self._clear_graph_btn.setStyleSheet(
                        'background-color: #353535; color: #ffffff; border: 1px solid #666; padding: 2px 8px;')
                else:
                    self._clear_graph_btn.setStyleSheet(
                        'background-color: #ffffff; border: 1px solid #aaa; padding: 2px 8px;')
            if hasattr(self, '_pause_btn') and self._pause_btn is not None:
                self._update_pause_btn_style()
            self._rebuild_channel_toggles()  # label color depends on theme
        # Update FFT widget theme
        if self._fft_widget is not None:
            if dark:
                self._fft_widget.setBackground('#2b2b2b')
                fft_axis_color = '#ffffff'
            else:
                self._fft_widget.setBackground('#FFFFFF')
                fft_axis_color = '#000000'
            self._fft_widget.plotItem.getAxis('bottom').setPen(pg.mkPen(color=fft_axis_color))
            self._fft_widget.plotItem.getAxis('left').setPen(pg.mkPen(color=fft_axis_color))
            self._fft_widget.plotItem.getAxis('bottom').setTextPen(pg.mkPen(color=fft_axis_color))
            self._fft_widget.plotItem.getAxis('left').setTextPen(pg.mkPen(color=fft_axis_color))

    def _position_overlay_buttons(self):
        """Position the Pause and Clear Plot buttons in the lower-right of the graph."""
        if self.graphWidget is None:
            return
        gw = self.graphWidget
        margin = 5
        y = gw.height() - margin
        x = gw.width() - margin
        if self._clear_graph_btn:
            btn = self._clear_graph_btn
            btn.adjustSize()
            x -= btn.width()
            btn.move(x, y - btn.height())
            x -= margin
        if hasattr(self, '_pause_btn') and self._pause_btn:
            btn = self._pause_btn
            btn.adjustSize()
            x -= btn.width()
            btn.move(x, y - btn.height())

    def eventFilter(self, obj, event):
        if obj is self.graphWidget and event.type() == QtCore.QEvent.Resize:
            self._position_overlay_buttons()
        return super().eventFilter(obj, event)

    def _on_mouse_moved(self, scene_pos):
        """Update crosshair cursor and show channel values at mouse X position."""
        if self.graphWidget is None or not self.plot_data:
            return
        vb = self.graphWidget.plotItem.vb
        if not vb.sceneBoundingRect().contains(scene_pos):
            self._crosshair.setVisible(False)
            self._cursor_label.setVisible(False)
            return
        mouse_point = vb.mapSceneToView(scene_pos)
        x = mouse_point.x()
        x_idx = int(round(x))
        n_points = len(self.plot_data[0]) if self.plot_data else 0
        if x_idx < 0 or x_idx >= n_points:
            self._crosshair.setVisible(False)
            self._cursor_label.setVisible(False)
            return
        self._crosshair.setPos(x)
        self._crosshair.setVisible(True)
        # Build value text with channel colors
        parts = []
        for i, arr in enumerate(self.plot_data):
            if i in self.hidden_channels:
                continue
            name = html.escape(self._channel_display(i))
            color = self._channel_color(i)
            val = arr[x_idx]
            val_str = f'{val:.4f}' if math.isfinite(val) else '—'
            parts.append(f'<span style="color:{color}"><b>{name}</b>: {val_str}</span>')
        for i, (mch, arr) in enumerate(zip(self._math_channels, self._math_data)):
            color = self._math_channel_color(i)
            name = html.escape(mch.get('name', f'Math {i}'))
            if x_idx < len(arr):
                val = arr[x_idx]
                val_str = f'{val:.4f}' if math.isfinite(val) else '—'
                parts.append(f'<span style="color:{color}"><b>{name}</b>: {val_str}</span>')
        label_html = '<br>'.join(parts)
        self._cursor_label.setHtml(f'<div style="background:rgba(255,255,255,200);padding:2px">{label_html}</div>')
        self._cursor_label.setPos(x, mouse_point.y())
        self._cursor_label.setVisible(True)

    def _on_range_changed(self):
        """Track Y-axis range changes and save (debounced).

        While Y auto-range is on, this fires on virtually every data update;
        there is nothing user-chosen to persist then, so skip saving entirely
        rather than rewriting the settings file at the render rate.
        """
        if self.graphWidget is None:
            return
        vb = self.graphWidget.plotItem.vb
        auto = vb.autoRangeEnabled()[1]  # [x_auto, y_auto]
        was_auto = self._y_auto_scale
        self._y_auto_scale = bool(auto)
        if auto:
            if not was_auto:
                self._save_settings()  # user just re-enabled auto-range
            return
        y_range = vb.viewRange()[1]
        self._y_min = y_range[0]
        self._y_max = y_range[1]
        self._save_settings()

    def _on_plot_mouse_clicked(self, ev):
        """Right-click on a legend entry to rename or change color."""
        if ev.button() != QtCore.Qt.RightButton:
            return
        legend = self.graphWidget.plotItem.legend
        if legend is None:
            return
        pos = ev.scenePos()
        for sample, label in legend.items:
            row_rect = sample.sceneBoundingRect().united(label.sceneBoundingRect())
            if row_rect.contains(pos):
                # Find channel index by matching the PlotDataItem
                ch_index = None
                for idx, line in enumerate(self.plot_lines):
                    if sample.item is line:
                        ch_index = idx
                        break
                if ch_index is not None:
                    self._show_channel_context_menu(ch_index, label, sample)
                    ev.accept()
                break

    def _show_channel_context_menu(self, channel_index, label, sample):
        """Show context menu for a legend entry."""
        menu = QtWidgets.QMenu(self)
        rename_action = menu.addAction('Rename...')
        color_action = menu.addAction('Change Color...')
        scale_action = menu.addAction('Scale / Offset / Units...')
        # Y-axis toggle
        current_axis = self.channel_axes.get(channel_index, 1)
        if current_axis == 1:
            axis_action = menu.addAction('Move to Y2 axis')
        else:
            axis_action = menu.addAction('Move to Y1 axis')
        reset_action = menu.addAction('Reset to Default')

        action = menu.exec_(QtGui.QCursor.pos())
        if action == scale_action:
            self._edit_channel_scale(channel_index, label)
        elif action == axis_action:
            new_axis = 2 if current_axis == 1 else 1
            if new_axis == 1:
                self.channel_axes.pop(channel_index, None)
            else:
                self.channel_axes[channel_index] = 2
            self._on_channels_changed()
        elif action == rename_action:
            current = self._channel_name(channel_index)
            new_name, ok = QtWidgets.QInputDialog.getText(
                self, 'Rename Channel', f'Channel {channel_index} name:', text=current)
            if ok and new_name.strip():
                self.channel_names[channel_index] = new_name.strip()
                label.setText(self._channel_display(channel_index))
                self._rebuild_channel_toggles()
                self._save_settings()
        elif action == color_action:
            current_color = QColor(self._channel_color(channel_index))
            color = QtWidgets.QColorDialog.getColor(current_color, self, 'Channel Color')
            if color.isValid():
                hex_color = color.name()
                self.channel_colors[channel_index] = hex_color
                on_y2 = self.channel_axes.get(channel_index, 1) == 2
                pen_style = QtCore.Qt.DashLine if on_y2 else QtCore.Qt.SolidLine
                self.plot_lines[channel_index].setPen(pg.mkPen(hex_color, width=2, style=pen_style))
                sample.item = self.plot_lines[channel_index]
                sample.update()
                self._rebuild_channel_toggles()
                self._save_settings()
        elif action == reset_action:
            was_y2 = self.channel_axes.get(channel_index, 1) == 2
            self.channel_names.pop(channel_index, None)
            self.channel_colors.pop(channel_index, None)
            self.channel_axes.pop(channel_index, None)
            self.channel_scale.pop(channel_index, None)
            self.channel_offset.pop(channel_index, None)
            self.channel_units.pop(channel_index, None)
            self._invalidate_scale_cache()
            if was_y2:
                # Axis changed, need full rebuild
                self._on_channels_changed()
            else:
                default_name = f'Ch {channel_index}'
                default_color = PLOT_COLORS[channel_index % len(PLOT_COLORS)]
                label.setText(default_name)
                self.plot_lines[channel_index].setPen(pg.mkPen(default_color, width=2))
                sample.item = self.plot_lines[channel_index]
                sample.update()
                self._rebuild_channel_toggles()
                self._save_settings()

    def _edit_channel_scale(self, channel_index, label):
        """Prompt for per-channel gain, offset, and unit; apply to the plot."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f'Channel {channel_index}: Scale / Offset / Units')
        form = QtWidgets.QFormLayout(dlg)
        scale_edit = QtWidgets.QLineEdit(str(self.channel_scale.get(channel_index, 1.0)))
        offset_edit = QtWidgets.QLineEdit(str(self.channel_offset.get(channel_index, 0.0)))
        unit_edit = QtWidgets.QLineEdit(self.channel_units.get(channel_index, ''))
        unit_edit.setPlaceholderText('e.g. V, °C, rpm')
        hint = QtWidgets.QLabel('Plotted value = raw x scale + offset')
        hint.setStyleSheet('color: #888;')
        form.addRow('Scale (gain):', scale_edit)
        form.addRow('Offset:', offset_edit)
        form.addRow('Unit:', unit_edit)
        form.addRow(hint)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        try:
            scale = float(scale_edit.text())
            offset = float(offset_edit.text())
        except ValueError:
            self.window().statusText.setText('Scale/offset must be numbers')
            return
        unit = unit_edit.text().strip()
        # Store only non-defaults so settings stay clean
        if scale == 1.0:
            self.channel_scale.pop(channel_index, None)
        else:
            self.channel_scale[channel_index] = scale
        if offset == 0.0:
            self.channel_offset.pop(channel_index, None)
        else:
            self.channel_offset[channel_index] = offset
        if unit:
            self.channel_units[channel_index] = unit
        else:
            self.channel_units.pop(channel_index, None)
        self._invalidate_scale_cache()
        label.setText(self._channel_display(channel_index))
        self._rebuild_channel_toggles()
        self._save_settings()

    def _fft_state_changed(self):
        """Create or destroy the FFT widget when the checkbox toggles."""
        if self._fft_check.isChecked():
            if self.graphWidget is not None and self._fft_widget is None:
                self._create_fft_widget()
        else:
            self._destroy_fft_widget()

    def _create_fft_widget(self):
        """Create the FFT PlotWidget and add it to the splitter below the main graph."""
        _ensure_pg()
        dark = getattr(self.window(), '_dark_mode', False)
        self._fft_widget = pg.PlotWidget()
        self._fft_widget.setBackground('#2b2b2b' if dark else '#FFFFFF')
        self._fft_widget.setMinimumHeight(120)
        axis_color = '#ffffff' if dark else '#000000'
        self._fft_widget.plotItem.getAxis('bottom').setPen(pg.mkPen(color=axis_color))
        self._fft_widget.plotItem.getAxis('left').setPen(pg.mkPen(color=axis_color))
        self._fft_widget.plotItem.getAxis('bottom').setTextPen(pg.mkPen(color=axis_color))
        self._fft_widget.plotItem.getAxis('left').setTextPen(pg.mkPen(color=axis_color))
        self._fft_widget.plotItem.setLabel('bottom', 'Frequency bin')
        self._fft_widget.plotItem.setLabel('left', 'Magnitude')
        self._fft_widget.plotItem.showGrid(True, True, 0.3)
        self._fft_widget.plotItem.vb.setMouseMode(pg.ViewBox.PanMode)
        # Insert after the main graph (index 1 in the splitter)
        self.splitter.insertWidget(1, self._fft_widget)
        self._create_fft_lines()
        self._fft_update_counter = 0

    def _create_fft_lines(self):
        """Create FFT plot lines matching current channel count and colors."""
        if self._fft_widget is None:
            return
        # Remove old lines
        for line in self._fft_lines:
            self._fft_widget.plotItem.removeItem(line)
        self._fft_lines = []
        n = self.graph_channels.value()
        for i in range(n):
            color = self._channel_color(i)
            line = self._fft_widget.plotItem.plot(pen=pg.mkPen(color, width=2))
            line.setVisible(i not in self.hidden_channels)
            self._fft_lines.append(line)

    def _destroy_fft_widget(self):
        """Remove and destroy the FFT widget."""
        if self._fft_widget is not None:
            self._fft_widget.setParent(None)
            self._fft_widget.deleteLater()
            self._fft_widget = None
            self._fft_lines = []
            self._fft_update_counter = 0

    def _update_fft(self):
        """Compute and display FFT magnitude for each channel."""
        if self._fft_widget is None or not self.plot_data:
            return
        n_total = len(self.plot_data[0])
        n = min(self._plot_fill, n_total)
        if n < 8:
            return
        # Hann window + single-sided amplitude normalization; skip the NaN head.
        # Window/scale are cached because n is constant once the buffer fills.
        if self._fft_window is None or self._fft_window_n != n:
            self._fft_window = np.hanning(n)
            self._fft_window_n = n
            self._fft_scale = 2.0 / self._fft_window.sum()
        window, scale = self._fft_window, self._fft_scale
        for i, arr in enumerate(self.plot_data):
            if i >= len(self._fft_lines):
                break
            if i in self.hidden_channels:
                continue
            data = np.nan_to_num(arr[n_total - n:])
            fft_mag = np.abs(np.fft.rfft(data * window)) * scale
            # The 2x single-sided factor doesn't apply to DC (and Nyquist when
            # n is even), which have no mirror-image negative-frequency twin.
            fft_mag[0] *= 0.5
            if n % 2 == 0:
                fft_mag[-1] *= 0.5
            self._fft_lines[i].setData(fft_mag)

    # --- Math / Computed Channels ---

    def _open_math_dialog(self):
        """Open the math channel configuration dialog."""
        dlg = MathChannelDialog(self._math_channels, self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self._math_channels = dlg.get_math_channels()
            self._rebuild_math_lines()
            self._save_settings()

    def _math_channel_color(self, math_index):
        """Return a color for a math channel, picking from unused PLOT_COLORS."""
        n_regular = self.graph_channels.value()
        color_index = n_regular + math_index
        return PLOT_COLORS[color_index % len(PLOT_COLORS)]

    def _rebuild_math_lines(self):
        """Create or remove math channel plot lines to match definitions."""
        if self.graphWidget is None:
            self._math_lines = []
            self._math_data = []
            self._math_errors = set()
            return

        # Remove old math lines
        for line in self._math_lines:
            self.graphWidget.plotItem.removeItem(line)
            if self.graphWidget.plotItem.legend is not None:
                self.graphWidget.plotItem.legend.removeItem(line)
        self._math_lines = []
        self._math_data = []
        self._math_errors = set()

        plot_length = self.plot_length_spin.value()
        for i, mch in enumerate(self._math_channels):
            color = self._math_channel_color(i)
            pen = pg.mkPen(color, width=2, style=QtCore.Qt.DotLine)
            name = mch.get('name', f'Math {i}')
            line = pg.PlotDataItem(pen=pen, name=name)
            arr = np.zeros(plot_length)
            line.setData(arr)
            self.graphWidget.plotItem.addItem(line)
            if self.graphWidget.plotItem.legend is not None:
                self.graphWidget.plotItem.legend.addItem(line, name)
            self._math_lines.append(line)
            self._math_data.append(arr)

    # AST node types allowed in math expressions. Names are restricted to
    # ch0..chN / np / numpy, attribute access to non-underscore attributes
    # rooted at np, so a hostile settings file cannot execute arbitrary code.
    _MATH_ALLOWED_NODES = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
        ast.IfExp, ast.Call, ast.Attribute, ast.Name, ast.Load,
        ast.Constant, ast.Tuple, ast.List, ast.Subscript, ast.Slice, ast.Index,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd, ast.Invert, ast.BitAnd, ast.BitOr, ast.BitXor,
        ast.LShift, ast.RShift, ast.And, ast.Or, ast.Not,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.keyword,
    )

    @classmethod
    def _validate_math_ast(cls, tree):
        """Return True if every node in the expression tree is whitelisted."""
        for node in ast.walk(tree):
            if not isinstance(node, cls._MATH_ALLOWED_NODES):
                return False
            if isinstance(node, ast.Name):
                name = node.id
                if name not in ('np', 'numpy') and not (
                        name.startswith('ch') and name[2:].isdigit()):
                    return False
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith('_'):
                    return False
            elif isinstance(node, ast.Constant):
                if not isinstance(node.value, (int, float, complex, bool)):
                    return False
        return True

    def _compile_math_expression(self, expression):
        """Validate + compile an expression once; cache the result (None = rejected)."""
        if expression in self._math_expr_cache:
            return self._math_expr_cache[expression]
        code = None
        try:
            tree = ast.parse(expression, mode='eval')
            if self._validate_math_ast(tree):
                code = compile(tree, '<math channel>', 'eval')
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            code = None
        if len(self._math_expr_cache) > 256:
            self._math_expr_cache.clear()
        self._math_expr_cache[expression] = code
        return code

    def _build_math_namespace(self):
        """Namespace for math eval: channels bound as read-only views so an
        expression like ``ch0.sort()`` raises instead of corrupting the live
        plot buffer. Built once per flush and reused across all expressions."""
        namespace = {'np': np, 'numpy': np}
        for i, arr in enumerate(self.plot_data):
            view = arr.view()
            view.flags.writeable = False
            namespace[f'ch{i}'] = view
        return namespace

    def _eval_math_expression(self, expression, namespace=None):
        """Evaluate a math expression safely and return the result array."""
        code = self._compile_math_expression(expression)
        if code is None:
            return None
        if namespace is None:
            namespace = self._build_math_namespace()
        try:
            result = eval(code, {"__builtins__": {}}, namespace)
            # Ensure result is a numpy array of the right length
            if isinstance(result, (int, float)):
                result = np.full(len(self.plot_data[0]), result)
            result = np.asarray(result, dtype=float)
            if result.shape != self.plot_data[0].shape:
                return None
            # A pass-through expression (e.g. "ch0") returns the read-only
            # channel view; copy so the gap-fill below can write to it.
            if not result.flags.writeable:
                result = result.copy()
            # inf wrecks Y auto-range; render as a gap instead
            result[~np.isfinite(result)] = np.nan
            return result
        except Exception:
            return None

    def _update_math_channels(self):
        """Evaluate all math expressions and update their plot lines."""
        if not self._math_channels or not self._math_lines or not self.plot_data:
            return
        namespace = self._build_math_namespace()
        for i, mch in enumerate(self._math_channels):
            if i >= len(self._math_lines):
                break
            result = self._eval_math_expression(mch['expression'], namespace)
            if result is not None:
                self._math_data[i][:] = result
                self._math_lines[i].setData(self._math_data[i])
                if i in self._math_errors:
                    self._math_errors.discard(i)
                    # Restore normal pen
                    color = self._math_channel_color(i)
                    self._math_lines[i].setPen(pg.mkPen(color, width=2, style=QtCore.Qt.DotLine))
            else:
                if i not in self._math_errors:
                    self._math_errors.add(i)
                    # Set red pen to indicate error
                    self._math_lines[i].setPen(pg.mkPen('#ff0000', width=2, style=QtCore.Qt.DotLine))

    def _ingest_row(self, values):
        """Queue one sample row (one value per channel) for the next display flush.

        Only cheap bookkeeping happens here; all numpy work and setData calls
        are batched in flush_plot() at ~30 fps.
        """
        if self._plot_paused or not self.plot_data:
            return

        # Trigger detection on the configured channel, evaluated per row
        if self._trigger_enabled:
            if (self._trigger_armed and self._trigger_countdown < 0
                    and self._trigger_channel < len(values)):
                value = values[self._trigger_channel]
                if value is not None and math.isfinite(value):
                    # Compare in displayed units: the plot shows
                    # raw*scale+offset, so the level the user types must be
                    # matched against the same transform (raw units before
                    # made the trigger fire at the wrong level).
                    ch = self._trigger_channel
                    value = (value * self.channel_scale.get(ch, 1.0)
                             + self.channel_offset.get(ch, 0.0))
                    prev = self._trigger_prev_value
                    self._trigger_prev_value = value
                    if prev is not None:
                        if self._trigger_edge == 'rising':
                            fired = prev < self._trigger_level <= value
                        else:
                            fired = prev > self._trigger_level >= value
                        if fired:
                            # Pause once the trigger point reaches mid-window
                            self._trigger_countdown = len(self.plot_data[0]) // 2
            elif self._trigger_countdown > 0:
                self._trigger_countdown -= 1
                if self._trigger_countdown <= 0:
                    self._plot_paused = True
                    self._trigger_armed = False
                    self._trigger_countdown = -1
                    self._update_pause_btn_style()
                    self._position_overlay_buttons()
                    return  # freeze with the trigger point centered

        self._pending_rows.append(values)

    def flush_plot(self):
        """Push all queued samples into the plot buffers and redraw once.

        Called from the ~30 fps display timer: one in-place array shift and one
        setData per visible channel per frame, regardless of the sample rate.
        Samples arrive either as numpy blocks (binary, fully vectorized) or as
        rows (ASCII/frame); both are combined into one batch here.
        """
        if not self._pending_rows and not self._pending_blocks:
            return
        rows, self._pending_rows = self._pending_rows, []
        blocks, self._pending_blocks = self._pending_blocks, []
        if self.graphWidget is None or not self.plot_data:
            return

        nch = len(self.plot_data)
        n_points = len(self.plot_data[0])

        parts = []
        if rows:
            rb = np.full((len(rows), nch), np.nan)
            for j, row in enumerate(rows):
                m = min(len(row), nch)
                rb[j, :m] = row[:m]
            parts.append(rb)
        for b in blocks:
            if b.shape[1] == nch:
                parts.append(b)
            elif b.shape[1] > nch:
                parts.append(b[:, :nch])  # extra channels in stream: clip
            else:
                pad = np.full((b.shape[0], nch), np.nan)  # fewer: NaN-pad
                pad[:, :b.shape[1]] = b
                parts.append(pad)
        if not parts:
            return
        batch = parts[0] if len(parts) == 1 else np.vstack(parts)

        k = len(batch)
        # Measure sample rate for the time axis (count every sample, even the
        # ones a too-full flush is about to drop).
        self._update_sample_rate(k)
        if k > n_points:
            batch = batch[-n_points:]  # one flush delivered more than the window
            k = n_points

        # Apply per-channel scale/offset (vectorized): plotted = raw*scale+offset
        scale, offset = self._build_scale_vectors(nch)
        if scale is not None:
            batch = batch * scale + offset

        # inf would collapse Y auto-range; render non-finite values as gaps.
        # Runs on the size-capped batch (<= n_points rows), so it stays cheap.
        finite = np.isfinite(batch)
        if not finite.all():
            batch = np.where(finite, batch, np.nan)

        for i, arr in enumerate(self.plot_data):
            if k >= n_points:
                arr[:] = batch[:, i]
            else:
                arr[:-k] = arr[k:]
                arr[-k:] = batch[:, i]
            if i not in self.hidden_channels:
                self.plot_lines[i].setData(arr)
        self._plot_fill = min(n_points, self._plot_fill + k)

        if self._math_lines:
            self._update_math_channels()
        if self._fft_widget is not None:
            self._fft_update_counter += 1
            if self._fft_update_counter >= 3:  # ~10 Hz is plenty for a spectrum
                self._fft_update_counter = 0
                self._update_fft()
        if self._x_time_mode:
            self._update_x_axis_scale_value()

    def _update_sample_rate(self, k):
        """Track an average sample rate (samples/sec) from wall-clock time."""
        if k <= 0:
            return
        now = time.monotonic()
        if self._x_start_time is None:
            self._x_start_time = now
            self._x_sample_total = 0
        self._x_sample_total += k
        elapsed = now - self._x_start_time
        if elapsed > 0.2:  # ignore the first noisy fraction of a second
            self._x_rate = self._x_sample_total / elapsed

    def _reset_sample_rate(self):
        self._x_sample_total = 0
        self._x_start_time = None
        self._x_rate = 0.0

    def _apply_x_axis_scale(self):
        """Set the bottom axis label + scale for the current X mode.

        Only the axis tick scale/label changes; plotted data stays in index
        space, so the crosshair and ranges keep working unchanged. Call on
        mode toggle / plot creation; flush_plot uses the lighter scale-only
        update below.
        """
        if self.graphWidget is None:
            return
        axis = self.graphWidget.plotItem.getAxis('bottom')
        if self._x_time_mode:
            axis.setLabel('Time', units='s')
            axis.setScale(1.0 / self._x_rate if self._x_rate > 0 else 1.0)
        else:
            axis.setLabel('Sample')
            axis.setScale(1.0)

    def _update_x_axis_scale_value(self):
        """Live per-flush update of just the time-axis tick scale (no relabel)."""
        if self.graphWidget is None or not self._x_time_mode or self._x_rate <= 0:
            return
        self.graphWidget.plotItem.getAxis('bottom').setScale(1.0 / self._x_rate)

    def set_x_time_mode(self, enabled):
        """Toggle the time-based X axis."""
        self._x_time_mode = bool(enabled)
        self._apply_x_axis_scale()
        self._save_settings()

    def _get_delimiter(self):
        """Return the active delimiter string, or None for auto-detect."""
        mode = self.delimiter_combo.currentText()
        delim_map = {'Comma': ',', 'Semicolon': ';', 'Space': ' ', 'Tab': '\t'}
        if mode == 'Auto':
            return None
        if mode == 'Other':
            custom = self.delimiter_custom.text()
            return custom if custom else None
        return delim_map.get(mode)

    def _parse_plot_values(self, line):
        """Parse a line of serial data into numeric values."""
        line = line.strip()
        if not line:
            return []
        delim = self._get_delimiter()
        if delim is None:
            # Auto-detect: tab > comma > semicolon > space
            if '\t' in line:
                fields = line.split('\t')
            elif ',' in line:
                fields = line.split(',')
            elif ';' in line:
                fields = line.split(';')
            else:
                fields = line.split()
        elif delim == ' ':
            fields = line.split()
        else:
            fields = line.split(delim)
        values = []
        was_empty = []  # parallel flag: value came from an empty field
        for field in fields:
            field = field.strip()
            if not field:
                # An empty field between delimiters is still a column: keep the
                # position so later values stay on their own channels.
                values.append(float('nan'))
                was_empty.append(True)
                continue
            if ':' in field:
                field = field.split(':', 1)[1].strip()
            try:
                values.append(float(field))
            except ValueError:
                # Non-numeric column (e.g. a label, a sensor error token):
                # emit a gap rather than dropping it, which would shift every
                # following value into the wrong channel.
                values.append(float('nan'))
            was_empty.append(False)
        # Drop trailing NaNs that came from a line-ending delimiter (e.g.
        # "1,2,3,"), but keep an explicit nan/error token in the last column so
        # that channel isn't silently dropped.
        while values and was_empty[-1]:
            values.pop()
            was_empty.pop()
        return values

    def translate_data(self):
        """Convert input text using the selected conversion type."""
        conversion = self.convert_A_type.currentText()
        input_text = self.convert_A_text.toPlainText()
        self.convert_B_text.clear()
        if not input_text:
            return
        converter = CONVERTERS.get(conversion)
        if converter:
            try:
                self.convert_B_text.insertPlainText(converter(input_text))
            except Exception:
                self.convert_B_text.insertPlainText("not valid")

    def _do_search(self):
        """Highlight all occurrences of search text in the ASCII view."""
        term = self.search_input.text()
        if not term:
            self._clear_search()
            return
        # Reset all formatting first
        cursor = self.serialData.textCursor()
        cursor.select(QTextCursor.Document)
        fmt = QtGui.QTextCharFormat()
        fmt.setBackground(QColor('transparent'))
        cursor.mergeCharFormat(fmt)
        cursor.clearSelection()
        # Find and highlight all matches (background only, keep RX/TX colors)
        highlight_fmt = QtGui.QTextCharFormat()
        highlight_fmt.setBackground(QColor('#FFFF00'))
        count = 0
        find_cursor = self.serialData.document().find(term)
        while not find_cursor.isNull():
            find_cursor.mergeCharFormat(highlight_fmt)
            count += 1
            find_cursor = self.serialData.document().find(term, find_cursor)
        self.search_count.setText(f'{count} matches')

    def _clear_search(self):
        """Remove all search highlights and clear the search field."""
        self.search_input.clear()
        self.search_count.setText('')
        cursor = self.serialData.textCursor()
        cursor.select(QTextCursor.Document)
        fmt = QtGui.QTextCharFormat()
        fmt.setBackground(QColor('transparent'))
        cursor.mergeCharFormat(fmt)
        cursor.clearSelection()

    def toggle_search(self):
        """Show or hide the search bar."""
        vis = not self.search_bar.isVisible()
        self.search_bar.setVisible(vis)
        if vis:
            self.search_input.setFocus()
        else:
            self._clear_search()  # don't leave stale highlights behind

    def clear_button_Clicked(self):
        self.serialDataHex.clear()
        self.serialData.clear()
        self.convert_A_text.clear()
        self.convert_B_text.clear()
        self.reset_stream_state()
        self._hex_col = 0
        monitor = self.window()
        if hasattr(monitor, '_rx_buffer'):
            monitor._rx_buffer.clear()

    def _on_mode_changed(self):
        """Show/hide controls based on data mode."""
        mode = self.data_mode.currentText()
        is_ascii = (mode == 'ASCII')
        is_binary = (mode == 'Binary Stream')
        is_frame = (mode == 'Custom Frame')

        # Binary/frame shared widgets
        for w in self._binary_frame_widgets:
            w.setVisible(is_binary or is_frame)

        # ASCII-only widgets
        self.delimiter_combo.setVisible(is_ascii)
        self.delimiter_custom.setVisible(
            is_ascii and self.delimiter_combo.currentText() == 'Other')

        # Frame-only widgets
        for w in self._frame_only_widgets:
            w.setVisible(is_frame)

        # Binary-only sync button
        self.sync_button.setVisible(is_binary)

        if is_ascii:
            self.label_sent_data.setText('Data: ASCII')
        else:
            self.label_sent_data.setText('Data: Decoded')

        self.serialData.clear()
        self.serialDataHex.clear()
        self.reset_stream_state()
        self._hex_col = 0
        monitor = self.window()
        if hasattr(monitor, '_rx_buffer'):
            monitor._rx_buffer.clear()

        self._apply_reader_settings()
        self._save_settings()

    def _on_setting_changed(self):
        """Apply current settings to readers."""
        self._apply_reader_settings()
        self._save_settings()

    def _on_pts_changed(self):
        """Rebuild plot arrays when the Pts value changes."""
        if self.graphWidget is not None:
            # Rebuild main plot lines with new array length
            for line in self.plot_lines:
                self.graphWidget.plotItem.removeItem(line)
            if self._y2_viewbox:
                for line in self._y2_plot_lines.values():
                    self._y2_viewbox.removeItem(line)
                self._y2_plot_lines.clear()
            if self.graphWidget.plotItem.legend is not None:
                self.graphWidget.plotItem.legend.clear()
            self._create_plot_lines()
            self.graphWidget.setXRange(0, self.plot_length_spin.value())
        self._save_settings()

    def _on_delimiter_changed(self):
        """Show/hide custom delimiter input and save."""
        self.delimiter_custom.setVisible(self.delimiter_combo.currentText() == 'Other')
        self._save_settings()

    def _on_size_field_changed(self):
        """Enable/disable frame size spinner based on size field type."""
        self.frame_size_spin.setEnabled(self.size_field_combo.currentText().startswith('Fixed'))
        self._on_setting_changed()

    def _on_sync_clicked(self):
        """Clear binary stream buffer for re-alignment."""
        self._binary_reader.sync()

    def _apply_reader_settings(self):
        """Push current UI settings to both reader objects."""
        dtype = self.type_combo.currentText()
        endian = 'little' if self.endian_combo.currentText().startswith('Little') else 'big'
        nch = self.graph_channels.value()

        self._binary_reader.data_type = dtype
        self._binary_reader.endianness = endian
        self._binary_reader.num_channels = nch

        self._frame_reader.data_type = dtype
        self._frame_reader.endianness = endian
        self._frame_reader.num_channels = nch
        sf_text = self.size_field_combo.currentText()
        self._frame_reader.size_field = 'fixed' if sf_text.startswith('Fixed') else sf_text.split(' ')[0]
        self._frame_reader.frame_size = self.frame_size_spin.value()
        self._frame_reader.checksum_enabled = self.checksum_check.isChecked()

        # An invalid or empty sync word keeps the previous one, but must say
        # so - silently decoding against a sync word different from what the
        # field shows is a debugging trap. (Warning emitted at the end so the
        # frame-size warning below can't overwrite it.)
        sync_text = self.sync_word_edit.text().replace(' ', '')
        try:
            sw = bytes.fromhex(sync_text)
        except ValueError:
            sw = b''
        if sw:
            self._frame_reader.sync_word = sw

        # Settings changed: leftover bytes from the old layout would misalign
        self._binary_reader.sync()
        self._frame_reader.reset()

        # Warn when a fixed frame can't hold a whole sample (decodes nothing)
        if (self.data_mode.currentText() == 'Custom Frame'
                and self._frame_reader.size_field == 'fixed'):
            _, type_size = DATA_TYPES[dtype]
            sample_size = nch * type_size
            frame_size = self.frame_size_spin.value()
            monitor = self.window()
            if frame_size % sample_size != 0 and hasattr(monitor, 'statusText'):
                detail = (f'sample size {sample_size} ({nch} ch x {type_size} B {dtype})')
                if frame_size < sample_size:
                    monitor.statusText.setText(
                        f'Warning: frame size {frame_size} < {detail} - no samples will decode')
                else:
                    monitor.statusText.setText(
                        f'Warning: frame size {frame_size} is not a multiple of {detail} - '
                        f'trailing bytes are ignored')

        if not sw and self.data_mode.currentText() == 'Custom Frame':
            monitor = self.window()
            if hasattr(monitor, 'statusText'):
                monitor.statusText.setText(
                    f'Invalid start byte {self.sync_word_edit.text()!r} - '
                    f'still using {self._frame_reader.sync_word.hex().upper()}')

    def _get_settings_dict(self):
        """Build the current settings as a dict."""
        return {
            'mode': self.data_mode.currentText(),
            'num_channels': self.graph_channels.value(),
            'num_points': self.plot_length_spin.value(),
            'show_plot': self.graph_mode.isChecked(),
            'show_fft': self._fft_check.isChecked(),
            'delimiter': self.delimiter_combo.currentText(),
            'delimiter_custom': self.delimiter_custom.text(),
            'channel_names': {str(k): v for k, v in self.channel_names.items()},
            'channel_colors': {str(k): v for k, v in self.channel_colors.items()},
            'channel_axes': {str(k): v for k, v in self.channel_axes.items()},
            'channel_scale': {str(k): v for k, v in self.channel_scale.items()},
            'channel_offset': {str(k): v for k, v in self.channel_offset.items()},
            'channel_units': {str(k): v for k, v in self.channel_units.items()},
            'hidden_channels': sorted(self.hidden_channels),
            'x_time_mode': self._x_time_mode,
            'y_auto_scale': self._y_auto_scale,
            'y_min': self._y_min,
            'y_max': self._y_max,
            'binary': {
                'data_type': self.type_combo.currentText(),
                'endianness': 'little' if self.endian_combo.currentText().startswith('Little') else 'big',
            },
            'frame': {
                'data_type': self.type_combo.currentText(),
                'endianness': 'little' if self.endian_combo.currentText().startswith('Little') else 'big',
                'sync_word': self.sync_word_edit.text(),
                'size_field': 'fixed' if self.size_field_combo.currentText().startswith('Fixed') else self.size_field_combo.currentText().split(' ')[0],
                'frame_size': self.frame_size_spin.value(),
                'checksum': self.checksum_check.isChecked(),
            },
            'math_channels': self._math_channels,
        }

    def _save_settings(self, path=None):
        """Persist settings via parent SerialMonitor (debounced unless a path is given)."""
        monitor = self.window()
        if path is not None:
            if hasattr(monitor, 'save_all_settings'):
                monitor.save_all_settings(path)
        elif hasattr(monitor, 'schedule_save'):
            monitor.schedule_save()

    def _load_plot_settings(self, s):
        """Restore plot/decode settings from a dict (subsection of full settings)."""
        # If a plot is currently visible (File > Load Settings), tear it down
        # first so it is rebuilt below with the loaded channel count/Pts/colors
        # instead of staying stale.
        if self.graph_mode.isChecked():
            self.graph_mode.setChecked(False)

        widgets = [self.data_mode, self.type_combo, self.endian_combo,
                   self.graph_channels, self.plot_length_spin,
                   self.delimiter_combo, self.delimiter_custom,
                   self.sync_word_edit, self.size_field_combo,
                   self.frame_size_spin, self.checksum_check,
                   self._fft_check]
        for w in widgets:
            w.blockSignals(True)

        # Restore per key: one malformed value must only lose that key, not
        # abort the whole restore (which then got auto-saved back over the
        # user's good settings file).
        if not isinstance(s, dict):
            s = {}
        bad_keys = []

        def _try(label, fn):
            try:
                fn()
            except (KeyError, TypeError, ValueError, AttributeError):
                bad_keys.append(label)

        def _int_dict(raw, coerce):
            out = {}
            if isinstance(raw, dict):
                for k, v in raw.items():
                    try:
                        out[int(k)] = coerce(v)
                    except (TypeError, ValueError):
                        bad_keys.append(f'channel entry {k!r}')
            return out

        _try('mode', lambda: self.data_mode.setCurrentText(str(s.get('mode', 'ASCII'))))
        _try('num_channels', lambda: self.graph_channels.setValue(int(s.get('num_channels', 4))))
        _try('num_points', lambda: self.plot_length_spin.setValue(int(s.get('num_points', DEFAULT_PLOT_LENGTH))))
        _try('delimiter', lambda: self.delimiter_combo.setCurrentText(str(s.get('delimiter', 'Auto'))))
        _try('delimiter_custom', lambda: self.delimiter_custom.setText(str(s.get('delimiter_custom', ''))))

        # Load channel properties AND axis ranges BEFORE enabling the graph
        # so that graph_state_changed() sees the correct names/colors/axes/
        # hidden set and restores the saved manual Y range.
        self.channel_names = _int_dict(s.get('channel_names', {}), str)
        self.channel_colors = _int_dict(s.get('channel_colors', {}), str)
        self.channel_axes = _int_dict(s.get('channel_axes', {}), int)
        self.channel_scale = _int_dict(s.get('channel_scale', {}), float)
        self.channel_offset = _int_dict(s.get('channel_offset', {}), float)
        self.channel_units = _int_dict(s.get('channel_units', {}), str)
        self._invalidate_scale_cache()

        def _restore_hidden():
            self.hidden_channels = {int(i) for i in s.get('hidden_channels', [])}
        _try('hidden_channels', _restore_hidden)

        def _restore_x_time():
            self._x_time_mode = bool(s.get('x_time_mode', False))
            self._time_axis_check.blockSignals(True)
            self._time_axis_check.setChecked(self._x_time_mode)
            self._time_axis_check.blockSignals(False)
        _try('x_time_mode', _restore_x_time)

        def _restore_y_range():
            self._y_auto_scale = bool(s.get('y_auto_scale', True))
            self._y_min = float(s.get('y_min', -1.0))
            self._y_max = float(s.get('y_max', 1.0))
        _try('y_range', _restore_y_range)

        def _restore_math():
            # Math channels must be known before the graph is created
            saved_math = s.get('math_channels', [])
            if isinstance(saved_math, list):
                self._math_channels = [
                    {'name': m.get('name', ''), 'expression': m.get('expression', '')}
                    for m in saved_math
                    if isinstance(m, dict) and m.get('expression', '').strip()
                ]
        _try('math_channels', _restore_math)

        _try('show_fft', lambda: self._fft_check.setChecked(bool(s.get('show_fft', False))))
        _try('show_plot', lambda: self.graph_mode.setChecked(bool(s.get('show_plot', False))))

        def _restore_decode():
            frame = s.get('frame', {})
            binary = s.get('binary', {})
            dtype = frame.get('data_type', binary.get('data_type', 'float32'))
            endian = frame.get('endianness', binary.get('endianness', 'little'))

            self.type_combo.setCurrentText(dtype)
            self.endian_combo.setCurrentText('Little Endian' if endian == 'little' else 'Big Endian')
            self.sync_word_edit.setText(str(frame.get('sync_word', 'AA')))
            sf = frame.get('size_field', 'fixed')
            sf_map = {'fixed': 'Fixed', '1-byte': '1-byte size field', '2-byte': '2-byte size field'}
            self.size_field_combo.setCurrentText(sf_map.get(sf, 'Fixed'))
            self.frame_size_spin.setValue(int(frame.get('frame_size', 12)))
            self.checksum_check.setChecked(bool(frame.get('checksum', False)))
        _try('frame/binary', _restore_decode)

        for w in widgets:
            w.blockSignals(False)

        if bad_keys:
            monitor = self.window()
            if hasattr(monitor, 'statusText'):
                monitor.statusText.setText(
                    'Some saved plot settings were invalid and skipped: '
                    + ', '.join(bad_keys[:5]))

        self._apply_reader_settings()
        self.frame_size_spin.setEnabled(self.size_field_combo.currentText().startswith('Fixed'))
        self.delimiter_custom.setVisible(self.delimiter_combo.currentText() == 'Other')
        self._on_mode_changed()

    def reset_stream_state(self):
        """Drop partial decode/parse state (new connection or mode change)."""
        self._binary_reader.sync()
        self._frame_reader.reset()
        self._pending_cr = False
        self._ascii_line_buffer = ''
        self._pending_rows = []
        self._pending_blocks = []
        # New session: don't glue its first bytes onto the old partial hex row
        self._hex_col = 0
        # The whole-session average rate would otherwise count the idle gap
        # (disconnect, device silent) and permanently skew the time axis
        self._reset_sample_rate()

    def handleReceivedData(self, raw_bytes):
        """Route incoming serial bytes based on current data mode."""
        mode = self.data_mode.currentText()

        if mode == 'ASCII':
            text = raw_bytes.decode('ISO-8859-1')
            self.appendSerialText(text, "read")
            return

        # Binary/Frame modes: always show raw HEX
        self._append_hex_view(raw_bytes)
        plotting = self.graph_mode.isChecked() and self.graphWidget is not None

        if mode == 'Binary Stream':
            # Fully vectorized decode -> numpy block, no per-sample Python.
            block = self._binary_reader.feed_np(raw_bytes)
            if block is None or len(block) == 0:
                return
            self._append_decoded_arr(block)
            if plotting:
                if self._trigger_enabled:
                    # Trigger needs per-sample evaluation; use the row path
                    for row in block:
                        self._ingest_row(row.tolist())
                elif not self._plot_paused:
                    # The row path checks _plot_paused in _ingest_row; the
                    # block path must too, or Pause doesn't pause binary mode
                    self._pending_blocks.append(block)
        else:
            samples = self._frame_reader.feed(raw_bytes)
            if not samples:
                return
            self._append_decoded_lines(samples)
            if plotting:
                for sample in samples:
                    self._ingest_row([float(v) for v in sample])

    def _format_hex(self, raw_bytes):
        """Format bytes as space-separated uppercase pairs, 16 per line.

        Uses and updates self._hex_col so chunks of any size continue the
        current line correctly (with a separating space) and a newline is
        emitted as soon as a line completes — chunk boundaries never merge
        pairs or glue rows together.
        """
        hex_str = raw_bytes.hex().upper()
        pairs = [hex_str[i:i + 2] for i in range(0, len(hex_str), 2)]
        out = []
        col = self._hex_col
        i = 0
        while i < len(pairs):
            take = pairs[i:i + 16 - col]
            if col > 0:
                out.append(' ')
            out.append(' '.join(take))
            col += len(take)
            i += len(take)
            if col >= 16:
                out.append('\n')
                col = 0
        self._hex_col = col
        return ''.join(out)

    def _insert_colored_text(self, text_edit, text, color):
        """Append text at the end without disturbing user selection or scroll.

        Uses a standalone cursor (so an active user selection survives) with an
        explicit format (so new text never inherits a search highlight), and
        only auto-scrolls when the view was already at the bottom.
        """
        sb = text_edit.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        fmt = QtGui.QTextCharFormat()
        fmt.setForeground(color)
        fmt.setBackground(QtGui.QBrush(QtCore.Qt.transparent))
        fmt.setFontFamily('Segoe UI')
        cursor = QTextCursor(text_edit.document())
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text, fmt)
        self._trim_text_buffer(text_edit)
        if at_bottom:
            sb.setValue(sb.maximum())

    def _insert_html(self, text_edit, html_text):
        """Append HTML at the end; same selection/scroll behavior as above."""
        sb = text_edit.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        cursor = QTextCursor(text_edit.document())
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml(html_text)
        self._trim_text_buffer(text_edit)
        if at_bottom:
            sb.setValue(sb.maximum())

    # ~120 KB/s worth of bytes per 33 ms flush. Normal serial (<=921600 baud
    # = ~92 KB/s) never hits this; only multi-MB/s USB-CDC streams do, where
    # formatting every byte to hex (only to have it trimmed) would dominate.
    HEX_VIEW_MAX_BYTES_PER_FLUSH = 4096

    def _append_hex_view(self, raw_bytes):
        """Append raw bytes to the HEX text view (right panel)."""
        if len(raw_bytes) > self.HEX_VIEW_MAX_BYTES_PER_FLUSH:
            dropped = len(raw_bytes) - self.HEX_VIEW_MAX_BYTES_PER_FLUSH
            raw_bytes = raw_bytes[-self.HEX_VIEW_MAX_BYTES_PER_FLUSH:]
            self._hex_col = 0
            self._insert_colored_text(
                self.serialDataHex,
                f'... [{dropped} bytes not shown - stream too fast for hex view] ...\n',
                QtGui.QColor(128, 128, 128))
        self._insert_colored_text(
            self.serialDataHex, self._format_hex(raw_bytes), QtGui.QColor(255, 0, 0))

    def _trim_text_buffer(self, text_edit, max_lines=MAX_TEXT_LINES):
        """Remove oldest lines if text exceeds max_lines."""
        doc = text_edit.document()
        if doc.blockCount() > max_lines:
            cursor = QTextCursor(doc.begin())
            excess = doc.blockCount() - max_lines
            for _ in range(excess):
                cursor.movePosition(QTextCursor.NextBlock, QTextCursor.KeepAnchor)
            # Selection ends at the start of the first surviving block, so the
            # removed blocks' newlines go with them — nothing extra to delete.
            cursor.removeSelectedText()

    MAX_DECODED_LINES_PER_FLUSH = 100
    MAX_DECODED_CELLS_PER_FLUSH = 600  # rows x channels budget per flush

    def _decoded_row_cap(self, nch):
        """How many rows to render this flush, bounded by a cell budget so the
        HTML cost stays flat regardless of channel count."""
        if nch <= 0:
            return self.MAX_DECODED_LINES_PER_FLUSH
        return max(10, min(self.MAX_DECODED_LINES_PER_FLUSH,
                           self.MAX_DECODED_CELLS_PER_FLUSH // nch))

    def _append_decoded_lines(self, samples):
        """Append decoded samples to the left panel as one batched HTML insert."""
        nch = max((len(s) for s in samples), default=0)
        cap = self._decoded_row_cap(nch)
        skipped = 0
        if len(samples) > cap:
            skipped = len(samples) - cap
            samples = samples[-cap:]
        names = [html.escape(self._channel_name(i)) for i in range(nch)]
        colors = [self._channel_color(i) for i in range(nch)]
        lines = []
        if skipped:
            lines.append(f'<span style="color:#888888"><i>... {skipped} samples not shown</i></span>')
        for sample in samples:
            parts = []
            for i, val in enumerate(sample):
                text = f'{names[i]}: {val:.4f}' if isinstance(val, float) else f'{names[i]}: {val}'
                parts.append(f'<span style="color:{colors[i]}">{text}</span>')
            lines.append('&nbsp; '.join(parts))
        self._insert_html(self.serialData, '<br>'.join(lines) + '<br>')

    def _append_decoded_arr(self, block):
        """Append a numpy (n, nch) decoded block to the left panel.

        Only the last MAX_DECODED_LINES_PER_FLUSH rows are formatted -- at high
        sample rates nobody can read more, and formatting every row would
        re-introduce a per-sample cost in the hot path.
        """
        n = len(block)
        nch = block.shape[1]
        cap = self._decoded_row_cap(nch)
        skipped = 0
        if n > cap:
            skipped = n - cap
            block = block[-cap:]
        names = [html.escape(self._channel_name(i)) for i in range(nch)]
        colors = [self._channel_color(i) for i in range(nch)]
        lines = []
        if skipped:
            lines.append(f'<span style="color:#888888"><i>... {skipped} samples not shown</i></span>')
        for row in block.tolist():
            parts = [f'<span style="color:{colors[i]}">{names[i]}: {row[i]:.4f}</span>'
                     for i in range(nch)]
            lines.append('&nbsp; '.join(parts))
        self._insert_html(self.serialData, '<br>'.join(lines) + '<br>')

    def _stamp_lines(self, text):
        """Prefix each line start in text with the current time.

        One timestamp is taken per chunk (a chunk is one ~33 ms display
        flush, so per-line times would all be equal anyway)."""
        prefix = '[' + datetime.now().strftime('%H:%M:%S.%f')[:-3] + '] '
        segs = text.split('\n')
        out = []
        for i, seg in enumerate(segs):
            if i > 0:
                out.append('\n')
                self._ts_at_line_start = True
            if seg and self._ts_at_line_start:
                out.append(prefix)
                self._ts_at_line_start = False
            out.append(seg)
        return ''.join(out)

    def appendSerialText(self, appendText, direction, mode="ASCII"):
        is_send = direction == "send"
        color = QtGui.QColor(0, 0, 255) if is_send else QtGui.QColor(255, 0, 0)

        # QTextEdit treats BOTH '\r' and '\n' as paragraph separators, so a
        # CRLF stream renders with a blank line between each message. Keep
        # the raw bytes for the HEX view, but normalize for the ASCII view.
        # Serial data arrives in arbitrary-size chunks; a '\r\n' pair can be
        # split across two reads, so we also carry a _pending_cr flag that
        # swallows a leading '\n' when the previous chunk ended in '\r'.
        displayText = appendText.replace('\r\n', '\n').replace('\r', '\n')
        if is_send:
            self._pending_cr = False
        else:
            if self._pending_cr and displayText.startswith('\n'):
                displayText = displayText[1:]
            self._pending_cr = appendText.endswith('\r')

        # Recover the raw bytes so the HEX pane shows what actually went over
        # the wire (ISO-8859-1 mirrors bytes 1:1; re-encoding as UTF-8 would
        # corrupt every byte >= 0x80).
        ascii_text = displayText
        if is_send and mode == 'HEX':
            try:
                raw = bytes.fromhex(appendText)
                ascii_text = raw.decode('ISO-8859-1').replace('\r\n', '\n').replace('\r', '\n')
            except ValueError:
                raw = appendText.encode('ISO-8859-1', 'replace')
        elif is_send and mode == 'BINARY':
            try:
                bits = appendText.replace(' ', '')
                raw = int(bits, 2).to_bytes(max(1, (len(bits) + 7) // 8), 'big')
            except (ValueError, OverflowError):
                raw = appendText.encode('ISO-8859-1', 'replace')
        else:
            raw = appendText.encode('ISO-8859-1', 'replace')

        if not is_send and self.show_timestamps and ascii_text:
            ascii_text = self._stamp_lines(ascii_text)
        elif is_send and ascii_text:
            # A send breaks/continues the current line like received text does
            self._ts_at_line_start = ascii_text.endswith('\n')
        self._insert_colored_text(self.serialData, ascii_text, color)
        if raw:
            # One shared formatter for RX and TX keeps hex column tracking consistent
            self._insert_colored_text(self.serialDataHex, self._format_hex(raw), color)

        # Feed the plot from complete received lines
        if not is_send and self.graph_mode.isChecked() and self.graphWidget is not None:
            combined = self._ascii_line_buffer + displayText
            lines = combined.split('\n')
            # Cap the partial-line carry so a newline-free stream can't grow it forever
            self._ascii_line_buffer = lines[-1][-4096:]
            nch = len(self.plot_data)
            for line in lines[:-1]:
                values = self._parse_plot_values(line)
                if not values:
                    continue
                row = values[:nch]
                if len(row) < nch:
                    # Pad so all channels advance together and stay time-aligned
                    row = row + [float('nan')] * (nch - len(row))
                self._ingest_row(row)


# Horizontal separator line
class HLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class MathChannelDialog(QtWidgets.QDialog):
    """Dialog for adding/removing user-defined math channel expressions."""

    def __init__(self, math_channels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Math Channels")
        self.setMinimumWidth(550)
        self.setMinimumHeight(300)
        self._rows = []

        main_layout = QtWidgets.QVBoxLayout(self)

        # Header
        header = QtWidgets.QLabel(
            "Define computed channels using numpy expressions.\n"
            "Use ch0, ch1, ... to reference plot channels. "
            "numpy is available as np.")
        header.setFont(QtGui.QFont('Segoe UI', 9))
        header.setWordWrap(True)
        main_layout.addWidget(header)

        # Scrollable area for channel rows
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self._row_container = QtWidgets.QWidget()
        self._row_layout = QtWidgets.QVBoxLayout(self._row_container)
        self._row_layout.setContentsMargins(0, 0, 0, 0)
        self._row_layout.addStretch()
        scroll.setWidget(self._row_container)
        main_layout.addWidget(scroll)

        # Add button
        add_btn = QtWidgets.QPushButton("+ Add Math Channel")
        add_btn.setFont(QtGui.QFont('Segoe UI', 10))
        add_btn.clicked.connect(self._add_row)
        main_layout.addWidget(add_btn)

        # OK / Cancel
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

        # Populate from existing definitions
        for ch in math_channels:
            self._add_row(ch.get('name', ''), ch.get('expression', ''))

    def _add_row(self, name='', expression=''):
        """Add one math channel row to the dialog."""
        # When called from button click, name will be False (Qt signal arg)
        if isinstance(name, bool):
            name = ''
        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 2, 0, 2)

        name_edit = QtWidgets.QLineEdit(name)
        name_edit.setPlaceholderText("Name (e.g. Power)")
        name_edit.setFont(QtGui.QFont('Segoe UI', 10))
        name_edit.setFixedWidth(120)

        expr_edit = QtWidgets.QLineEdit(expression)
        expr_edit.setPlaceholderText("Expression (e.g. ch0 * ch1)")
        expr_edit.setFont(QtGui.QFont('Segoe UI', 10))

        remove_btn = QtWidgets.QPushButton("X")
        remove_btn.setFont(QtGui.QFont('Segoe UI', 10))
        remove_btn.setFixedWidth(30)
        remove_btn.setToolTip("Remove this math channel")

        row_layout.addWidget(name_edit)
        row_layout.addWidget(expr_edit)
        row_layout.addWidget(remove_btn)

        row_data = {'widget': row_widget, 'name': name_edit, 'expr': expr_edit}
        self._rows.append(row_data)

        # Insert before the stretch
        self._row_layout.insertWidget(self._row_layout.count() - 1, row_widget)

        remove_btn.clicked.connect(lambda: self._remove_row(row_data))

    def _remove_row(self, row_data):
        """Remove a math channel row from the dialog."""
        if row_data in self._rows:
            self._rows.remove(row_data)
            row_data['widget'].setParent(None)
            row_data['widget'].deleteLater()

    def get_math_channels(self):
        """Return list of {'name': str, 'expression': str} from the dialog."""
        result = []
        for row in self._rows:
            name = row['name'].text().strip()
            expr = row['expr'].text().strip()
            if expr:  # skip empty expressions
                result.append({'name': name or f'Math {len(result)}', 'expression': expr})
        return result


class MacroEditDialog(QtWidgets.QDialog):
    """Dialog for editing a macro button's label and payload.

    In CAN mode the dialog also shows the complete frame the button will put
    on the wire -- ID, type, DLC and all eight data byte cells -- and every
    part of it is editable. The payload is the single source of truth: the
    byte cells, the DLC spinner and the hex/ASCII/dec/bin fields are four
    views of the same bytes. Nothing is padded behind the user's back, which
    the explanation under the frame says in so many words.
    """

    CAN_HELP = (
        "Sent exactly as entered \u2014 AxxTerm never pads. "
        "<b>FF</b> goes out as a <b>1-byte</b> frame (DLC 1), "
        "<i>not</i> as FF 00 00 00 00 00 00 00.<br>"
        "Set DLC (or press Pad 00 / Pad FF) for a full 8-byte frame. "
        "J1939 expects 8 bytes with unused ones set to FF."
    )

    def __init__(self, label, hex_data, parent=None, can_id='', can_ext=False,
                 can_mode=False):
        super().__init__(parent)
        self.setWindowTitle('Edit CAN Macro' if can_mode else 'Edit Macro Button')
        self.setMinimumWidth(500)
        self._updating = False
        self._can_mode = bool(can_mode)

        root = QtWidgets.QVBoxLayout(self)
        layout = QtWidgets.QFormLayout()
        root.addLayout(layout)

        self.label_edit = QtWidgets.QLineEdit(label)
        self.label_edit.setFont(QtGui.QFont('Segoe UI', 11))

        self.input_mode = QtWidgets.QComboBox()
        self.input_mode.addItems(['HEX', 'ASCII', 'Decimal (bytes)', 'Binary (bytes)'])
        self.input_mode.setFont(QtGui.QFont('Segoe UI', 11))
        self.input_mode.currentIndexChanged.connect(self._mode_changed)

        self.hex_edit = QtWidgets.QLineEdit(hex_data)
        self.hex_edit.setFont(QtGui.QFont('Segoe UI', 11))
        self.hex_edit.setPlaceholderText("e.g. 48 65 6C 6C 6F")

        self.ascii_edit = QtWidgets.QLineEdit()
        self.ascii_edit.setFont(QtGui.QFont('Segoe UI', 11))
        self.ascii_edit.setPlaceholderText("e.g. Hello")

        self.dec_edit = QtWidgets.QLineEdit()
        self.dec_edit.setFont(QtGui.QFont('Segoe UI', 11))
        self.dec_edit.setPlaceholderText("e.g. 72 101 108 108 111")

        self.bin_edit = QtWidgets.QLineEdit()
        self.bin_edit.setFont(QtGui.QFont('Segoe UI', 11))
        self.bin_edit.setPlaceholderText("e.g. 01001000 01100101")

        # Stack the input fields, show one at a time
        self.input_stack = QtWidgets.QStackedWidget()
        self.input_stack.addWidget(self.hex_edit)
        self.input_stack.addWidget(self.ascii_edit)
        self.input_stack.addWidget(self.dec_edit)
        self.input_stack.addWidget(self.bin_edit)
        self.input_stack.setCurrentIndex(0)

        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setFont(QtGui.QFont('Segoe UI', 10))

        self.hex_edit.textChanged.connect(lambda: self._sync_from('hex'))
        self.ascii_edit.textChanged.connect(lambda: self._sync_from('ascii'))
        self.dec_edit.textChanged.connect(lambda: self._sync_from('dec'))
        self.bin_edit.textChanged.connect(lambda: self._sync_from('bin'))

        # CAN mode fields: the same payload is sent as a CAN frame with this ID
        self.can_id_edit = QtWidgets.QLineEdit(can_id)
        self.can_id_edit.setFont(QtGui.QFont('Consolas', 11))
        self.can_id_edit.setPlaceholderText("123 (standard) or 18FEF100 (extended)")
        self.can_id_edit.textChanged.connect(self._update_frame_preview)

        self.can_ext_check = QtWidgets.QCheckBox("Extended (29-bit) ID")
        self.can_ext_check.setFont(QtGui.QFont('Segoe UI', 10))
        self.can_ext_check.setChecked(bool(can_ext))
        self.can_ext_check.toggled.connect(self._update_frame_preview)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addRow("Button Label:", self.label_edit)
        layout.addRow("Input Mode:", self.input_mode)
        layout.addRow("Data:", self.input_stack)
        layout.addRow("Preview:", self.preview_label)

        self.can_group = self._build_can_group()
        root.addWidget(self.can_group)
        self.can_group.setVisible(self._can_mode)

        root.addWidget(buttons)

        # Initialize ASCII/Decimal/Binary fields and the frame view from hex
        self._sync_from('hex')

    # --- CAN frame editor ---------------------------------------------------

    def _build_can_group(self):
        """The 'what actually goes on the wire' panel: ID, DLC, D0..D7."""
        group = QtWidgets.QGroupBox('CAN frame')
        grid = QtWidgets.QGridLayout(group)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)

        small = QtGui.QFont('Segoe UI', 9)

        id_label = QtWidgets.QLabel('ID (hex):')
        id_label.setFont(QtGui.QFont('Segoe UI', 10))
        grid.addWidget(id_label, 0, 0)
        grid.addWidget(self.can_id_edit, 0, 1, 1, 5)
        grid.addWidget(self.can_ext_check, 0, 6, 1, 3)

        dlc_label = QtWidgets.QLabel('DLC:')
        dlc_label.setFont(QtGui.QFont('Segoe UI', 10))
        self.dlc_spin = QtWidgets.QSpinBox()
        self.dlc_spin.setRange(0, 8)
        self.dlc_spin.setFont(QtGui.QFont('Segoe UI', 10))
        self.dlc_spin.setToolTip(
            'Number of data bytes in the frame.\n'
            'Raising it appends 00 bytes, lowering it drops the tail.')
        self.dlc_spin.valueChanged.connect(self._dlc_changed)
        grid.addWidget(dlc_label, 1, 0)
        grid.addWidget(self.dlc_spin, 1, 1)

        self.pad00_btn = QtWidgets.QPushButton('Pad 00')
        self.pad00_btn.setFont(small)
        self.pad00_btn.setToolTip('Fill up to 8 data bytes with 00')
        self.pad00_btn.clicked.connect(lambda: self._pad_to_eight(0x00))
        self.padff_btn = QtWidgets.QPushButton('Pad FF')
        self.padff_btn.setFont(small)
        self.padff_btn.setToolTip('Fill up to 8 data bytes with FF (J1939 style)')
        self.padff_btn.clicked.connect(lambda: self._pad_to_eight(0xFF))
        grid.addWidget(self.pad00_btn, 1, 2)
        grid.addWidget(self.padff_btn, 1, 3)

        # D0..D7 byte cells: the literal frame, editable byte by byte
        byte_font = QtGui.QFont('Consolas', 11)
        hex_validator = QtGui.QRegExpValidator(QtCore.QRegExp('[0-9A-Fa-f]{0,2}'))
        self.byte_edits = []
        for i in range(8):
            cap = QtWidgets.QLabel('D%d' % i)
            cap.setFont(small)
            cap.setAlignment(QtCore.Qt.AlignHCenter)
            grid.addWidget(cap, 2, i + 1)

            cell = QtWidgets.QLineEdit()
            cell.setFont(byte_font)
            cell.setMaxLength(2)
            cell.setFixedWidth(40)
            cell.setAlignment(QtCore.Qt.AlignHCenter)
            cell.setValidator(hex_validator)
            cell.textEdited.connect(self._bytes_edited)
            grid.addWidget(cell, 3, i + 1)
            self.byte_edits.append(cell)

        wire_caption = QtWidgets.QLabel('On the wire:')
        wire_caption.setFont(small)
        grid.addWidget(wire_caption, 4, 0, 1, 2)

        self.frame_preview = QtWidgets.QLabel()
        self.frame_preview.setFont(QtGui.QFont('Consolas', 10))
        self.frame_preview.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        grid.addWidget(self.frame_preview, 5, 0, 1, 9)

        help_label = QtWidgets.QLabel(self.CAN_HELP)
        help_label.setFont(small)
        help_label.setWordWrap(True)
        help_label.setObjectName('sectionLabel')
        grid.addWidget(help_label, 6, 0, 1, 9)

        return group

    def _current_bytes(self):
        """The payload as bytes, or None when the hex field is not valid."""
        try:
            return bytes.fromhex(self.hex_edit.text().replace(' ', ''))
        except ValueError:
            return None

    def _set_bytes(self, raw):
        """Write raw back into the hex field (which re-syncs everything else)."""
        self.hex_edit.setText(' '.join('%02X' % b for b in raw))

    def _bytes_edited(self):
        """A D0..D7 cell was typed into: rebuild the payload from the cells."""
        if self._updating:
            return
        count = self.dlc_spin.value()
        raw = bytes(int(self.byte_edits[i].text() or '0', 16) for i in range(count))
        self._set_bytes(raw)

    def _dlc_changed(self, value):
        """DLC drives the payload length: grow with 00, shrink from the tail."""
        if self._updating:
            return
        raw = self._current_bytes()
        if raw is None:
            return
        if len(raw) == value:
            return
        raw = raw[:value] if len(raw) > value else raw + bytes(value - len(raw))
        self._set_bytes(raw)

    def _pad_to_eight(self, fill):
        raw = self._current_bytes()
        if raw is None:
            return
        self._set_bytes(raw[:8] + bytes([fill]) * max(0, 8 - len(raw)))

    def _update_frame_preview(self):
        """Refresh the byte cells, the DLC spinner and the wire-format line."""
        if not self._can_mode:
            return
        raw = self._current_bytes()
        outer = self._updating
        self._updating = True
        try:
            if raw is not None:
                n = min(len(raw), 8)
                if self.dlc_spin.value() != len(raw) and len(raw) <= 8:
                    self.dlc_spin.setValue(len(raw))
                for i, cell in enumerate(self.byte_edits):
                    # Never rewrite the cell being typed into: normalising
                    # 'F' to '0F' mid-keystroke would eat the second digit.
                    if not cell.hasFocus():
                        cell.setText('%02X' % raw[i] if i < n else '')
                    # Cells past the DLC are not part of the frame; greying
                    # them out is what makes "FF is one byte" visible.
                    cell.setEnabled(i < self.dlc_spin.value())
        finally:
            self._updating = outer

        id_text = self.can_id_edit.text().strip()
        ext = self.can_ext_check.isChecked()
        if raw is None:
            self.frame_preview.setText('(data is not valid hex)')
            return
        if len(raw) > 8:
            self.frame_preview.setText(
                '%d data bytes - a classic CAN frame carries at most 8' % len(raw))
            return
        try:
            can_id = int(id_text, 16) if id_text else None
        except ValueError:
            can_id = None
        if can_id is None:
            id_part = '(no valid ID)'
        else:
            id_part = format_can_id(can_id, ext)
        data_part = ' '.join('%02X' % b for b in raw) if raw else '-'
        line = 'ID %s   %s   DLC %d   Data %s' % (
            id_part, 'EXT (29-bit)' if ext else 'STD (11-bit)',
            len(raw), data_part)
        if ext and can_id is not None:
            line += '   PGN %05X' % j1939_pgn(can_id)
        self.frame_preview.setText(line)

    # --- validation / conversion -------------------------------------------

    def accept(self):
        """Refuse to save an invalid hex payload (it would fail silently on send)."""
        try:
            raw = bytes.fromhex(self.hex_edit.text().replace(' ', ''))
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, 'Invalid Macro',
                'The hex payload is not valid. Fix it or cancel.')
            return
        can_id_text = self.can_id_edit.text().strip()
        if can_id_text:
            try:
                can_id = int(can_id_text, 16)
            except ValueError:
                QtWidgets.QMessageBox.warning(
                    self, 'Invalid Macro', 'The CAN ID is not valid hex.')
                return
            max_id = 0x1FFFFFFF if self.can_ext_check.isChecked() else 0x7FF
            if not 0 <= can_id <= max_id:
                QtWidgets.QMessageBox.warning(
                    self, 'Invalid Macro',
                    f'CAN ID {can_id:X} is out of range (max {max_id:X} for '
                    f'{"an extended" if self.can_ext_check.isChecked() else "a standard"} ID).')
                return
        if self._can_mode:
            # A CAN macro that cannot be sent is worth catching here rather
            # than as a status-bar line on the first click.
            if len(raw) > 8:
                QtWidgets.QMessageBox.warning(
                    self, 'Invalid Macro',
                    f'A CAN frame carries at most 8 data bytes (this one has {len(raw)}).')
                return
            if not can_id_text:
                # Not fatal - an empty slot is a legitimate thing to save -
                # but worth one confirmation, since the button would do
                # nothing but print a status line when clicked.
                answer = QtWidgets.QMessageBox.question(
                    self, 'No CAN ID',
                    'This macro has no CAN ID, so clicking it in CAN mode '
                    'will not send anything.\n\nSave it anyway?',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No)
                if answer != QtWidgets.QMessageBox.Yes:
                    return
        super().accept()

    def _mode_changed(self, index):
        self.input_stack.setCurrentIndex(index)

    def _sync_from(self, source):
        """Convert from the edited field to all other fields + preview."""
        if self._updating:
            return
        self._updating = True
        try:
            raw = None
            if source == 'hex':
                raw = bytes.fromhex(self.hex_edit.text())
            elif source == 'ascii':
                raw = self.ascii_edit.text().encode('ISO-8859-1')
            elif source == 'dec':
                parts = self.dec_edit.text().strip().split()
                raw = bytes([int(b) for b in parts]) if parts and parts != [''] else b''
            elif source == 'bin':
                parts = self.bin_edit.text().strip().split()
                raw = bytes([int(b, 2) for b in parts]) if parts and parts != [''] else b''

            if raw is not None:
                if source != 'hex':
                    self.hex_edit.setText(raw.hex().upper())
                if source != 'ascii':
                    self.ascii_edit.setText(raw.decode('ISO-8859-1'))
                if source != 'dec':
                    self.dec_edit.setText(' '.join(str(b) for b in raw))
                if source != 'bin':
                    self.bin_edit.setText(' '.join(format(b, '08b') for b in raw))
        except (ValueError, OverflowError):
            pass
        self._update_preview()
        self._updating = False
        self._update_frame_preview()

    def _update_preview(self):
        try:
            raw = bytes.fromhex(self.hex_edit.text())
            display = ''.join(c if 32 <= ord(c) < 127 else '.' for c in raw.decode('ISO-8859-1'))
            self.preview_label.setText(f"ASCII: {display}  ({len(raw)} bytes)")
        except ValueError:
            self.preview_label.setText("(invalid data)")


class MacroButton(QtWidgets.QPushButton):
    """A macro button that sends hex data on click. Right-click to edit.

    In CAN mode the payload is sent as a CAN frame using the macro's CAN ID
    (and extended flag) instead of raw serial bytes.
    """

    macroChanged = QtCore.pyqtSignal()

    def __init__(self, label, hex_data, send_callback, parent=None,
                 can_id='', can_ext=False):
        super().__init__(label, parent)
        self.hex_data = hex_data
        self.can_id = can_id
        self.can_ext = can_ext
        self._can_mode = False
        self.send_callback = send_callback
        self.setFont(QtGui.QFont('Segoe UI', 9))
        self.setFixedHeight(28)
        self.setMinimumWidth(40)
        # Ignored horizontally: long labels must not inflate the layout --
        # the 8 buttons share the row evenly and the label elides instead.
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.clicked.connect(lambda: self.send_callback(self))
        self.refresh_tooltip()

    def set_can_mode(self, is_can):
        self._can_mode = is_can
        self.refresh_tooltip()

    def refresh_tooltip(self):
        """Show the exact frame/bytes this button sends, and that it's editable."""
        payload = self.hex_data.strip() or '(empty)'
        if self._can_mode:
            if self.can_id.strip():
                try:
                    raw = bytes.fromhex(self.hex_data.replace(' ', ''))
                    data = ' '.join(f'{b:02X}' for b in raw) if raw else '-'
                    dlc = len(raw)
                except ValueError:
                    data, dlc = payload, '?'
                # Spelling out the DLC is the point: a one-byte macro is a
                # one-byte frame, and the tooltip should not let that surprise.
                self.setToolTip(
                    f'Send CAN frame\n'
                    f'ID {self.can_id.upper()}  '
                    f'{"EXT (29-bit)" if self.can_ext else "STD (11-bit)"}\n'
                    f'DLC {dlc}  Data {data}\n'
                    f'Right-click to edit')
            else:
                self.setToolTip('No CAN ID set for this macro\nRight-click to edit')
        else:
            self.setToolTip(f'Send: {payload}\nRight-click to edit')

    def paintEvent(self, event):
        """Draw the label elided so long macro names never clip mid-glyph."""
        painter = QtWidgets.QStylePainter(self)
        option = QtWidgets.QStyleOptionButton()
        self.initStyleOption(option)
        metrics = option.fontMetrics
        option.text = metrics.elidedText(
            self.text(), QtCore.Qt.ElideRight, max(self.width() - 14, 10))
        painter.drawControl(QtWidgets.QStyle.CE_PushButton, option)

    def _show_context_menu(self, pos):
        menu = QtWidgets.QMenu(self)
        edit_action = menu.addAction("Edit Macro...")
        action = menu.exec_(self.mapToGlobal(pos))
        if action == edit_action:
            self._edit_macro()

    def _edit_macro(self):
        dialog = MacroEditDialog(self.text(), self.hex_data, self,
                                 can_id=self.can_id, can_ext=self.can_ext,
                                 can_mode=self._can_mode)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.setText(dialog.label_edit.text())
            self.hex_data = dialog.hex_edit.text()
            self.can_id = dialog.can_id_edit.text().strip()
            self.can_ext = dialog.can_ext_check.isChecked()
            self.refresh_tooltip()
            self.macroChanged.emit()


class SerialSendView(QtWidgets.QWidget):

    serialSendSignal = QtCore.pyqtSignal(str)
    canSendSignal = QtCore.pyqtSignal(str, bool, str)  # id_hex, extended, data_hex

    def __init__(self, parent):
        super().__init__(parent)

        self.history = []
        self.history_index = 0
        self._can_mode = False

        send_font = QtGui.QFont('Segoe UI', 10)

        # CAN mode widgets (swap in for charMode / lineEnding)
        self.canIdEdit = QtWidgets.QLineEdit(self)
        self.canIdEdit.setPlaceholderText('ID (hex)')
        self.canIdEdit.setToolTip('CAN arbitration ID as hex, e.g. 123 or 18FEF100')
        self.canIdEdit.setFont(send_font)
        self.canIdEdit.setMinimumHeight(30)
        self.canIdEdit.setVisible(False)

        self.canExtCheck = QtWidgets.QCheckBox('Ext (29-bit)', self)
        self.canExtCheck.setToolTip('Send with an extended 29-bit identifier')
        self.canExtCheck.setFont(send_font)
        self.canExtCheck.setMinimumHeight(30)
        self.canExtCheck.setVisible(False)

        self.charMode = QtWidgets.QComboBox(self)
        self.charMode.addItems(['ASCII', 'HEX', 'BINARY'])
        self.charMode.setCurrentIndex(0)
        self.charMode.setMinimumHeight(30)
        self.charMode.setFont(send_font)
        self.charMode.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        self.lineEnding = QtWidgets.QComboBox(self)
        self.lineEnding.addItems([
            "No line ending",
            "LF '\\n', 0x0A",
            "CR '\\r', 0x0D",
            "Both CR LF '\\r\\n'",
        ])
        self.lineEnding.setCurrentIndex(1)
        self.lineEnding.setMinimumHeight(30)
        self.lineEnding.setFont(send_font)
        self.lineEnding.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        self.sendData = QtWidgets.QTextEdit(self)
        self.sendData.installEventFilter(self)
        self.sendData.setAcceptRichText(False)
        self.sendData.setMaximumHeight(31)
        self.sendData.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.sendData.textChanged.connect(self._strip_newlines)
        self.sendData.setFont(send_font)

        self.sendButton = QtWidgets.QPushButton('Send')
        self.sendButton.setObjectName('primaryButton')
        self.sendButton.clicked.connect(self.sendButtonClicked)
        self.sendButton.setFont(send_font)
        self.sendButton.setMinimumHeight(30)
        self.sendButton.setMinimumWidth(90)
        self.sendButton.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        # Periodic/repeat send: polling commands, keep-alives, CAN cyclic TX.
        # With Repeat checked, Send (or a macro click) re-sends the same
        # payload every N ms until unchecked / disconnect / mode switch.
        self.repeatCheck = QtWidgets.QCheckBox('Repeat', self)
        self.repeatCheck.setFont(send_font)
        self.repeatCheck.setMinimumHeight(30)
        self.repeatCheck.setToolTip(
            'Re-send the payload periodically after pressing Send\n'
            '(also works for macro buttons and CAN frames)')
        self.repeatSpin = QtWidgets.QSpinBox(self)
        self.repeatSpin.setRange(10, 60000)
        self.repeatSpin.setValue(1000)
        self.repeatSpin.setSuffix(' ms')
        self.repeatSpin.setFont(send_font)
        self.repeatSpin.setMinimumHeight(30)
        self.repeatSpin.setToolTip('Repeat interval')
        self._repeat_timer = QtCore.QTimer(self)
        self._repeat_timer.timeout.connect(self._repeat_fire)
        self._repeat_payload = None  # ('serial'|'raw', ...) or ('can', id, ext, data)
        self._repeat_failures = 0    # consecutive failed sends while repeating
        self.repeatCheck.toggled.connect(self._on_repeat_toggled)
        self.repeatSpin.valueChanged.connect(self._repeat_timer.setInterval)

        # Macro buttons (right-click to edit). Serial and CAN each have their
        # own set of eight; the buttons are reused and their contents swapped
        # on mode change, so a CAN frame macro never sits in the serial row.
        self._macro_sets = self._load_macro_sets()
        self._active_set = 'serial'
        self.macro_buttons = []
        for macro in self._macro_sets['serial']:
            btn = MacroButton(macro.get("label", ""), macro.get("hex", ""),
                              self._macro_clicked, self,
                              can_id=macro.get("can_id", ""),
                              can_ext=bool(macro.get("can_ext", False)))
            btn.macroChanged.connect(self._on_macro_changed)
            self.macro_buttons.append(btn)

        self.setLayout(QtWidgets.QGridLayout())

        self.layout().addWidget(HLine(),       0, 0, 1, NUM_MACRO_BUTTONS)
        for i, btn in enumerate(self.macro_buttons):
            self.layout().addWidget(btn,       2, i, 1, 1)
        # charMode/canIdEdit and lineEnding/canExtCheck share grid cells;
        # only one of each pair is visible at a time (Serial vs CAN mode).
        self.layout().addWidget(self.charMode, 1, 0, 1, 1)
        self.layout().addWidget(self.canIdEdit,         1, 0, 1, 1)
        self.layout().addWidget(self.sendData,          1, 1, 1, 3)
        self.layout().addWidget(self.repeatCheck,       1, 4, 1, 1)
        self.layout().addWidget(self.repeatSpin,        1, 5, 1, 1)
        self.layout().addWidget(self.lineEnding,        1, 6, 1, 1)
        self.layout().addWidget(self.canExtCheck,       1, 6, 1, 1)
        self.layout().addWidget(self.sendButton,        1, 7, 1, 1)
        self.layout().setHorizontalSpacing(6)
        self.layout().setVerticalSpacing(6)
        self.layout().setContentsMargins(2, 1, 2, 4)
        # Equal stretch for all 8 columns: macro buttons share the row evenly,
        # and the CAN ID field (col 0) no longer balloons to half the window
        # while the macros next to it get squeezed into elided stubs.
        for col in range(NUM_MACRO_BUTTONS):
            self.layout().setColumnStretch(col, 1)

    def _strip_newlines(self):
        """Remove newlines from input (single-line send field)."""
        text = self.sendData.toPlainText()
        if '\n' in text:
            self.sendData.blockSignals(True)
            self.sendData.setPlainText(text.replace('\n', ''))
            self.sendData.moveCursor(QTextCursor.End)
            self.sendData.blockSignals(False)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress and obj is self.sendData:
            if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter) and self.sendData.hasFocus():
                self._emit_send()
                return True
            elif event.key() == QtCore.Qt.Key_Up and self.sendData.hasFocus():
                if self.history and self.history_index < len(self.history):
                    self.history_index += 1
                    self.sendData.blockSignals(True)
                    self.sendData.clear()
                    self.sendData.insertPlainText(self.history[-self.history_index])
                    self.sendData.blockSignals(False)
                return True
            elif event.key() == QtCore.Qt.Key_Down and self.sendData.hasFocus():
                if self.history_index > 1:
                    self.history_index -= 1
                    self.sendData.blockSignals(True)
                    self.sendData.clear()
                    self.sendData.insertPlainText(self.history[-self.history_index])
                    self.sendData.blockSignals(False)
                elif self.history_index == 1:
                    self.history_index = 0
                    self.sendData.clear()
                return True
        return super().eventFilter(obj, event)

    def set_can_mode(self, is_can):
        """Swap the send row between serial (mode + line ending) and CAN
        (ID + extended flag) layouts, and swap in that mode's macro set."""
        self._can_mode = is_can
        self.charMode.setVisible(not is_can)
        self.lineEnding.setVisible(not is_can)
        self.canIdEdit.setVisible(is_can)
        self.canExtCheck.setVisible(is_can)
        self.sendData.setPlaceholderText(
            'Data bytes as hex, e.g. 01 A2 FF - sent as-is, no padding (max 8)'
            if is_can else '')
        self.sendData.setToolTip(
            'CAN data bytes as hex (max 8).\n'
            'The DLC is however many bytes you type - nothing is padded:\n'
            'FF is a 1-byte frame, FF 00 00 00 00 00 00 00 is an 8-byte one.'
            if is_can else '')
        self._activate_macro_set('can' if is_can else 'serial')
        for btn in self.macro_buttons:
            btn.set_can_mode(is_can)

    def sendRaw(self, raw_hex_data):
        oldmode = self.charMode.currentIndex()
        oldending = self.lineEnding.currentIndex()
        self.charMode.setCurrentText("HEX")
        self.lineEnding.setCurrentText("No line ending")
        self.serialSendSignal.emit(raw_hex_data)
        self.charMode.setCurrentIndex(oldmode)
        self.lineEnding.setCurrentIndex(oldending)

    def _macro_clicked(self, btn):
        """Send a macro: serial hex payload, or a CAN frame in CAN mode."""
        if self._can_mode:
            if not btn.can_id.strip():
                monitor = self.window()
                if hasattr(monitor, 'statusText'):
                    monitor.statusText.setText(
                        f'Macro "{btn.text()}" has no CAN ID (right-click to edit)')
                return
            payload = ('can', btn.can_id, btn.can_ext, btn.hex_data)
        else:
            payload = ('raw', btn.hex_data)
        self._send_payload(payload)
        self._arm_repeat(payload)

    def _record_history(self, text):
        """Add a sent command to history, skipping blanks and repeats."""
        if text and (not self.history or self.history[-1] != text):
            self.history.append(text)
            if len(self.history) > 200:
                del self.history[:-200]

    # --- periodic send -------------------------------------------------------

    def _send_payload(self, payload):
        if payload[0] == 'can':
            self.canSendSignal.emit(payload[1], payload[2], payload[3])
        elif payload[0] == 'raw':
            self.sendRaw(payload[1])
        else:
            self.serialSendSignal.emit(payload[1])

    def _arm_repeat(self, payload):
        """Start periodic re-send of payload if Repeat is checked."""
        if self.repeatCheck.isChecked():
            self._repeat_payload = payload
            self._repeat_failures = 0
            self._repeat_timer.start(self.repeatSpin.value())

    def _repeat_fire(self):
        if self._repeat_payload is not None:
            self._send_payload(self._repeat_payload)

    def _on_repeat_toggled(self, checked):
        if not checked:
            self._repeat_timer.stop()
            self._repeat_payload = None

    def note_send_result(self, ok):
        """Send outcome feedback from the monitor. A repeating payload that
        keeps failing (closed port, non-ACKing CAN bus where each attempt
        blocks 300 ms) must stop instead of hammering the connection and the
        GUI thread forever."""
        if not self._repeat_timer.isActive():
            return
        if ok:
            self._repeat_failures = 0
            return
        self._repeat_failures += 1
        if self._repeat_failures >= 3:
            self.stop_repeat()
            monitor = self.window()
            if hasattr(monitor, 'statusText'):
                monitor.statusText.setText(
                    'Repeat stopped: sending keeps failing')

    def stop_repeat(self):
        """Stop periodic sending (disconnect, mode switch)."""
        self._repeat_timer.stop()
        self._repeat_payload = None
        self._repeat_failures = 0
        if self.repeatCheck.isChecked():
            self.repeatCheck.setChecked(False)

    def _emit_send(self):
        text = self.sendData.toPlainText()
        if self._can_mode:
            payload = ('can', self.canIdEdit.text().strip(),
                       self.canExtCheck.isChecked(), text)
        else:
            payload = ('serial', text)
        self._send_payload(payload)
        self._record_history(text)
        if self.repeatCheck.isChecked():
            # Keep the text visible while it repeats
            self._arm_repeat(payload)
        else:
            self.sendData.clear()
        self.history_index = 0

    def sendButtonClicked(self):
        self._emit_send()

    # --- macro sets (one per mode) ------------------------------------------

    @staticmethod
    def _normalize_macro_list(macros, defaults):
        """Pad/truncate a stored list to the button count, dropping junk.

        A list saved by a build with a different button count still restores
        what it can instead of being thrown away wholesale.
        """
        if not isinstance(macros, list) or not macros:
            return [dict(m) for m in defaults]
        out = [m if isinstance(m, dict) else {}
               for m in macros[:NUM_MACRO_BUTTONS]]
        while len(out) < NUM_MACRO_BUTTONS:
            out.append(dict(defaults[len(out)]))
        return out

    def _capture_macro_set(self):
        """Copy what the buttons currently hold into the active set."""
        self._macro_sets[self._active_set] = [
            {"label": btn.text(), "hex": btn.hex_data,
             "can_id": btn.can_id, "can_ext": btn.can_ext}
            for btn in self.macro_buttons]

    def _activate_macro_set(self, name):
        """Store the visible macros, then load the other set onto the buttons."""
        if name == self._active_set:
            return
        self._capture_macro_set()
        self._active_set = name
        for btn, macro in zip(self.macro_buttons, self._macro_sets[name]):
            btn.setText(macro.get("label", ""))
            btn.hex_data = macro.get("hex", "")
            btn.can_id = macro.get("can_id", "")
            btn.can_ext = bool(macro.get("can_ext", False))
            btn.refresh_tooltip()

    def _get_macros_list(self, which=None):
        """Macro definitions for one mode ('serial'/'can'); default: active."""
        self._capture_macro_set()
        return [dict(m) for m in self._macro_sets[which or self._active_set]]

    def _load_macros_from_list(self, macros, which='serial'):
        """Restore one mode's macro set; refresh the buttons if it is active."""
        if not isinstance(macros, list):
            return
        self._macro_sets[which] = self._normalize_macro_list(
            macros, DEFAULT_CAN_MACROS if which == 'can' else DEFAULT_MACROS)
        if which == self._active_set:
            # _activate_macro_set is a no-op for the current set, so push the
            # freshly loaded definitions onto the buttons directly.
            for btn, macro in zip(self.macro_buttons, self._macro_sets[which]):
                btn.setText(macro.get("label", ""))
                btn.hex_data = macro.get("hex", "")
                btn.can_id = macro.get("can_id", "")
                btn.can_ext = bool(macro.get("can_ext", False))
                btn.refresh_tooltip()

    def _load_macro_sets(self):
        """Load both macro sets from the settings file, or use defaults."""
        serial, can = None, None
        try:
            # OSError (not just FileNotFoundError): a PermissionError here
            # used to crash the whole app during __init__. isinstance guards
            # against a non-dict top level (AttributeError on s.get).
            with open(SETTINGS_FILE, 'r') as f:
                s = json.load(f)
            if isinstance(s, dict):
                serial = s.get('macros', None)
                can = s.get('macros_can', None)
        except (OSError, json.JSONDecodeError, ValueError, UnicodeDecodeError):
            pass
        return {
            'serial': self._normalize_macro_list(serial, DEFAULT_MACROS),
            # Settings written before macro sets existed have no CAN set; the
            # defaults seed it rather than cloning the serial byte strings,
            # which would be meaningless frames with no ID.
            'can': self._normalize_macro_list(can, DEFAULT_CAN_MACROS),
        }

    def _on_macro_changed(self):
        """A button was edited: keep the active set in step, then persist."""
        self._capture_macro_set()
        self._save_macros()

    def _save_macros(self):
        """Persist macros via parent SerialMonitor (debounced)."""
        monitor = self.window()
        if hasattr(monitor, 'schedule_save'):
            monitor.schedule_save()


class ToolBar(QtWidgets.QToolBar):
    def __init__(self, parent):
        super().__init__('Serial Port', parent)
        self.setMovable(False)

        toolbar_font = QtGui.QFont('Segoe UI', 10)

        self.modeCombo = QtWidgets.QComboBox(self)
        self.modeCombo.addItems(['Serial', 'CAN'])
        self.modeCombo.setMinimumHeight(30)
        self.modeCombo.setFont(toolbar_font)
        self.modeCombo.setToolTip('Switch between serial terminal and CAN bus mode')
        self.addWidget(self.modeCombo)

        serial_label = QtWidgets.QLabel(' Serial Port: ')
        serial_label.setFont(QtGui.QFont('Segoe UI', 10))
        self._serial_label_action = self.addWidget(serial_label)

        self.portOpenButton = QtWidgets.QPushButton('Open')
        self.portOpenButton.setObjectName('primaryButton')
        self.portOpenButton.setCheckable(True)
        self.portOpenButton.setMinimumHeight(30)
        self.portOpenButton.setMinimumWidth(80)
        self.portOpenButton.setFont(toolbar_font)

        self.portScanButton = QtWidgets.QPushButton('Scan')
        self.portScanButton.setCheckable(False)
        self.portScanButton.clicked.connect(self.scan_button_Clicked)
        self.portScanButton.setMinimumHeight(30)
        self.portScanButton.setFont(toolbar_font)

        self.portNames = QtWidgets.QComboBox(self)
        self._populate_ports()
        self.portNames.setMinimumHeight(30)
        self.portNames.setFont(toolbar_font)

        self.baudRates = QtWidgets.QComboBox(self)
        self.baudRates.addItems([
            '9600', '14400', '19200', '28800', '31250', '38400', '51200',
            '56000', '57600', '76800', '115200', '128000', '230400', '256000', '921600'
        ])
        # Editable: non-standard rates (74880 ESP8266 boot, 250000 Marlin,
        # 1M-3M high-speed logging) are daily needs the fixed list blocked
        self.baudRates.setEditable(True)
        self.baudRates.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.baudRates.setValidator(
            QtGui.QIntValidator(1, 100_000_000, self.baudRates))
        self.baudRates.setToolTip(
            'Baud rate - pick a preset or type any custom rate\n'
            '(e.g. 74880, 250000, 2000000)')
        self.baudRates.setCurrentText('115200')
        self.baudRates.setMinimumHeight(30)
        self.baudRates.setFont(toolbar_font)

        self.dataBits = QtWidgets.QComboBox(self)
        self.dataBits.addItems(['5 bit', '6 bit', '7 bit', '8 bit'])
        self.dataBits.setCurrentIndex(3)
        self.dataBits.setMinimumHeight(30)
        self.dataBits.setFont(toolbar_font)

        self._parity = QtWidgets.QComboBox(self)
        self._parity.addItems(['No Parity', 'Even Parity', 'Odd Parity', 'Space Parity', 'Mark Parity'])
        self._parity.setCurrentIndex(0)
        self._parity.setMinimumHeight(30)
        self._parity.setFont(toolbar_font)

        self.stopBits = QtWidgets.QComboBox(self)
        self.stopBits.addItems(['One Stop', 'One And Half Stop', 'Two Stop'])
        self.stopBits.setCurrentIndex(0)
        self.stopBits.setMinimumHeight(30)
        self.stopBits.setFont(toolbar_font)

        self._flowControl = QtWidgets.QComboBox(self)
        self._flowControl.addItems(['No Flow Control', 'Hardware Control', 'Software Control'])
        self._flowControl.setCurrentIndex(0)
        self._flowControl.setFont(toolbar_font)
        self._flowControl.setMinimumHeight(30)

        # Line controls: usable while the port is open (that is their point -
        # DTR/RTS toggles reset ESP32/Arduino boards and drive bootstrap pins,
        # Break interrupts U-Boot/RTOS consoles)
        self.dtrCheck = QtWidgets.QCheckBox('DTR', self)
        self.dtrCheck.setChecked(True)
        self.dtrCheck.setFont(toolbar_font)
        self.dtrCheck.setToolTip('Data Terminal Ready line (toggle to reset many dev boards)')
        self.rtsCheck = QtWidgets.QCheckBox('RTS', self)
        self.rtsCheck.setChecked(True)
        self.rtsCheck.setFont(toolbar_font)
        self.rtsCheck.setToolTip('Request To Send line (ignored under hardware flow control)')
        self.breakButton = QtWidgets.QPushButton('Break', self)
        self.breakButton.setMinimumHeight(30)
        self.breakButton.setFont(toolbar_font)
        self.breakButton.setToolTip('Hold TX in break state for 300 ms')

        # CAN mode widgets (hidden until CAN mode is selected)
        can_label = QtWidgets.QLabel(' CAN: ')
        can_label.setFont(toolbar_font)

        self.canInterfaces = QtWidgets.QComboBox(self)
        self.canInterfaces.addItems(list(CAN_INTERFACES.keys()))
        self.canInterfaces.setMinimumHeight(30)
        self.canInterfaces.setFont(toolbar_font)
        self.canInterfaces.setToolTip(
            'CAN hardware backend (Kvaser CANlib / Ixxat VCI drivers must be installed)')

        self.canChannels = QtWidgets.QComboBox(self)
        self.canChannels.addItems([f'Channel {i}' for i in range(8)])
        self.canChannels.setMinimumHeight(30)
        self.canChannels.setFont(toolbar_font)

        self.canBitrates = QtWidgets.QComboBox(self)
        self.canBitrates.addItems([f'{b} kbit/s' for b in CAN_BITRATES])
        self.canBitrates.setCurrentText('500 kbit/s')
        self.canBitrates.setMinimumHeight(30)
        self.canBitrates.setFont(toolbar_font)

        self.addWidget(self.portOpenButton)
        # addWidget returns a QAction; keep them so the two widget groups can
        # be shown/hidden when the mode combo changes.
        self._serial_actions = [
            self._serial_label_action,
            self.addWidget(self.portNames),
            self.addWidget(self.portScanButton),
            self.addWidget(self.baudRates),
            self.addWidget(self.dataBits),
            self.addWidget(self._parity),
            self.addWidget(self.stopBits),
            self.addWidget(self._flowControl),
            self.addWidget(self.dtrCheck),
            self.addWidget(self.rtsCheck),
            self.addWidget(self.breakButton),
        ]
        self._can_actions = [
            self.addWidget(can_label),
            self.addWidget(self.canInterfaces),
            self.addWidget(self.canChannels),
            self.addWidget(self.canBitrates),
        ]
        for action in self._can_actions:
            action.setVisible(False)

    def set_can_mode(self, is_can):
        """Show the CAN widget group instead of the serial one (or back)."""
        for action in self._serial_actions:
            action.setVisible(not is_can)
        for action in self._can_actions:
            action.setVisible(is_can)

    def canChannel(self):
        return self.canChannels.currentIndex()

    def canBitrate(self):
        return int(self.canBitrates.currentText().split()[0]) * 1000

    def _populate_ports(self):
        previous = self.portNames.currentData()
        self.portNames.clear()
        for port in QSerialPortInfo().availablePorts():
            name = port.portName()
            desc = port.description()
            vid = port.vendorIdentifier()
            pid = port.productIdentifier()
            label = name
            if desc:
                label += f'  {desc}'
            if vid or pid:
                label += f'  [{vid:04X}:{pid:04X}]'
            self.portNames.addItem(label, name)
        # Keep the previously selected port selected if it is still present
        if previous:
            idx = self.portNames.findData(previous)
            if idx >= 0:
                self.portNames.setCurrentIndex(idx)

    def scan_button_Clicked(self):
        self._populate_ports()

    def serialControlEnable(self, flag):
        self.modeCombo.setEnabled(flag)
        self.portNames.setEnabled(flag)
        self.portScanButton.setEnabled(flag)
        self.baudRates.setEnabled(flag)
        self.dataBits.setEnabled(flag)
        self._parity.setEnabled(flag)
        self.stopBits.setEnabled(flag)
        self._flowControl.setEnabled(flag)
        self.canInterfaces.setEnabled(flag)
        self.canChannels.setEnabled(flag)
        self.canBitrates.setEnabled(flag)

    def baudRate(self):
        try:
            return int(self.baudRates.currentText())
        except ValueError:
            return 115200  # empty custom-baud field

    def portName(self):
        return self.portNames.currentData() or self.portNames.currentText()

    def dataBit(self):
        return int(self.dataBits.currentIndex() + 5)

    def parity(self):
        if self._parity.currentIndex() > 0:
            return self._parity.currentIndex() + 1
        else:
            return self._parity.currentIndex()

    def stopBit(self):
        return STOP_BIT_VALUES[self.stopBits.currentIndex()]

    def flowControl(self):
        return self._flowControl.currentIndex()


def _install_excepthook():
    """Keep the app alive if an unhandled exception escapes a Qt slot.

    Under PyQt5 an exception propagating out of a slot (e.g. the RX/flush/decode
    path) aborts the whole process. Log it to stderr and carry on instead.
    """
    import traceback

    def hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        traceback.print_exception(exc_type, exc_value, exc_tb)

    sys.excepthook = hook


if __name__ == '__main__':
    _install_excepthook()
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QIcon(create_connector_pixmap(CONNECTED_COLOR)))
    window = SerialMonitor()
    if not window._geometry_restored:
        screen = app.primaryScreen().availableGeometry()
        window.resize(screen.width() * 8 // 15, screen.height() * 3 // 5)
    window.show()
    app.exec()
