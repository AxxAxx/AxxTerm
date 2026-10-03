# -*- coding: utf-8 -*-
"""Headless tests for AxxTerm.

Covers the decode/parse/format logic and the GUI-level data path without a
display, using Qt's offscreen platform. Run with:

    QT_QPA_PLATFORM=offscreen python -m pytest tests/         (with pytest), or
    QT_QPA_PLATFORM=offscreen python tests/test_axxterm.py    (standalone)

The tests load AxxTerm.py as a module and point SETTINGS_FILE at a temp file
so they never touch a real AxxTerm_settings.json.
"""
import os
import sys
import json
import struct
import tempfile
import importlib.util

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PyQt5 import QtWidgets, QtCore

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, os.pardir, 'AxxTerm.py')

_spec = importlib.util.spec_from_file_location('axxterm', _SRC)
axx = importlib.util.module_from_spec(_spec)
sys.modules['axxterm'] = axx
_spec.loader.exec_module(axx)

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
_TMPDIR = tempfile.mkdtemp()
_seq = [0]


def _isolate_settings():
    """Point SETTINGS_FILE at a fresh, nonexistent path so each window starts
    from defaults (SerialMonitor saves on close, which would otherwise leak
    state between tests)."""
    _seq[0] += 1
    axx.SETTINGS_FILE = os.path.join(_TMPDIR, f'settings_{_seq[0]}.json')


def _fresh_monitor():
    _isolate_settings()
    return axx.SerialMonitor()


def _new_view(mode='Binary Stream', nch=3, pts=100, dtype='float32', endian='Little Endian'):
    win = _fresh_monitor()
    dv = win.serialDataView
    dv.data_mode.setCurrentText(mode)
    if mode != 'ASCII':
        dv.type_combo.setCurrentText(dtype)
        dv.endian_combo.setCurrentText(endian)
    dv.graph_channels.setValue(nch)
    dv.plot_length_spin.setValue(pts)
    dv.graph_mode.setChecked(True)
    dv._apply_reader_settings()
    return win, dv


# --- decoders --------------------------------------------------------------

def test_binary_reader_chunk_boundaries():
    r = axx.BinaryStreamReader(); r.data_type = 'float32'; r.num_channels = 2
    payload = struct.pack('<4f', 1.0, 2.0, 3.0, 4.0)
    out = []
    for b in range(len(payload)):  # feed one byte at a time
        out += r.feed(payload[b:b + 1])
    assert out == [(1.0, 2.0), (3.0, 4.0)]


def test_binary_reader_np_matches_struct():
    r = axx.BinaryStreamReader(); r.data_type = 'int16'; r.num_channels = 3; r.endianness = 'big'
    data = struct.pack('>6h', 1, 2, 3, 4, 5, 6) + b'\x00'  # 1 leftover byte
    arr = r.feed_np(data)
    assert arr.shape == (2, 3)
    assert arr.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert len(r.buffer) == 1  # leftover preserved


def test_frame_reader_fixed_and_resync():
    f = axx.FrameReader()
    f.sync_word = b'\xAA'; f.size_field = 'fixed'; f.frame_size = 8
    f.num_channels = 2; f.data_type = 'float32'
    frame = b'\xAA' + struct.pack('<2f', 5.0, 6.0)
    out = f.feed(b'junk' + frame + b'\x00\x01' + frame)
    assert out == [(5.0, 6.0), (5.0, 6.0)]


def test_frame_reader_checksum_resync():
    f = axx.FrameReader()
    f.sync_word = b'\xAA\xBB'; f.size_field = '1-byte'; f.checksum_enabled = True
    f.num_channels = 1; f.data_type = 'uint8'

    def mk(vals):
        p = bytes(vals)
        return b'\xAA\xBB' + bytes([len(p)]) + p + bytes([sum(p) & 0xFF])

    bad = bytearray(mk([1, 2])); bad[-1] ^= 0xFF  # corrupt checksum
    out = f.feed(bytes(bad) + mk([7, 8]) + mk([7, 8]))
    assert out == [(7,), (8,), (7,), (8,)]  # recovers after the bad frame


def test_frame_reader_repeated_prefix_sync():
    f = axx.FrameReader()
    f.sync_word = b'\x01\x02\x01\x03'; f.size_field = 'fixed'; f.frame_size = 1
    f.num_channels = 1; f.data_type = 'uint8'
    out = f.feed(b'\x01\x02' + b'\x01\x02\x01\x03' + b'\x2A')
    assert out == [(42,)]


# --- converters ------------------------------------------------------------

def test_converters():
    C = axx.CONVERTERS
    assert C['ASCII --> BINARY']('AB') == '01000001 01000010'
    assert C['HEX --> BINARY']('0F') == '00001111'
    assert C['BINARY --> DECIMAL']('0000 1111') == '15'
    assert C['HEX --> DECIMAL']('FF') == '255'


# --- hex formatter ---------------------------------------------------------

def test_hex_formatter_wraps_and_continues():
    win, dv = _new_view()
    dv._hex_col = 0
    assert dv._format_hex(bytes(range(16))) == \
        '00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F\n'
    assert dv._hex_col == 0
    assert dv._format_hex(b'\xAA') == 'AA' and dv._hex_col == 1
    assert dv._format_hex(b'\xBB') == ' BB' and dv._hex_col == 2  # continuation space
    win.close()


def test_hex_view_bytes_above_0x80():
    win, dv = _new_view(mode='ASCII')
    dv._hex_col = 0
    dv.serialDataHex.clear()
    dv.appendSerialText('\xff\x80', 'read')
    assert dv.serialDataHex.toPlainText() == 'FF 80'
    win.close()


# --- math sandbox ----------------------------------------------------------

def test_math_sandbox_blocks_code_exec():
    win, dv = _new_view()
    assert dv._compile_math_expression('ch0 * 2 + np.sin(ch1)') is not None
    for evil in ("__import__('os')", "np.__loader__", "ch0.__class__",
                 "[x for x in (1,2)]", "(lambda: 1)()"):
        assert dv._compile_math_expression(evil) is None, evil
    win.close()


# --- ASCII parsing alignment ----------------------------------------------

def test_ascii_parse_keeps_channel_alignment():
    win, dv = _new_view(mode='ASCII', nch=3)
    vals = dv._parse_plot_values('1.0,bad,3.0')   # middle field non-numeric
    assert vals[0] == 1.0 and vals[2] == 3.0
    assert vals[1] != vals[1]                       # NaN gap, not a shift
    assert dv._parse_plot_values('1,2,3,') == [1.0, 2.0, 3.0]  # trailing delim dropped
    win.close()


# --- batched plot path -----------------------------------------------------

def test_binary_plot_values_and_chunk_split():
    win, dv = _new_view(nch=3, pts=100)
    win._rx_buffer.extend(struct.pack('<9f', 1, 2, 3, 4, 5, 6, 7, 8, 9))
    win._flush_display()
    assert dv._plot_fill == 3
    assert (dv.plot_data[0][-1], dv.plot_data[1][-1], dv.plot_data[2][-1]) == (7, 8, 9)
    # partial sample split across flushes
    dv._clear_graph()
    data = struct.pack('<6f', 10, 11, 12, 13, 14, 15)
    win._rx_buffer.extend(data[:7]); win._flush_display()
    win._rx_buffer.extend(data[7:]); win._flush_display()
    assert (dv.plot_data[0][-1], dv.plot_data[2][-1]) == (13, 15)
    win.close()


def test_inf_nan_become_gaps():
    win, dv = _new_view(nch=3, pts=50)
    win._rx_buffer.extend(struct.pack('<3f', float('inf'), float('nan'), 5.0))
    win._flush_display()
    assert np.isnan(dv.plot_data[0][-1])  # inf -> nan
    assert np.isnan(dv.plot_data[1][-1])  # nan stays
    assert dv.plot_data[2][-1] == 5.0
    win.close()


def test_more_samples_than_window():
    win, dv = _new_view(nch=1, pts=10)
    payload = struct.pack('<20f', *[float(i) for i in range(20)])
    win._rx_buffer.extend(payload); win._flush_display()
    assert dv._plot_fill == 10
    assert dv.plot_data[0][-1] == 19.0 and dv.plot_data[0][0] == 10.0
    win.close()


# --- per-channel scale / offset / units -----------------------------------

def test_channel_scale_offset_applied():
    win, dv = _new_view(nch=2, pts=50)
    dv.channel_scale[0] = 2.0
    dv.channel_offset[0] = 10.0
    dv.channel_units[0] = 'V'
    dv._invalidate_scale_cache()
    win._rx_buffer.extend(struct.pack('<2f', 3.0, 7.0))
    win._flush_display()
    assert dv.plot_data[0][-1] == 3.0 * 2.0 + 10.0  # scaled+offset
    assert dv.plot_data[1][-1] == 7.0               # untouched channel
    assert dv._channel_display(0) == 'Ch 0 (V)'
    win.close()


# --- time-based X axis -----------------------------------------------------

def test_time_axis_measures_rate_and_scales():
    win, dv = _new_view(nch=1, pts=1000)
    dv.set_x_time_mode(True)
    assert dv._x_time_mode
    # feed two batches with a measurable elapsed time
    import time
    win._rx_buffer.extend(struct.pack('<100f', *[1.0] * 100)); win._flush_display()
    time.sleep(0.25)
    win._rx_buffer.extend(struct.pack('<100f', *[1.0] * 100)); win._flush_display()
    assert dv._x_rate > 0
    axis = dv.graphWidget.plotItem.getAxis('bottom')
    assert abs(axis.scale - 1.0 / dv._x_rate) < 1e-9
    win.close()


# --- settings round trip ---------------------------------------------------

def test_settings_round_trip_preserves_channel_config():
    win, dv = _new_view(nch=4, pts=200)
    dv.channel_names[1] = 'Speed'
    dv.channel_scale[1] = 0.5
    dv.channel_offset[1] = -1.0
    dv.channel_units[1] = 'rpm'
    dv.set_x_time_mode(True)
    path = os.path.join(tempfile.gettempdir(), 'axxterm_rt.json')
    win.save_all_settings(path)
    win2 = _fresh_monitor()
    win2.load_all_settings(path)
    dv2 = win2.serialDataView
    assert dv2.channel_names.get(1) == 'Speed'
    assert dv2.channel_scale.get(1) == 0.5
    assert dv2.channel_offset.get(1) == -1.0
    assert dv2.channel_units.get(1) == 'rpm'
    assert dv2._x_time_mode is True
    os.remove(path)
    win.close(); win2.close()


def test_math_expr_cannot_mutate_channel_buffer():
    win, dv = _new_view(nch=2, pts=10)
    dv.plot_data[0][:] = np.arange(10, 0, -1)  # 10..1 descending
    before = dv.plot_data[0].copy()
    # In-place mutators must fail (channels are bound as read-only views) rather
    # than silently corrupt the live plot buffer.
    assert dv._eval_math_expression('ch0.sort()') is None
    assert np.array_equal(dv.plot_data[0], before)
    # A normal read-only expression still evaluates correctly.
    out = dv._eval_math_expression('ch0 * 2')
    assert np.allclose(out, before * 2)
    # A pass-through expression returns a usable (copied, writable) array.
    out2 = dv._eval_math_expression('ch0')
    assert np.array_equal(out2, before)
    win.close()


def test_ascii_parse_keeps_explicit_trailing_nan():
    win, dv = _new_view(mode='ASCII', nch=3)
    vals = dv._parse_plot_values('1,2,nan')       # explicit nan in last column
    assert len(vals) == 3 and vals[0] == 1.0 and vals[1] == 2.0
    assert vals[2] != vals[2]                       # kept as NaN, not stripped
    vals2 = dv._parse_plot_values('1,2,err')      # non-numeric token, last column
    assert len(vals2) == 3 and vals2[2] != vals2[2]
    assert dv._parse_plot_values('1,2,3,') == [1.0, 2.0, 3.0]  # empty field still dropped
    win.close()


def test_fft_dc_amplitude_not_doubled():
    win, dv = _new_view(nch=1, pts=64)
    dv._fft_check.setChecked(True)  # create the FFT widget + lines
    dv.plot_data[0][:] = 3.0
    dv._plot_fill = 64
    dv._update_fft()
    mag = dv._fft_lines[0].yData
    # DC bin must reflect the true amplitude (~3.0), not the 2x single-sided
    # factor that only applies to non-DC bins (~6.0).
    assert abs(mag[0] - 3.0) < 0.2
    win.close()


# --- CAN mode ----------------------------------------------------------------

def _can_msg(can_id, data=b'', ext=False, ts=100.0, rtr=False, err=False):
    """Minimal stand-in for can.Message (tests must not require python-can)."""
    import types
    return types.SimpleNamespace(
        arbitration_id=can_id, is_extended_id=ext, data=bytes(data),
        dlc=len(data), timestamp=ts, is_remote_frame=rtr, is_error_frame=err)


def test_j1939_pgn():
    # PDU2 (PF >= 0xF0): PS is part of the PGN
    assert axx.j1939_pgn(0x18FEF100) == 0xFEF1
    # PDU1 (PF < 0xF0): PS is a destination address, cleared from the PGN
    assert axx.j1939_pgn(0x18EF1234) == 0xEF00
    # Data page bit is included
    assert axx.j1939_pgn(0x19FEF100) == 0x1FEF1


def test_can_id_and_log_formatting():
    assert axx.format_can_id(0x123, False) == '123'
    assert axx.format_can_id(0x18FEF100, True) == '18FEF100'
    line = axx.format_can_log_line(_can_msg(0x18FEF100, b'\x01\xAB', ext=True))
    assert line == 'CAN 18FEF100 EXT PGN 0FEF1 DLC 2 DATA 01 AB'
    line = axx.format_can_log_line(_can_msg(0x7F, b''))
    assert line == 'CAN 07F STD DLC 0 DATA -'


def test_can_model_scrolling_rows_and_dt():
    m = axx.CanFrameModel()
    m.add_frames([('RX', _can_msg(0x100, b'\x01', ts=10.0)),
                  ('RX', _can_msg(0x200, b'\x02', ts=10.5)),
                  ('RX', _can_msg(0x100, b'\x03', ts=11.25))])
    assert m.rowCount() == 3
    # Row: Time, dt, Count, Dir, Type, ID, PGN, DLC, Data
    assert m.rows[0] == ('0.0000', '', '1', 'RX', 'STD', '100', '', '1', '01')
    assert m.rows[1][0] == '0.5000' and m.rows[1][5] == '200'
    # dt is time since the previous frame with the SAME ID (11.25 - 10.0)
    assert m.rows[2][1] == '1.2500' and m.rows[2][2] == '2'


def test_can_model_fixed_overwrites_in_place():
    m = axx.CanFrameModel()
    m.set_fixed_mode(True)
    m.add_frames([('RX', _can_msg(0x100, b'\x01', ts=10.0)),
                  ('RX', _can_msg(0x200, b'\x02', ts=10.5))])
    m.add_frames([('RX', _can_msg(0x100, b'\xEE\xFF', ts=12.0))])
    assert m.rowCount() == 2  # same ID overwrote its row
    row = m.rows[0]
    # Count for 0x100 is 2 (two frames), dt = 12.0 - 10.0, data overwritten
    assert row[5] == '100'
    assert row[2] == '2' and row[1] == '2.0000' and row[8] == 'EE FF'
    # TX and RX with the same ID stay on separate rows
    m.add_frames([('TX', _can_msg(0x100, b'\x00', ts=13.0))])
    assert m.rowCount() == 3


def test_can_model_mode_switch_keeps_timing_state():
    m = axx.CanFrameModel()
    m.add_frames([('RX', _can_msg(0x100, b'', ts=10.0))])
    m.set_fixed_mode(True)
    assert m.rowCount() == 0  # rows restart on mode switch
    m.add_frames([('RX', _can_msg(0x100, b'', ts=11.0))])
    # dt/count survive the switch: count=2, dt=1.0
    assert m.rows[0][2] == '2' and m.rows[0][1] == '1.0000'


def test_can_model_scroll_cap():
    m = axx.CanFrameModel()
    old_cap = axx.CAN_MAX_SCROLL_ROWS
    axx.CAN_MAX_SCROLL_ROWS = 5
    try:
        m.add_frames([('RX', _can_msg(i, b'', ts=float(i))) for i in range(8)])
        assert m.rowCount() == 5
        assert m.rows[0][5] == '003'  # oldest rows dropped
    finally:
        axx.CAN_MAX_SCROLL_ROWS = old_cap


def test_can_macros_settings_round_trip():
    win = _fresh_monitor()
    sv = win.serialSendView
    btn = sv.macro_buttons[0]
    btn.setText('EEC1')
    btn.hex_data = '0102030405060708'
    btn.can_id = '18FEF100'
    btn.can_ext = True
    win.save_all_settings()
    win.close()

    win2 = axx.SerialMonitor()  # same settings file -> restores the macro
    btn2 = win2.serialSendView.macro_buttons[0]
    assert btn2.can_id == '18FEF100' and btn2.can_ext is True
    assert btn2.hex_data == '0102030405060708'
    _isolate_settings()  # do not leak this settings file into later tests
    win2.close()


def test_can_mode_switch_swaps_ui():
    win = _fresh_monitor()
    win.toolBar.modeCombo.setCurrentText('CAN')
    assert win.canView.isVisibleTo(win) and not win.serialDataView.isVisibleTo(win)
    assert win.serialSendView._can_mode
    assert win.serialSendView.canIdEdit.isVisibleTo(win.serialSendView)
    assert not win.serialSendView.charMode.isVisibleTo(win.serialSendView)
    win.toolBar.modeCombo.setCurrentText('Serial')
    assert win.serialDataView.isVisibleTo(win) and not win.canView.isVisibleTo(win)
    assert not win.serialSendView._can_mode
    win.close()


def test_can_send_validation():
    win = _fresh_monitor()
    win.toolBar.modeCombo.setCurrentText('CAN')
    # No bus open -> error message, no crash
    win.sendCanFrame('123', False, '01')
    assert 'not open' in win.statusText.text()
    win.close()


def test_custom_baud_rate_entry():
    win = _fresh_monitor()
    win.toolBar.baudRates.setCurrentText('250000')  # not in the preset list
    assert win.toolBar.baudRate() == 250000
    win.toolBar.baudRates.setCurrentText('')
    assert win.toolBar.baudRate() == 115200  # safe fallback, no crash
    win.close()


def test_display_timestamps_prefix_each_line():
    win = _fresh_monitor()
    dv = win.serialDataView
    dv.show_timestamps = True
    dv.appendSerialText('one\ntwo', 'read')      # 'two' has no newline yet
    dv.appendSerialText(' more\nthree\n', 'read')  # continuation, then new line
    text = dv.serialData.toPlainText()
    lines = text.split('\n')
    assert lines[0].startswith('[') and lines[0].endswith('one')
    assert lines[1].startswith('[') and lines[1].endswith('two more')
    # continuation chunk must NOT get a second stamp mid-line
    assert lines[1].count('[') == 1
    assert lines[2].startswith('[') and lines[2].endswith('three')
    win.close()


def test_repeat_send_reemits_payload():
    win = _fresh_monitor()
    sv = win.serialSendView
    sent = []
    sv.serialSendSignal.connect(lambda t: sent.append(t))
    sv.repeatCheck.setChecked(True)
    sv.repeatSpin.setValue(50)
    sv.sendData.setPlainText('ping')
    sv._emit_send()
    assert sent == ['ping']
    assert sv._repeat_timer.isActive()
    assert sv.sendData.toPlainText() == 'ping'  # stays visible while repeating
    sv._repeat_fire()
    sv._repeat_fire()
    assert sent == ['ping', 'ping', 'ping']
    sv.repeatCheck.setChecked(False)  # uncheck stops the repeat
    assert not sv._repeat_timer.isActive() and sv._repeat_payload is None
    win.close()


def test_stop_repeat_on_disconnect_and_mode_switch():
    win = _fresh_monitor()
    sv = win.serialSendView
    sv.repeatCheck.setChecked(True)
    sv.sendData.setPlainText('x')
    sv._emit_send()
    assert sv._repeat_timer.isActive()
    win.toolBar.modeCombo.setCurrentText('CAN')  # mode switch kills the repeat
    assert not sv._repeat_timer.isActive()
    assert not sv.repeatCheck.isChecked()
    win.close()


def test_can_view_id_filter():
    win = _fresh_monitor()
    cv = win.canView

    class _M:
        def __init__(self, cid):
            self.arbitration_id = cid

    cv.filter_edit.setText('123, 18FEF100')
    assert cv._frame_passes(_M(0x123)) and cv._frame_passes(_M(0x18FEF100))
    assert not cv._frame_passes(_M(0x456))
    cv.filter_edit.setText('!456')
    assert cv._frame_passes(_M(0x123)) and not cv._frame_passes(_M(0x456))
    cv.filter_edit.setText('')
    assert cv._frame_passes(_M(0x456))
    win.close()


def test_freeze_buffers_display_and_resumes():
    win = _fresh_monitor()
    win.serialDataView.freeze_btn.setChecked(True)
    win._rx_buffer.extend(b'hello\n')
    win._flush_display()
    assert win._rx_buffer  # frozen: nothing consumed
    assert 'hello' not in win.serialDataView.serialData.toPlainText()
    win.serialDataView.freeze_btn.setChecked(False)
    win._flush_display()
    assert not win._rx_buffer
    assert 'hello' in win.serialDataView.serialData.toPlainText()
    win.close()


def test_trigger_level_compares_scaled_values():
    win, dv = _new_view(mode='ASCII', nch=1)
    dv.channel_scale = {0: 10.0}
    dv._invalidate_scale_cache()
    dv._trigger_enabled = True
    dv._trigger_armed = True
    dv._trigger_channel = 0
    dv._trigger_level = 5.0   # displayed units (raw * 10)
    dv._trigger_edge = 'rising'
    dv._ingest_row([0.4])    # scaled 4.0 - below level
    dv._ingest_row([0.6])    # scaled 6.0 - crosses level: must fire
    assert dv._trigger_countdown > 0, 'trigger did not fire in scaled units'
    win.close()


def test_settings_restore_survives_bad_values():
    _isolate_settings()
    bad = {
        'plot': {
            'num_points': 500,
            'channel_scale': {'0': 'not-a-number'},  # malformed entry
            'show_plot': False,
        },
    }
    with open(axx.SETTINGS_FILE, 'w') as f:
        json.dump(bad, f)
    win = axx.SerialMonitor()
    # the malformed scale is skipped; everything else still restores
    assert win.serialDataView.plot_length_spin.value() == 500
    assert win.serialDataView.channel_scale == {}
    _isolate_settings()
    win.close()


def test_load_settings_returns_false_on_garbage():
    win = _fresh_monitor()
    garbage = os.path.join(_TMPDIR, 'garbage.json')
    with open(garbage, 'w') as f:
        f.write('this is not json')
    assert win.load_all_settings(garbage) is False
    lst = os.path.join(_TMPDIR, 'list.json')
    with open(lst, 'w') as f:
        json.dump([1, 2, 3], f)
    assert win.load_all_settings(lst) is False
    win.close()


def test_invalid_sync_word_warns_and_keeps_old():
    win = _fresh_monitor()
    dv = win.serialDataView
    dv.data_mode.setCurrentText('Custom Frame')
    old_sync = dv._frame_reader.sync_word
    dv.sync_word_edit.setText('XYZ')  # not hex
    dv._apply_reader_settings()
    assert dv._frame_reader.sync_word == old_sync
    assert 'Invalid start byte' in win.statusText.text()
    win.close()


def test_can_fixed_view_capped_against_id_churn():
    model = axx.CanFrameModel()
    model.set_fixed_mode(True)
    old_cap = axx.CAN_MAX_SCROLL_ROWS
    axx.CAN_MAX_SCROLL_ROWS = 5
    try:
        class _M:
            def __init__(self, cid):
                self.arbitration_id = cid
                self.is_extended_id = False
                self.data = b'\x01'
                self.dlc = 1
                self.timestamp = 1.0
        model.add_frames([('RX', _M(i)) for i in range(20)])
        assert model.rowCount() == 5
    finally:
        axx.CAN_MAX_SCROLL_ROWS = old_cap


def _pycan_available():
    try:
        axx._ensure_pycan()
        return True
    except ImportError:
        return False  # python-can not installed; CAN tests skip


def _wait_for(cond, timeout=5.0):
    """Pump the event loop until cond() is true (bus open/close is async)."""
    deadline = axx.time.monotonic() + timeout
    while axx.time.monotonic() < deadline:
        _app.processEvents()
        if cond():
            return True
        axx.time.sleep(0.01)
    return cond()


def test_heavy_modules_import_lazily():
    """Importing AxxTerm must not import python-can or pyqtgraph (together
    they cost ~6 s on a cold start); they load on first CAN open / plot use."""
    import subprocess
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('axxterm', {os.path.abspath(_SRC)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "sys.modules['axxterm'] = m\n"
        "spec.loader.exec_module(m)\n"
        "assert 'pyqtgraph' not in sys.modules, 'pyqtgraph imported eagerly'\n"
        "assert 'can' not in sys.modules, 'python-can imported eagerly'\n"
    )
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_can_open_does_not_block_gui():
    """portOpen(True) must return immediately; the bus opens on a worker
    thread (real Kvaser/Ixxat drivers block for seconds during open)."""
    if not _pycan_available():
        return
    win = _fresh_monitor()
    win.toolBar.modeCombo.setCurrentText('CAN')
    win.toolBar.canInterfaces.setCurrentText('Virtual')
    orig = axx._create_can_bus

    def slow_create(interface, channel, bitrate):
        axx.time.sleep(0.5)
        return orig(interface, channel, bitrate)

    axx._create_can_bus = slow_create
    try:
        t = axx.time.monotonic()
        win.portOpen(True)
        elapsed = axx.time.monotonic() - t
        assert elapsed < 0.2, f'portOpen blocked the GUI thread for {elapsed:.2f}s'
        assert win._can_bus is None  # not open yet
        assert _wait_for(lambda: win._can_bus is not None)
    finally:
        axx._create_can_bus = orig
        win.portOpen(False)
        _wait_for(lambda: not win._can_closers or
                  not any(c.isRunning() for c in win._can_closers))
        win.close()


def test_can_open_cancelled_while_opening():
    """Closing (or switching mode) while the bus is still opening must not
    leave a stray open bus behind."""
    if not _pycan_available():
        return
    win = _fresh_monitor()
    win.toolBar.modeCombo.setCurrentText('CAN')
    win.toolBar.canInterfaces.setCurrentText('Virtual')
    orig = axx._create_can_bus

    def slow_create(interface, channel, bitrate):
        axx.time.sleep(0.3)
        return orig(interface, channel, bitrate)

    axx._create_can_bus = slow_create
    try:
        win.portOpen(True)
        win.portOpen(False)  # cancel while the opener is still running
        assert _wait_for(lambda: win._can_opener is None)
        assert win._can_bus is None
        # the cancelled bus is shut down by a closer thread
        assert _wait_for(lambda: not any(c.isRunning() for c in win._can_closers))
    finally:
        axx._create_can_bus = orig
        win.close()


def test_can_reopen_while_cancelled_open_in_flight():
    """Open -> close -> open again while the first (cancelled) open is still
    resolving must end with an open bus, not a silently dropped click."""
    if not _pycan_available():
        return
    win = _fresh_monitor()
    win.toolBar.modeCombo.setCurrentText('CAN')
    win.toolBar.canInterfaces.setCurrentText('Virtual')
    orig = axx._create_can_bus

    def slow_create(interface, channel, bitrate):
        axx.time.sleep(0.3)
        return orig(interface, channel, bitrate)

    axx._create_can_bus = slow_create
    try:
        win.portOpen(True)
        win.portOpen(False)   # cancel while opening
        win.portOpen(True)    # reopen before the cancelled opener resolves
        assert _wait_for(lambda: win._can_bus is not None)
    finally:
        axx._create_can_bus = orig
        win.portOpen(False)
        _wait_for(lambda: not any(c.isRunning() for c in win._can_closers))
        win.close()


def test_can_open_failure_after_cancel_leaves_ui_alone():
    """A cancelled open that then fails must not reset button/status state
    the user may have since repurposed (e.g. for a serial session)."""
    if not _pycan_available():
        return
    win = _fresh_monitor()
    win.toolBar.modeCombo.setCurrentText('CAN')
    win.toolBar.canInterfaces.setCurrentText('Virtual')

    def failing_create(interface, channel, bitrate):
        axx.time.sleep(0.2)
        raise RuntimeError('no device')

    orig, axx._create_can_bus = axx._create_can_bus, failing_create
    try:
        win.portOpen(True)
        win.portOpen(False)  # cancel while opening; UI reset here
        win.statusText.setText('sentinel')
        win.toolBar.portOpenButton.setChecked(True)  # user moved on
        assert _wait_for(lambda: win._can_opener is None)
        _app.processEvents()
        assert win.statusText.text() == 'sentinel'
        assert win.toolBar.portOpenButton.isChecked()
    finally:
        axx._create_can_bus = orig
        win.toolBar.portOpenButton.setChecked(False)
        win.close()


def test_kvaser_open_rejects_virtual_channels():
    """Kvaser opens must pass accept_virtual=False: the Kvaser driver always
    installs virtual channels, so without it Open 'succeeds' on a virtual
    channel even with no hardware attached."""
    captured = {}

    class _DummyPycan:
        @staticmethod
        def Bus(**kwargs):
            captured.update(kwargs)
            raise RuntimeError('capture only')

    orig, axx.pycan = axx.pycan, _DummyPycan()
    try:
        for iface, expect_flag in [('kvaser', True), ('virtual', False),
                                   ('ixxat', False)]:
            captured.clear()
            try:
                axx._create_can_bus(iface, 0, 500000)
            except RuntimeError:
                pass
            if expect_flag:
                assert captured.get('accept_virtual') is False, iface
            else:
                assert 'accept_virtual' not in captured, iface
    finally:
        axx.pycan = orig


def test_can_send_and_receive_virtual_bus():
    if not _pycan_available():
        return
    win = _fresh_monitor()
    win.toolBar.modeCombo.setCurrentText('CAN')
    win.toolBar.canInterfaces.setCurrentText('Virtual')
    win.portOpen(True)
    assert _wait_for(lambda: win._can_bus is not None)

    # A peer on the same virtual channel sees our TX and can answer
    peer = axx.pycan.Bus(interface='virtual', channel=win.toolBar.canChannel())
    try:
        win.sendCanFrame('18FEF100', True, '01 02 03 04 05 06 07 08')
        assert win.statusText.text() == ''
        got = peer.recv(timeout=2.0)
        assert got is not None and got.arbitration_id == 0x18FEF100
        assert bytes(got.data) == bytes(range(1, 9))

        peer.send(axx.pycan.Message(arbitration_id=0x321, is_extended_id=False,
                                    data=b'\xAA\xBB'))
        deadline = axx.time.time() + 2.0
        while axx.time.time() < deadline:
            _app.processEvents()
            if win.canView.model.rowCount() >= 2:
                break
            axx.time.sleep(0.02)
        win._flush_display()
        rows = win.canView.model.rows
        dirs_ids = {(r[3], r[5]) for r in rows}
        assert ('TX', '18FEF100') in dirs_ids
        assert ('RX', '321') in dirs_ids
        # RX row shows the data as hex
        rx_row = next(r for r in rows if r[3] == 'RX')
        assert rx_row[8] == 'AA BB' and rx_row[4] == 'STD'
    finally:
        peer.shutdown()
        win.portOpen(False)
        assert win._can_bus is None
        win.close()


def test_can_recording_logs_frames(tmp_path=None):
    if not _pycan_available():
        return
    win = _fresh_monitor()
    win.toolBar.modeCombo.setCurrentText('CAN')
    win.toolBar.canInterfaces.setCurrentText('Virtual')
    win.portOpen(True)
    assert _wait_for(lambda: win._can_bus is not None)
    win._toggle_recording()
    assert win._recording
    assert win._asc_writer is not None  # CAN recording mirrors to Vector .asc
    log_path = win._log_file.name
    asc_path = log_path[:-4] + '.asc'
    try:
        win.sendCanFrame('7F', False, 'AA')
        win._flush_display()
    finally:
        win._toggle_recording()
        win.portOpen(False)
    assert win._asc_writer is None  # stopped with recording
    with open(log_path, encoding='utf-8') as f:
        content = f.read()
    os.remove(log_path)
    assert 'TX: CAN 07F STD DLC 1 DATA AA' in content
    with open(asc_path, encoding='utf-8') as f:
        asc = f.read()
    os.remove(asc_path)
    assert '7f' in asc.lower() and 'aa' in asc.lower()
    win.close()


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f'  PASS  {fn.__name__}')
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f'  FAIL  {fn.__name__}: {e!r}')
    print(f'\n{len(fns) - failed}/{len(fns)} passed')
    return failed


if __name__ == '__main__':
    sys.exit(1 if _run_all() else 0)
