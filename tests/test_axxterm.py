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
