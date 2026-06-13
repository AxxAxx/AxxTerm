# -*- coding: utf-8 -*-
import sys
import ast
import html
import math
import struct
import os
import json
import time
from datetime import datetime
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from PyQt5.QtSerialPort import QSerialPort, QSerialPortInfo
from PyQt5.QtGui import QPixmap, QTextCursor, QIcon, QPainter, QColor
from PyQt5.QtWidgets import *
import numpy as np

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


def create_connector_pixmap(color, width=71, height=30):
    """Draw a DB-9 connector icon programmatically (no external PNG needed)."""
    pixmap = QPixmap(width, height)
    pixmap.fill(QtCore.Qt.transparent)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.Antialiasing)

    cy = height / 2.0

    # 1. Outer white rounded rectangle (metal shell)
    p.setPen(QtGui.QPen(QColor('#333333'), 1.5))
    p.setBrush(QColor('#FFFFFF'))
    p.drawRoundedRect(QtCore.QRectF(0.75, 0.75, width - 1.5, height - 1.5), 4, 4)

    # 2. Inner D-shaped colored area (trapezoid: wider at top, narrower at bottom)
    d_left = 15.0
    d_right = width - 15.0
    d_top = 4.0
    d_bot = height - 4.0
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
    p.setPen(QtGui.QPen(QColor('#333333'), 1.0))
    p.setBrush(QColor(color))
    p.drawPath(d_path)

    # 3. Mounting screws with Phillips cross-head
    screw_r = 5.0
    screw_lx = 8.0
    screw_rx = width - 8.0
    p.setPen(QtGui.QPen(QColor('#666666'), 1.0))
    p.setBrush(QColor('#DDDDDD'))
    p.drawEllipse(QtCore.QPointF(screw_lx, cy), screw_r, screw_r)
    p.drawEllipse(QtCore.QPointF(screw_rx, cy), screw_r, screw_r)
    cross = 3.0
    p.setPen(QtGui.QPen(QColor('#888888'), 1.0))
    for sx in [screw_lx, screw_rx]:
        p.drawLine(QtCore.QPointF(sx - cross, cy), QtCore.QPointF(sx + cross, cy))
        p.drawLine(QtCore.QPointF(sx, cy - cross), QtCore.QPointF(sx, cy + cross))

    # 4. Pin holes: 5 top row, 4 bottom row
    d_cx = (d_left + d_right) / 2.0
    pin_r = 1.7
    pin_spacing = 7.0
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(QColor('#111111'))
    for i in range(5):
        p.drawEllipse(QtCore.QPointF(d_cx + (i - 2) * pin_spacing, cy - 3.5), pin_r, pin_r)
    for i in range(4):
        p.drawEllipse(QtCore.QPointF(d_cx + (i - 1.5) * pin_spacing, cy + 3.5), pin_r, pin_r)

    p.end()
    return pixmap


class SerialMonitor(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.port = QSerialPort()
        self.serialDataView = SerialDataView(self)
        self.serialSendView = SerialSendView(self)

        self.setCentralWidget(QtWidgets.QWidget(self))
        self.layout = QtWidgets.QVBoxLayout(self.centralWidget())
        self.layout.addWidget(self.serialDataView)
        self.layout.addWidget(self.serialSendView)
        self.layout.setContentsMargins(3, 3, 3, 3)

        self.setWindowTitle('AxxTerm')
        self.setWindowIcon(QIcon(create_connector_pixmap('#22bb22')))

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
        self._record_btn.setCheckable(True)
        self._record_btn.setFixedHeight(22)
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
        self.port.readyRead.connect(self.readFromPort)
        self.port.errorOccurred.connect(self._on_port_error)

        # Save when serial port settings change (debounced)
        self.toolBar.baudRates.currentIndexChanged.connect(lambda: self.schedule_save())
        self.toolBar.dataBits.currentIndexChanged.connect(lambda: self.schedule_save())
        self.toolBar._parity.currentIndexChanged.connect(lambda: self.schedule_save())
        self.toolBar.stopBits.currentIndexChanged.connect(lambda: self.schedule_save())
        self.toolBar._flowControl.currentIndexChanged.connect(lambda: self.schedule_save())

        ### Load all settings ###
        self.load_all_settings()

    def portOpen(self, flag):
        # Manual open/close always cancels any pending auto-reconnect
        self._reconnect_timer.stop()
        self._reconnect_port_name = ''

        if flag:
            self.port.setBaudRate(self.toolBar.baudRate())
            self.port.setPortName(self.toolBar.portName())
            self.port.setDataBits(self.toolBar.dataBit())
            self.port.setParity(self.toolBar.parity())
            self.port.setStopBits(self.toolBar.stopBit())
            self.port.setFlowControl(self.toolBar.flowControl())

            r = self.port.open(QtCore.QIODevice.ReadWrite)
            if not r:
                self.statusText.setText('Port open error')
                self.toolBar.portOpenButton.setChecked(False)
                self.toolBar.serialControlEnable(True)
            else:
                self.statusText.setText('Port opened')
                self._rx_bytes = 0
                self._tx_bytes = 0
                self._rx_total = 0
                self._tx_total = 0
                self._reset_stream_state()
                self.toolBar.serialControlEnable(False)
                self.serialDataView.label.setPixmap(create_connector_pixmap('#22bb22'))
        else:
            self.port.close()
            self.statusText.setText('Port closed')
            self.toolBar.serialControlEnable(True)
            self.serialDataView.label.setPixmap(create_connector_pixmap('#cc2222'))

    def _reset_stream_state(self):
        """Drop buffered/partial stream state so a new connection starts clean."""
        self._rx_buffer.clear()
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

    def _flush_display(self):
        """Process buffered RX data and update the display (~30 fps)."""
        if self._rx_buffer:
            data = bytes(self._rx_buffer)
            self._rx_buffer.clear()
            self.serialDataView.handleReceivedData(data)
        # Push any pending plot samples to the curves (one setData per channel)
        self.serialDataView.flush_plot()

    def sendFromPort(self, text):
        if not self.port.isOpen():
            self.statusText.setText('Port is not open')
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
                self.port.write(tx)
                self._tx_bytes += len(tx)
                self._tx_total += len(tx)
                self.statusText.setText('')
                sent = True
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
                self.port.write(tx)
                self._tx_bytes += len(tx)
                self._tx_total += len(tx)
                self.statusText.setText('')
                sent = True
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
                self.port.write(tx)
                self._tx_bytes += len(tx)
                self._tx_total += len(tx)
                self.statusText.setText('')
                sent = True
            except (ValueError, OverflowError):
                self.statusText.setText('Not a valid BINARY string')

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

        if self.port.isOpen():
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
            "QToolTip { color: #ffffff; background-color: #2a2a2a; border: 1px solid white; }")

    def _apply_light_palette(self):
        QtWidgets.QApplication.instance().setPalette(
            QtWidgets.QApplication.style().standardPalette())
        QtWidgets.QApplication.instance().setStyleSheet("")

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

    def _on_port_error(self, error):
        """Handle serial port errors. On ResourceError (device unplugged), close
        gracefully and optionally start scanning for reconnection."""
        if error == QSerialPort.ResourceError and self.port.isOpen():
            self._reconnect_port_name = self.port.portName()
            self.port.close()
            self.toolBar.portOpenButton.setChecked(False)
            self.toolBar.serialControlEnable(True)
            self.serialDataView.label.setPixmap(create_connector_pixmap('#cc2222'))
            if self._auto_reconnect:
                self.statusText.setText(
                    f'Port disconnected - reconnecting to {self._reconnect_port_name}...')
                self._reconnect_timer.start(1000)
            else:
                self.statusText.setText('Port disconnected')
                self._reconnect_port_name = ''

    def _try_reconnect(self):
        """Poll available ports for the previously connected port name."""
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
            self.serialDataView.label.setPixmap(create_connector_pixmap('#22bb22'))
            self._rx_bytes = 0
            self._tx_bytes = 0
            self._rx_total = 0
            self._tx_total = 0
            # Drop partial frames/lines from before the disconnect so the
            # decoders don't permanently misalign on the reconnected stream
            self._reset_stream_state()
            self.statusText.setText('Port reconnected')

    # --- Recording / Logging ---------------------------------------------------

    def _log_data(self, direction, text):
        """Write a timestamped line to the log file if recording is active."""
        if not self._recording or self._log_file is None:
            return
        now = datetime.now()
        ts = now.strftime('%Y-%m-%d %H:%M:%S.') + f'{now.microsecond // 1000:03d}'
        try:
            for line in text.splitlines():
                if line:
                    self._log_file.write(f'[{ts}] {direction}: {line}\n')
            self._log_file.flush()
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
            self._record_btn.setStyleSheet('')
            self._record_action.setText('Start Recording')
            self.statusText.setText('Recording stopped: log file write failed')

    def _toggle_recording(self):
        """Start or stop recording serial data to a log file."""
        if self._recording:
            # Stop recording: flush any partial RX line first
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
            self._record_btn.setStyleSheet('')
            self._record_action.setText('Start Recording')
            self.statusText.setText('Recording stopped')
        else:
            # Start recording
            ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f'AxxTerm_log_{ts}.txt'
            filepath = os.path.join(SCRIPT_DIR, filename)
            try:
                self._log_file = open(filepath, 'w', encoding='utf-8')
            except OSError as e:
                self.statusText.setText(f'Cannot create log file: {e}')
                self._record_btn.setChecked(False)
                return
            self._recording = True
            self._record_btn.setChecked(True)
            self._record_btn.setText('Recording...')
            self._record_btn.setStyleSheet(
                'QPushButton { background-color: #cc2222; color: white; font-weight: bold; }')
            self._record_action.setText('Stop Recording')
            self.statusText.setText(f'Recording to {filename}')

    def closeEvent(self, event):
        """Flush pending state and close the log file when the application exits."""
        self._save_timer.stop()
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
        self._save_timer.start()

    def save_all_settings(self, path=None):
        """Save all settings (plot, serial port, macros) to one JSON file."""
        settings = {
            'dark_mode': self._dark_mode,
            'auto_reconnect': self._auto_reconnect,
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
            'macros': self.serialSendView._get_macros_list(),
        }
        try:
            with open(path or SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=2)
        except OSError:
            pass

    def load_all_settings(self, path=None):
        """Load all settings from one JSON file."""
        try:
            with open(path or SETTINGS_FILE, 'r') as f:
                s = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError, UnicodeDecodeError):
            return
        if not isinstance(s, dict):
            return

        # Window geometry / splitter sizes. We restore the size/position but
        # never reopen maximized or full-screen -- the app should always start
        # as a normal resizable window.
        win = s.get('window', {})
        try:
            geo = win.get('geometry', '')
            if geo:
                self.restoreGeometry(QtCore.QByteArray.fromHex(geo.encode('ascii')))
                self.setWindowState(self.windowState() & ~(
                    QtCore.Qt.WindowMaximized | QtCore.Qt.WindowFullScreen))
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

        # Plot/decode settings
        plot = s.get('plot', s)  # fallback: old format had plot keys at top level
        self.serialDataView._load_plot_settings(plot)

        # Serial port settings (block signals to avoid cascading saves)
        serial = s.get('serial', {})
        if serial:
            for w in [self.toolBar.baudRates, self.toolBar.dataBits,
                      self.toolBar._parity, self.toolBar.stopBits, self.toolBar._flowControl]:
                w.blockSignals(True)
            self.toolBar.baudRates.setCurrentText(serial.get('baud_rate', '115200'))
            self.toolBar.dataBits.setCurrentIndex(serial.get('data_bits', 3))
            self.toolBar._parity.setCurrentIndex(serial.get('parity', 0))
            self.toolBar.stopBits.setCurrentIndex(serial.get('stop_bits', 0))
            self.toolBar._flowControl.setCurrentIndex(serial.get('flow_control', 0))
            for w in [self.toolBar.baudRates, self.toolBar.dataBits,
                      self.toolBar._parity, self.toolBar.stopBits, self.toolBar._flowControl]:
                w.blockSignals(False)

        # Macros
        macros = s.get('macros', None)
        if macros:
            self.serialSendView._load_macros_from_list(macros)

        # Apply splitter sizes after the plot widget (if any) has been created
        if self._splitter_sizes_to_restore:
            sizes = self._splitter_sizes_to_restore
            self._splitter_sizes_to_restore = None
            if len(sizes) == self.serialDataView.splitter.count():
                self.serialDataView.splitter.setSizes(sizes)

    def _menu_save_settings(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Settings', '', 'JSON Files (*.json);;All Files (*)')
        if path:
            self.save_all_settings(path)
            self.statusText.setText(f'Settings saved to {os.path.basename(path)}')

    def _menu_load_settings(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Load Settings', '', 'JSON Files (*.json);;All Files (*)')
        if path:
            self.load_all_settings(path)
            self.statusText.setText(f'Settings loaded from {os.path.basename(path)}')

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
            with open(path, 'w') as f:
                f.write('\n'.join(lines) + '\n')
            self.statusText.setText(f'CSV exported to {os.path.basename(path)}')
        except OSError as e:
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

        # FFT view state
        self._fft_widget = None
        self._fft_lines = []
        self._fft_update_counter = 0

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
        self.label_data_flow.setFont(QtGui.QFont('Segoe UI', 10))
        self.label_data_flow.setIndent(5)

        self.label_sent_data = QtWidgets.QLabel('Data: ASCII')
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
        self.label.setPixmap(create_connector_pixmap('#cc2222'))

        self.converter_label = QtWidgets.QLabel('Converter')
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
        self.endian_combo.setMinimumWidth(70)
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
            # Clean up Y2 axis before destroying graph
            self._y2_plot_lines = {}
            self._y2_viewbox = None
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
            self._pause_btn.setStyleSheet(
                'background-color: #cc6600; color: #ffffff; border: 1px solid #995500; '
                'padding: 2px 8px; font-weight: bold;')
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
        html = '<br>'.join(parts)
        self._cursor_label.setHtml(f'<div style="background:rgba(255,255,255,200);padding:2px">{html}</div>')
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
        # Hann window + amplitude normalization; skip the NaN-prefilled head
        window = np.hanning(n)
        scale = 2.0 / window.sum()
        for i, arr in enumerate(self.plot_data):
            if i >= len(self._fft_lines):
                break
            if i in self.hidden_channels:
                continue
            data = np.nan_to_num(arr[n_total - n:])
            fft_mag = np.abs(np.fft.rfft(data * window)) * scale
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

    def _eval_math_expression(self, expression):
        """Evaluate a math expression safely and return the result array."""
        code = self._compile_math_expression(expression)
        if code is None:
            return None
        namespace = {'np': np, 'numpy': np}
        for i, arr in enumerate(self.plot_data):
            namespace[f'ch{i}'] = arr
        try:
            result = eval(code, {"__builtins__": {}}, namespace)
            # Ensure result is a numpy array of the right length
            if isinstance(result, (int, float)):
                result = np.full(len(self.plot_data[0]), result)
            result = np.asarray(result, dtype=float)
            if result.shape != self.plot_data[0].shape:
                return None
            # inf wrecks Y auto-range; render as a gap instead
            result[~np.isfinite(result)] = np.nan
            return result
        except Exception:
            return None

    def _update_math_channels(self):
        """Evaluate all math expressions and update their plot lines."""
        if not self._math_channels or not self._math_lines or not self.plot_data:
            return
        for i, mch in enumerate(self._math_channels):
            if i >= len(self._math_lines):
                break
            result = self._eval_math_expression(mch['expression'])
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
        if not np.isfinite(batch).all():
            batch = np.where(np.isfinite(batch), batch, np.nan)

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
        for field in fields:
            field = field.strip()
            if not field:
                # An empty field between delimiters is still a column: keep the
                # position so later values stay on their own channels.
                values.append(float('nan'))
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
        # Drop trailing NaNs so a line-ending delimiter (e.g. "1,2,3,")
        # doesn't add a phantom empty channel.
        while values and values[-1] != values[-1]:  # NaN check
            values.pop()
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

        try:
            sw = bytes.fromhex(self.sync_word_edit.text().replace(' ', ''))
            if len(sw) > 0:
                self._frame_reader.sync_word = sw
        except ValueError:
            pass

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

        try:
            self.data_mode.setCurrentText(s.get('mode', 'ASCII'))
            self.graph_channels.setValue(s.get('num_channels', 4))
            self.plot_length_spin.setValue(s.get('num_points', DEFAULT_PLOT_LENGTH))
            self.delimiter_combo.setCurrentText(s.get('delimiter', 'Auto'))
            self.delimiter_custom.setText(s.get('delimiter_custom', ''))

            # Load channel properties AND axis ranges BEFORE enabling the graph
            # so that graph_state_changed() sees the correct names/colors/axes/
            # hidden set and restores the saved manual Y range.
            saved_names = s.get('channel_names', {})
            self.channel_names = {int(k): v for k, v in saved_names.items()}
            saved_colors = s.get('channel_colors', {})
            self.channel_colors = {int(k): v for k, v in saved_colors.items()}
            saved_axes = s.get('channel_axes', {})
            self.channel_axes = {int(k): v for k, v in saved_axes.items()}
            self.channel_scale = {int(k): float(v) for k, v in s.get('channel_scale', {}).items()}
            self.channel_offset = {int(k): float(v) for k, v in s.get('channel_offset', {}).items()}
            self.channel_units = {int(k): v for k, v in s.get('channel_units', {}).items()}
            self._invalidate_scale_cache()
            self.hidden_channels = set(s.get('hidden_channels', []))

            self._x_time_mode = bool(s.get('x_time_mode', False))
            self._time_axis_check.blockSignals(True)
            self._time_axis_check.setChecked(self._x_time_mode)
            self._time_axis_check.blockSignals(False)

            self._y_auto_scale = s.get('y_auto_scale', True)
            self._y_min = s.get('y_min', -1.0)
            self._y_max = s.get('y_max', 1.0)

            # Math channels must be known before the graph is created
            saved_math = s.get('math_channels', [])
            if isinstance(saved_math, list):
                self._math_channels = [
                    {'name': m.get('name', ''), 'expression': m.get('expression', '')}
                    for m in saved_math
                    if isinstance(m, dict) and m.get('expression', '').strip()
                ]

            self._fft_check.setChecked(s.get('show_fft', False))
            self.graph_mode.setChecked(s.get('show_plot', False))

            frame = s.get('frame', {})
            binary = s.get('binary', {})
            dtype = frame.get('data_type', binary.get('data_type', 'float32'))
            endian = frame.get('endianness', binary.get('endianness', 'little'))

            self.type_combo.setCurrentText(dtype)
            self.endian_combo.setCurrentText('Little Endian' if endian == 'little' else 'Big Endian')
            self.sync_word_edit.setText(frame.get('sync_word', 'AA'))
            sf = frame.get('size_field', 'fixed')
            sf_map = {'fixed': 'Fixed', '1-byte': '1-byte size field', '2-byte': '2-byte size field'}
            self.size_field_combo.setCurrentText(sf_map.get(sf, 'Fixed'))
            self.frame_size_spin.setValue(frame.get('frame_size', 12))
            self.checksum_check.setChecked(frame.get('checksum', False))
        except (KeyError, TypeError, ValueError, AttributeError):
            pass

        for w in widgets:
            w.blockSignals(False)

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
                else:
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
    """Dialog for editing a macro button's label and hex payload."""

    def __init__(self, label, hex_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Macro Button")
        self.setMinimumWidth(450)
        self._updating = False

        layout = QtWidgets.QFormLayout(self)

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
        self._update_preview()

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addRow("Button Label:", self.label_edit)
        layout.addRow("Input Mode:", self.input_mode)
        layout.addRow("Data:", self.input_stack)
        layout.addRow("Preview:", self.preview_label)
        layout.addRow(buttons)

        # Initialize ASCII and Decimal fields from the hex data
        self._sync_from('hex')

    def accept(self):
        """Refuse to save an invalid hex payload (it would fail silently on send)."""
        try:
            bytes.fromhex(self.hex_edit.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, 'Invalid Macro',
                'The hex payload is not valid. Fix it or cancel.')
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

    def _update_preview(self):
        try:
            raw = bytes.fromhex(self.hex_edit.text())
            display = ''.join(c if 32 <= ord(c) < 127 else '.' for c in raw.decode('ISO-8859-1'))
            self.preview_label.setText(f"ASCII: {display}  ({len(raw)} bytes)")
        except ValueError:
            self.preview_label.setText("(invalid data)")


class MacroButton(QtWidgets.QPushButton):
    """A macro button that sends hex data on click. Right-click to edit."""

    macroChanged = QtCore.pyqtSignal()

    def __init__(self, label, hex_data, send_callback, parent=None):
        super().__init__(label, parent)
        self.hex_data = hex_data
        self.send_callback = send_callback
        self.setFont(QtGui.QFont('Segoe UI', 10))
        self.setStyleSheet('color: white; background-color: #006600')
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.clicked.connect(lambda: self.send_callback(self.hex_data))

    def _show_context_menu(self, pos):
        menu = QtWidgets.QMenu(self)
        edit_action = menu.addAction("Edit Macro...")
        action = menu.exec_(self.mapToGlobal(pos))
        if action == edit_action:
            self._edit_macro()

    def _edit_macro(self):
        dialog = MacroEditDialog(self.text(), self.hex_data, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.setText(dialog.label_edit.text())
            self.hex_data = dialog.hex_edit.text()
            self.macroChanged.emit()


class SerialSendView(QtWidgets.QWidget):

    serialSendSignal = QtCore.pyqtSignal(str)

    def __init__(self, parent):
        super().__init__(parent)

        self.history = []
        self.history_index = 0

        send_font = QtGui.QFont('Segoe UI', 10)

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
        self.sendButton.clicked.connect(self.sendButtonClicked)
        self.sendButton.setFont(send_font)
        self.sendButton.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        # Macro buttons (right-click to edit)
        macros = self._load_macros()
        self.macro_buttons = []
        for macro in macros:
            btn = MacroButton(macro["label"], macro["hex"], self.sendRaw, self)
            btn.macroChanged.connect(self._save_macros)
            self.macro_buttons.append(btn)

        self.setLayout(QtWidgets.QGridLayout())

        self.layout().addWidget(HLine(),       0, 0, 1, NUM_MACRO_BUTTONS)
        for i, btn in enumerate(self.macro_buttons):
            self.layout().addWidget(btn,       2, i, 1, 1)
        self.layout().addWidget(self.charMode, 1, 0, 1, 1)
        self.layout().addWidget(self.sendData,          1, 1, 1, 5)
        self.layout().addWidget(self.lineEnding,        1, 6, 1, 1)
        self.layout().addWidget(self.sendButton,        1, 7, 1, 1)
        self.layout().setContentsMargins(1, 1, 1, 1)

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
                self.serialSendSignal.emit(self.sendData.toPlainText())
                self._record_history(self.sendData.toPlainText())
                self.sendData.clear()
                self.history_index = 0
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

    def sendRaw(self, raw_hex_data):
        oldmode = self.charMode.currentIndex()
        oldending = self.lineEnding.currentIndex()
        self.charMode.setCurrentText("HEX")
        self.lineEnding.setCurrentText("No line ending")
        self.serialSendSignal.emit(raw_hex_data)
        self.charMode.setCurrentIndex(oldmode)
        self.lineEnding.setCurrentIndex(oldending)

    def _record_history(self, text):
        """Add a sent command to history, skipping blanks and repeats."""
        if text and (not self.history or self.history[-1] != text):
            self.history.append(text)
            if len(self.history) > 200:
                del self.history[:-200]

    def sendButtonClicked(self):
        self.serialSendSignal.emit(self.sendData.toPlainText())
        self._record_history(self.sendData.toPlainText())
        self.sendData.clear()
        self.history_index = 0

    def _get_macros_list(self):
        """Return current macro definitions as a list of dicts."""
        return [{"label": btn.text(), "hex": btn.hex_data} for btn in self.macro_buttons]

    def _load_macros_from_list(self, macros):
        """Restore macro buttons from a list of dicts."""
        if not isinstance(macros, list) or len(macros) != NUM_MACRO_BUTTONS:
            return
        for btn, macro in zip(self.macro_buttons, macros):
            btn.setText(macro.get("label", ""))
            btn.hex_data = macro.get("hex", "")

    def _load_macros(self):
        """Load macro definitions from settings file, or use defaults."""
        try:
            with open(SETTINGS_FILE, 'r') as f:
                s = json.load(f)
                macros = s.get('macros', None)
                if isinstance(macros, list) and len(macros) == NUM_MACRO_BUTTONS:
                    return macros
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            pass
        return [dict(m) for m in DEFAULT_MACROS]

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

        serial_label = QtWidgets.QLabel(' Serial Port: ')
        serial_label.setFont(QtGui.QFont('Segoe UI', 10))
        self.addWidget(serial_label)

        self.portOpenButton = QtWidgets.QPushButton('Open')
        self.portOpenButton.setCheckable(True)
        self.portOpenButton.setMinimumHeight(32)
        self.portOpenButton.setFont(toolbar_font)

        self.portScanButton = QtWidgets.QPushButton('Scan')
        self.portScanButton.setCheckable(False)
        self.portScanButton.clicked.connect(self.scan_button_Clicked)
        self.portScanButton.setMinimumHeight(32)
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

        self.addWidget(self.portOpenButton)
        self.addWidget(self.portNames)
        self.addWidget(self.portScanButton)
        self.addWidget(self.baudRates)
        self.addWidget(self.dataBits)
        self.addWidget(self._parity)
        self.addWidget(self.stopBits)
        self.addWidget(self._flowControl)

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
        self.portNames.setEnabled(flag)
        self.portScanButton.setEnabled(flag)
        self.baudRates.setEnabled(flag)
        self.dataBits.setEnabled(flag)
        self._parity.setEnabled(flag)
        self.stopBits.setEnabled(flag)
        self._flowControl.setEnabled(flag)

    def baudRate(self):
        return int(self.baudRates.currentText())

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


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QIcon(create_connector_pixmap('#22bb22')))
    window = SerialMonitor()
    if not window._geometry_restored:
        screen = app.primaryScreen().availableGeometry()
        window.resize(screen.width() * 8 // 15, screen.height() * 3 // 5)
    window.show()
    app.exec()
