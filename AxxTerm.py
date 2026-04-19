# -*- coding: utf-8 -*-
import sys
import math
import re
import struct
import os
import json
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
    'HEX --> DECIMAL': lambda v: str(int(v, 16)),
    'HEX --> BINARY': lambda v: bin(int(v, 16))[2:].zfill(8),
    'ASCII --> HEX': lambda v: '0x' + v.encode('ISO-8859-1').hex(),
    'ASCII --> DECIMAL': lambda v: ' '.join(str(b) for b in v.encode('ISO-8859-1')),
    'ASCII --> BINARY': lambda v: bin(int.from_bytes(v.encode('ISO-8859-1'), 'big')),
    'DECIMAL --> HEX': lambda v: hex(int(v)),
    'DECIMAL --> ASCII': lambda v: chr(int(v)),
    'DECIMAL --> BINARY': lambda v: format(int(v), '08b'),
    'BINARY --> HEX': lambda v: hex(int(v, 2)),
    'BINARY --> ASCII': lambda v: chr(int(v, 2)),
    'BINARY --> DECIMAL': lambda v: str(int(v, 2)),
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
        results = []
        while len(self.buffer) >= package_size:
            values = struct.unpack(fmt, self.buffer[:package_size])
            self.buffer = self.buffer[package_size:]
            results.append(values)
        return results

    def sync(self):
        """Clear buffer to re-align stream."""
        self.buffer.clear()


class FrameReader:
    """Decodes framed binary packets with sync word, optional size, and optional checksum.

    Note: Sync word matching uses a simple byte-by-byte scan. Sync words with
    internal prefix repetition (e.g. 01 02 01 03) may miss valid frames if the
    pattern overlaps in the stream. Simple sync words (AA, AA BB, etc.) work correctly.
    """

    SEARCHING = 0
    READING_SIZE = 1
    READING_PAYLOAD = 2

    def __init__(self):
        self.data_type = 'float32'
        self.endianness = 'little'
        self.num_channels = 4
        self.sync_word = bytes([0xAA])
        self.size_field = 'fixed'
        self.frame_size = 12
        self.checksum_enabled = False
        self.state = self.SEARCHING
        self._sync_index = 0
        self._size_buffer = bytearray()
        self._payload_buffer = bytearray()
        self._payload_size = 0

    def reset(self):
        """Reset state machine and clear buffers."""
        self.state = self.SEARCHING
        self._sync_index = 0
        self._size_buffer.clear()
        self._payload_buffer.clear()
        self._payload_size = 0

    def feed(self, data):
        """Feed raw bytes. Returns list of tuples, one per sample."""
        results = []
        fmt_char, type_size = DATA_TYPES[self.data_type]
        prefix = '<' if self.endianness == 'little' else '>'
        sample_size = self.num_channels * type_size
        if sample_size == 0:
            return []
        fmt = prefix + fmt_char * self.num_channels

        for byte in data:
            if self.state == self.SEARCHING:
                if byte == self.sync_word[self._sync_index]:
                    self._sync_index += 1
                    if self._sync_index == len(self.sync_word):
                        self._sync_index = 0
                        if self.size_field == 'fixed':
                            self._payload_size = self.frame_size
                            self.state = self.READING_PAYLOAD
                            self._payload_buffer.clear()
                        else:
                            self.state = self.READING_SIZE
                            self._size_buffer.clear()
                else:
                    self._sync_index = 1 if byte == self.sync_word[0] else 0

            elif self.state == self.READING_SIZE:
                self._size_buffer.append(byte)
                needed = 1 if self.size_field == '1-byte' else 2
                if len(self._size_buffer) >= needed:
                    if needed == 1:
                        self._payload_size = self._size_buffer[0]
                    else:
                        size_fmt = '<H' if self.endianness == 'little' else '>H'
                        self._payload_size = struct.unpack(size_fmt, self._size_buffer[:2])[0]
                    if self._payload_size == 0 or (self._payload_size % sample_size) != 0:
                        self.state = self.SEARCHING
                        self._sync_index = 0
                    else:
                        self.state = self.READING_PAYLOAD
                        self._payload_buffer.clear()

            elif self.state == self.READING_PAYLOAD:
                self._payload_buffer.append(byte)
                total_needed = self._payload_size + (1 if self.checksum_enabled else 0)
                if len(self._payload_buffer) >= total_needed:
                    payload = bytes(self._payload_buffer[:self._payload_size])
                    valid = True
                    if self.checksum_enabled:
                        received = self._payload_buffer[self._payload_size]
                        computed = sum(payload) & 0xFF
                        if received != computed:
                            valid = False
                    if valid:
                        offset = 0
                        while offset + sample_size <= len(payload):
                            values = struct.unpack_from(fmt, payload, offset)
                            results.append(values)
                            offset += sample_size
                    self.state = self.SEARCHING
                    self._sync_index = 0

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

        # Save when serial port settings change
        self.toolBar.baudRates.currentIndexChanged.connect(lambda: self.save_all_settings())
        self.toolBar.dataBits.currentIndexChanged.connect(lambda: self.save_all_settings())
        self.toolBar._parity.currentIndexChanged.connect(lambda: self.save_all_settings())
        self.toolBar.stopBits.currentIndexChanged.connect(lambda: self.save_all_settings())
        self.toolBar._flowControl.currentIndexChanged.connect(lambda: self.save_all_settings())

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
                self.toolBar.serialControlEnable(False)
                self.serialDataView.label.setPixmap(create_connector_pixmap('#22bb22'))
        else:
            self.port.close()
            self.statusText.setText('Port closed')
            self.toolBar.serialControlEnable(True)
            self.serialDataView.label.setPixmap(create_connector_pixmap('#cc2222'))

    def readFromPort(self):
        data = self.port.readAll()
        if len(data) > 0:
            raw_bytes = bytes(data.data())
            self._rx_bytes += len(raw_bytes)
            self._rx_total += len(raw_bytes)
            # Log immediately for accurate timestamps
            mode = self.serialDataView.data_mode.currentText()
            if mode == 'ASCII':
                self._log_data('RX', raw_bytes.decode('ISO-8859-1'))
            else:
                hex_str = raw_bytes.hex().upper()
                self._log_data('RX', f'[HEX] {hex_str}')
            # Buffer for throttled display update
            self._rx_buffer.extend(raw_bytes)

    def _flush_display(self):
        """Process buffered RX data and update the display (~30 fps)."""
        if not self._rx_buffer:
            return
        data = bytes(self._rx_buffer)
        self._rx_buffer.clear()
        self.serialDataView.handleReceivedData(data)

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
                tx = text.encode()
                self.port.write(tx)
                self._tx_bytes += len(tx)
                self._tx_total += len(tx)
                self.statusText.setText('')
                sent = True
            except (UnicodeEncodeError, ValueError):
                self.statusText.setText('Not a valid ASCII string')

        elif self.serialSendView.charMode.currentText() == 'BINARY':
            try:
                value = int(text, 2)
                num_bytes = max(1, (value.bit_length() + 7) // 8)
                tx = value.to_bytes(num_bytes, byteorder='big')
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
            self.statusText.setText('Port reconnected')

    # --- Recording / Logging ---------------------------------------------------

    def _log_data(self, direction, text):
        """Write a timestamped line to the log file if recording is active."""
        if not self._recording or self._log_file is None:
            return
        now = datetime.now()
        ts = now.strftime('%Y-%m-%d %H:%M:%S.') + f'{now.microsecond // 1000:03d}'
        for line in text.splitlines():
            if line:
                self._log_file.write(f'[{ts}] {direction}: {line}\n')
        self._log_file.flush()

    def _toggle_recording(self):
        """Start or stop recording serial data to a log file."""
        if self._recording:
            # Stop recording
            if self._log_file is not None:
                self._log_file.close()
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
        """Ensure the log file is closed when the application exits."""
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        self._recording = False
        super().closeEvent(event)

    def save_all_settings(self, path=None):
        """Save all settings (plot, serial port, macros) to one JSON file."""
        settings = {
            'dark_mode': self._dark_mode,
            'auto_reconnect': self._auto_reconnect,
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
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            return

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
            n_channels = len(dv.plot_data)
            n_points = len(dv.plot_data[0])
            # Build header: regular channels + math channels
            headers = [dv._channel_name(i) for i in range(n_channels)]
            for mch in dv._math_channels:
                headers.append(mch.get('name', 'Math'))
            header = ','.join(headers)
            lines = [header]
            for row in range(n_points):
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

        self.numberbuffer = []
        self.plot_lines = []
        self.plot_data = []
        self.graphWidget = None
        self.channel_names = {}   # {index: 'custom name'} for renamed channels
        self.channel_colors = {}  # {index: '#hex'} for custom channel colors
        self.channel_axes = {}    # {index: 1 or 2} axis assignment (1=left, 2=right)
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
        self._math_update_counter = 0

        self._binary_reader = BinaryStreamReader()
        self._frame_reader = FrameReader()

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
        self.serialData.setFontFamily('Segoe UI')
        self.serialData.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.serialDataHex = QtWidgets.QTextEdit(self)
        self.serialDataHex.setReadOnly(True)
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

    def _channel_color(self, index):
        """Return custom color for a channel, or default from PLOT_COLORS."""
        return self.channel_colors.get(index, PLOT_COLORS[index % len(PLOT_COLORS)])

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
        for i in range(n):
            color = self._channel_color(i)
            arr = np.zeros(plot_length)
            axis = self.channel_axes.get(i, 1)
            if axis == 2 and self._y2_viewbox is not None:
                # Dashed pen for Y2 channels
                pen = pg.mkPen(color, width=2, style=QtCore.Qt.DashLine)
                line = pg.PlotDataItem(pen=pen, name=self._channel_name(i))
                self._y2_viewbox.addItem(line)
                self._y2_plot_lines[i] = line
            else:
                pen = pg.mkPen(color, width=2)
                line = pg.PlotDataItem(pen=pen, name=self._channel_name(i))
                self.graphWidget.plotItem.addItem(line)
            # Add all channels to legend in order
            if self.graphWidget.plotItem.legend is not None:
                self.graphWidget.plotItem.legend.addItem(line, self._channel_name(i))
            line.setData(arr)
            self.plot_lines.append(line)
            self.plot_data.append(arr)
        # Rebuild math channel lines too
        self._rebuild_math_lines()

    def _on_channels_changed(self):
        """Rebuild plot lines when channel count changes while graph is active."""
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
            self.numberbuffer = []
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
            self.splitter.insertWidget(0, self.graphWidget)
            self._position_overlay_buttons()
            self._update_graph_theme()
        else:
            # Destroy FFT widget first if it exists
            self._destroy_fft_widget()
            # Clean up Y2 axis before destroying graph
            self._y2_plot_lines = {}
            self._y2_viewbox = None
            self.graphWidget.removeEventFilter(self)
            self.graphWidget.setParent(None)
            self.graphWidget.deleteLater()
            self.graphWidget = None
            self._clear_graph_btn = None
            self._pause_btn = None
            self._crosshair = None
            self._cursor_label = None
            self.plot_lines = []
            self.plot_data = []
            self._math_lines = []
            self._math_data = []
            self._math_errors = set()
            self._plot_paused = False

    def _clear_graph(self):
        """Reset all plot data to zeros."""
        for i, (arr, line) in enumerate(zip(self.plot_data, self.plot_lines)):
            arr[:] = 0
            line.setData(arr)
        for arr, line in zip(self._math_data, self._math_lines):
            arr[:] = 0
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
            name = self._channel_name(i)
            color = self._channel_color(i)
            val = arr[x_idx]
            parts.append(f'<span style="color:{color}"><b>{name}</b>: {val:.4f}</span>')
        for i, (mch, arr) in enumerate(zip(self._math_channels, self._math_data)):
            color = self._math_channel_color(i)
            name = mch.get('name', f'Math {i}')
            if x_idx < len(arr):
                val = arr[x_idx]
                parts.append(f'<span style="color:{color}"><b>{name}</b>: {val:.4f}</span>')
        html = '<br>'.join(parts)
        self._cursor_label.setHtml(f'<div style="background:rgba(255,255,255,200);padding:2px">{html}</div>')
        self._cursor_label.setPos(x, mouse_point.y())
        self._cursor_label.setVisible(True)

    def _on_range_changed(self):
        """Track Y-axis range changes and save."""
        if self.graphWidget is None:
            return
        vb = self.graphWidget.plotItem.vb
        auto = vb.autoRangeEnabled()[1]  # [x_auto, y_auto]
        self._y_auto_scale = bool(auto)
        if not auto:
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
        # Y-axis toggle
        current_axis = self.channel_axes.get(channel_index, 1)
        if current_axis == 1:
            axis_action = menu.addAction('Move to Y2 axis')
        else:
            axis_action = menu.addAction('Move to Y1 axis')
        reset_action = menu.addAction('Reset to Default')

        action = menu.exec_(QtGui.QCursor.pos())
        if action == axis_action:
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
                label.setText(new_name.strip())
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
                self._save_settings()
        elif action == reset_action:
            was_y2 = self.channel_axes.get(channel_index, 1) == 2
            self.channel_names.pop(channel_index, None)
            self.channel_colors.pop(channel_index, None)
            self.channel_axes.pop(channel_index, None)
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
        for i, arr in enumerate(self.plot_data):
            if i >= len(self._fft_lines):
                break
            fft_mag = np.abs(np.fft.rfft(arr))
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

    def _eval_math_expression(self, expression):
        """Evaluate a math expression safely and return the result array."""
        namespace = {'__builtins__': {}, 'np': np, 'numpy': np}
        for i, arr in enumerate(self.plot_data):
            namespace[f'ch{i}'] = arr
        try:
            result = eval(expression, {"__builtins__": {}}, namespace)
            # Ensure result is a numpy array of the right length
            if isinstance(result, (int, float)):
                result = np.full(len(self.plot_data[0]), result)
            result = np.asarray(result, dtype=float)
            if result.shape != self.plot_data[0].shape:
                return None
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

    def _append_data_point(self, value, channel):
        """Append a data point to a plot channel using in-place array shift."""
        if channel >= len(self.plot_data):
            return

        # When paused, don't update arrays at all -- plot freezes
        if self._plot_paused:
            return

        arr = self.plot_data[channel]

        # Trigger detection (only check on the trigger channel)
        if (self._trigger_armed and self._trigger_countdown < 0
                and channel == self._trigger_channel):
            prev = self._trigger_prev_value
            self._trigger_prev_value = value
            if prev is not None:
                fired = False
                if self._trigger_edge == 'rising':
                    fired = prev < self._trigger_level and value >= self._trigger_level
                else:
                    fired = prev > self._trigger_level and value <= self._trigger_level
                if fired:
                    self._trigger_countdown = len(arr) // 2

        # Decrement trigger countdown (per-sample on channel 0 to count once per sample set)
        if self._trigger_countdown > 0 and channel == 0:
            self._trigger_countdown -= 1
            if self._trigger_countdown <= 0:
                self._plot_paused = True
                self._trigger_armed = False
                self._trigger_countdown = -1
                self._update_pause_btn_style()
                self._position_overlay_buttons()

        arr[:-1] = arr[1:]
        arr[-1] = value
        self.plot_lines[channel].setData(arr)

        # Update FFT and math channels every 10 samples (count on channel 0)
        if channel == 0:
            if self._fft_widget is not None:
                self._fft_update_counter += 1
                if self._fft_update_counter >= 10:
                    self._fft_update_counter = 0
                    self._update_fft()
            if self._math_lines:
                self._math_update_counter += 1
                if self._math_update_counter >= 10:
                    self._math_update_counter = 0
                    self._update_math_channels()

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
                continue
            if ':' in field:
                field = field.split(':', 1)[1].strip()
            try:
                values.append(float(field))
            except ValueError:
                continue
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
        # Find and highlight all matches
        highlight_fmt = QtGui.QTextCharFormat()
        highlight_fmt.setBackground(QColor('#FFFF00'))
        highlight_fmt.setForeground(QColor('#000000'))
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

    def clear_button_Clicked(self):
        self.serialDataHex.clear()
        self.serialData.clear()
        self.convert_A_text.clear()
        self.convert_B_text.clear()
        self._binary_reader.sync()
        self._frame_reader.reset()
        self._pending_cr = False
        self._hex_col = 0
        self.numberbuffer = []

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
        self._binary_reader.sync()
        self._frame_reader.reset()
        self._pending_cr = False
        self._hex_col = 0
        self.numberbuffer = []

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

        self._frame_reader.reset()

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
        """Persist settings via parent SerialMonitor (saves everything to one file)."""
        monitor = self.window()
        if hasattr(monitor, 'save_all_settings'):
            monitor.save_all_settings(path)

    def _load_plot_settings(self, s):
        """Restore plot/decode settings from a dict (subsection of full settings)."""
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
            self.graph_mode.setChecked(s.get('show_plot', False))
            self._fft_check.setChecked(s.get('show_fft', False))
            self.delimiter_combo.setCurrentText(s.get('delimiter', 'Auto'))
            self.delimiter_custom.setText(s.get('delimiter_custom', ''))

            saved_names = s.get('channel_names', {})
            self.channel_names = {int(k): v for k, v in saved_names.items()}
            saved_colors = s.get('channel_colors', {})
            self.channel_colors = {int(k): v for k, v in saved_colors.items()}
            saved_axes = s.get('channel_axes', {})
            self.channel_axes = {int(k): v for k, v in saved_axes.items()}

            self._y_auto_scale = s.get('y_auto_scale', True)
            self._y_min = s.get('y_min', -1.0)
            self._y_max = s.get('y_max', 1.0)

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

            # Math channels
            saved_math = s.get('math_channels', [])
            if isinstance(saved_math, list):
                self._math_channels = [
                    {'name': m.get('name', ''), 'expression': m.get('expression', '')}
                    for m in saved_math
                    if isinstance(m, dict) and m.get('expression', '').strip()
                ]
        except (KeyError, TypeError):
            pass

        for w in widgets:
            w.blockSignals(False)

        self._apply_reader_settings()
        self.frame_size_spin.setEnabled(self.size_field_combo.currentText().startswith('Fixed'))
        self.delimiter_custom.setVisible(self.delimiter_combo.currentText() == 'Other')
        self._on_mode_changed()

    def handleReceivedData(self, raw_bytes):
        """Route incoming serial bytes based on current data mode."""
        mode = self.data_mode.currentText()

        if mode == 'ASCII':
            text = raw_bytes.decode('ISO-8859-1')
            self.appendSerialText(text, "read")
            return

        # Binary/Frame modes: always show raw HEX
        self._append_hex_view(raw_bytes)

        # Decode through appropriate reader
        if mode == 'Binary Stream':
            samples = self._binary_reader.feed(raw_bytes)
        else:
            samples = self._frame_reader.feed(raw_bytes)

        # Display decoded values and feed plot
        for sample in samples:
            self._append_decoded_line(sample)
            if self.graph_mode.isChecked() and self.graphWidget is not None:
                for i, val in enumerate(sample):
                    self._append_data_point(float(val), i)

    def _append_hex_view(self, raw_bytes):
        """Append raw bytes to the HEX text view (right panel)."""
        self.serialDataHex.moveCursor(QtGui.QTextCursor.End)
        self.serialDataHex.setFontFamily('Segoe UI')
        self.serialDataHex.setTextColor(QtGui.QColor(255, 0, 0))

        hex_str = raw_bytes.hex().upper()
        lastLength = self._hex_col

        pairs = [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]
        append_lists = []
        if lastLength > 0 and lastLength < 16:
            t = pairs[:16 - lastLength]
            pairs = pairs[16 - lastLength:]
            line = ' '.join(t)
            if pairs:
                line += '\n'
                self._hex_col = 0
            else:
                self._hex_col = lastLength + len(t)
            append_lists.append(line)

        for i in range(0, len(pairs), 16):
            chunk = pairs[i:i+16]
            line = ' '.join(chunk)
            if i + 16 < len(pairs):
                line += '\n'
                self._hex_col = 0
            else:
                self._hex_col = len(chunk)
            append_lists.append(line)

        for text in append_lists:
            self.serialDataHex.insertPlainText(text)
        self.serialDataHex.moveCursor(QtGui.QTextCursor.End)
        self._trim_text_buffer(self.serialDataHex)

    def _trim_text_buffer(self, text_edit, max_lines=MAX_TEXT_LINES):
        """Remove oldest lines if text exceeds max_lines."""
        doc = text_edit.document()
        if doc.blockCount() > max_lines:
            cursor = QTextCursor(doc.begin())
            excess = doc.blockCount() - max_lines
            for _ in range(excess):
                cursor.movePosition(QTextCursor.NextBlock, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            cursor.deleteChar()  # remove the leftover newline

    def _append_decoded_line(self, sample):
        """Append one decoded sample to the left panel with color-coded channels."""
        self.serialData.moveCursor(QtGui.QTextCursor.End)
        self.serialData.setFontFamily('Segoe UI')
        for i, val in enumerate(sample):
            color = QColor(self._channel_color(i))
            self.serialData.setTextColor(color)
            name = self._channel_name(i)
            if isinstance(val, float):
                text = f'{name}: {val:.4f}'
            else:
                text = f'{name}: {val}'
            if i < len(sample) - 1:
                text += '  '
            self.serialData.insertPlainText(text)
        self.serialData.setTextColor(QtGui.QColor(255, 0, 0))
        self.serialData.insertPlainText('\n')
        self.serialData.moveCursor(QtGui.QTextCursor.End)
        self._trim_text_buffer(self.serialData)

    def appendSerialText(self, appendText, direction, mode="ASCII"):
        if direction == "send":
            self.textcolor = QtGui.QColor(0, 0, 255)
        else:
            self.textcolor = QtGui.QColor(255, 0, 0)
        self.serialData.moveCursor(QtGui.QTextCursor.End)
        self.serialData.setFontFamily('Segoe UI')
        self.serialData.setTextColor(self.textcolor)
        self.serialDataHex.moveCursor(QtGui.QTextCursor.End)
        self.serialDataHex.setFontFamily('Segoe UI')
        self.serialDataHex.setTextColor(self.textcolor)

        # QTextEdit treats BOTH '\r' and '\n' as paragraph separators, so a
        # CRLF stream renders with a blank line between each message. Keep
        # the raw bytes for the HEX view, but normalize for the ASCII view.
        # Serial data arrives in arbitrary-size chunks; a '\r\n' pair can be
        # split across two reads, so we also carry a _pending_cr flag that
        # swallows a leading '\n' when the previous chunk ended in '\r'.
        displayText = appendText.replace('\r\n', '\n').replace('\r', '\n')
        if direction == "send":
            self._pending_cr = False
        else:
            if self._pending_cr and displayText.startswith('\n'):
                displayText = displayText[1:]
            self._pending_cr = appendText.endswith('\r')

        lastLength = self._hex_col

        appendLists = []
        splitedByTwoChar = re.split('(..)', appendText.encode().hex())[1::2]
        num_new_pairs = len(splitedByTwoChar)
        if lastLength > 0:
            t = splitedByTwoChar[: 16 - lastLength] + ['\n']
            appendLists.append(' '.join(t))
            splitedByTwoChar = splitedByTwoChar[16 - lastLength:]

        appendLists += [' '.join(splitedByTwoChar[i * 16: (i + 1) * 16] + ['\n'])
                        for i in range(math.ceil(len(splitedByTwoChar) / 16))]

        if appendLists and len(appendLists[-1]) < 47:
            appendLists[-1] = appendLists[-1][:-1]

        if direction == "send":
            if mode == 'HEX':
                try:
                    decoded = bytes.fromhex(appendText).decode('ISO-8859-1')
                    decoded = decoded.replace('\r\n', '\n').replace('\r', '\n')
                    self.serialData.insertPlainText(decoded)
                except ValueError:
                    self.serialData.insertPlainText(displayText)
                self.serialDataHex.insertPlainText(appendText.upper())
                # HEX send inserts raw hex chars without line formatting
                sent_pairs = len(appendText) // 2
                self._hex_col = (lastLength + sent_pairs) % 16
            elif mode == 'ASCII':
                self.serialData.insertPlainText(displayText)
                for insertText in appendLists:
                    self.serialDataHex.insertPlainText(insertText.upper())
                # Update _hex_col from appendLists (line-wrapped hex pairs)
                self._hex_col = (lastLength + num_new_pairs) % 16
            elif mode == 'BINARY':
                self.serialData.insertPlainText(displayText)
                try:
                    hex_val = format(int(appendText, 2), 'X')
                    if len(hex_val) % 2:
                        hex_val = '0' + hex_val
                    self.serialDataHex.insertPlainText(hex_val)
                    sent_pairs = len(hex_val) // 2
                    self._hex_col = (lastLength + sent_pairs) % 16
                except ValueError:
                    self.serialDataHex.insertPlainText(appendText)
        else:
            for insertText in appendLists:
                self.serialDataHex.insertPlainText(insertText.upper())
            self.serialData.insertPlainText(displayText)
            # Update _hex_col from appendLists (line-wrapped hex pairs)
            self._hex_col = (lastLength + num_new_pairs) % 16

            if self.graph_mode.isChecked() and self.graphWidget is not None:
                for char in displayText:
                    if char == '\n':
                        values = self._parse_plot_values(''.join(self.numberbuffer))
                        for i, val in enumerate(values):
                            self._append_data_point(val, i)
                        self.numberbuffer = []
                    else:
                        self.numberbuffer.append(char)

        self.serialData.moveCursor(QtGui.QTextCursor.End)
        self.serialDataHex.moveCursor(QtGui.QTextCursor.End)
        self._trim_text_buffer(self.serialData)
        self._trim_text_buffer(self.serialDataHex)


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
                self.history.append(self.sendData.toPlainText())
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

    def sendButtonClicked(self):
        self.serialSendSignal.emit(self.sendData.toPlainText())
        self.history.append(self.sendData.toPlainText())
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
        """Persist macros via parent SerialMonitor."""
        monitor = self.window()
        if hasattr(monitor, 'save_all_settings'):
            monitor.save_all_settings()


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
    screen = app.primaryScreen().availableGeometry()
    window.resize(screen.width() * 8 // 15, screen.height() * 3 // 5)
    window.show()
    app.exec()
