# -*- coding: utf-8 -*-
import sys
import math
import re
import struct
import os
import json
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from PyQt5.QtSerialPort import QSerialPort, QSerialPortInfo
from PyQt5.QtGui import QPixmap, QTextCursor, QIcon, QPainter, QColor
from PyQt5.QtWidgets import *
import numpy as np

# --- Constants ---

DEFAULT_PLOT_LENGTH = 100

PLOT_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#e67e00',
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
MACROS_FILE = os.path.join(SCRIPT_DIR, 'macros.json')
NUM_MACRO_BUTTONS = 8

SETTINGS_FILE = os.path.join(SCRIPT_DIR, 'plot_settings.json')

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

        export_csv_action = file_menu.addAction('Export CSV...')
        export_csv_action.triggered.connect(self._menu_export_csv)

        export_png_action = file_menu.addAction('Export PNG...')
        export_png_action.triggered.connect(self._menu_export_png)

        file_menu.addSeparator()

        quit_action = file_menu.addAction('Quit')
        quit_action.setShortcut('Ctrl+Q')
        quit_action.triggered.connect(self.close)

        ### Tool Bar ###
        self.toolBar = ToolBar(self)
        self.addToolBar(self.toolBar)

        ### Status Bar ###
        self.setStatusBar(QtWidgets.QStatusBar(self))
        self.statusText = QtWidgets.QLabel(self)
        self.statusBar().addWidget(self.statusText)

        ### Signal Connect ###
        self.toolBar.portOpenButton.clicked.connect(self.portOpen)
        self.serialSendView.serialSendSignal.connect(self.sendFromPort)
        self.port.readyRead.connect(self.readFromPort)

    def portOpen(self, flag):
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
            self.serialDataView.handleReceivedData(raw_bytes)

    def sendFromPort(self, text):
        if self.serialSendView.charMode.currentText() == 'HEX':
            if self.serialSendView.lineEnding.currentIndex() == 1:
                text = text + '0A'
            elif self.serialSendView.lineEnding.currentIndex() == 2:
                text = text + '0D'
            elif self.serialSendView.lineEnding.currentIndex() == 3:
                text = text + '0D0A'
            try:
                self.port.write(bytes.fromhex(text))
                self.statusText.setText('')
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
                self.port.write(text.encode())
                self.statusText.setText('')
            except (UnicodeEncodeError, ValueError):
                self.statusText.setText('Not a valid ASCII string')

        elif self.serialSendView.charMode.currentText() == 'BINARY':
            try:
                value = int(text, 2)
                num_bytes = max(1, (value.bit_length() + 7) // 8)
                self.port.write(value.to_bytes(num_bytes, byteorder='big'))
                self.statusText.setText('')
            except (ValueError, OverflowError):
                self.statusText.setText('Not a valid BINARY string')

        self.serialDataView.appendSerialText(text, "send", self.serialSendView.charMode.currentText())

    def _menu_save_settings(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Settings', '', 'JSON Files (*.json);;All Files (*)')
        if path:
            self.serialDataView._save_settings(path)
            self.statusText.setText(f'Settings saved to {os.path.basename(path)}')

    def _menu_load_settings(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Load Settings', '', 'JSON Files (*.json);;All Files (*)')
        if path:
            self.serialDataView._load_settings(path)
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
            header = ','.join(dv._channel_name(i) for i in range(n_channels))
            lines = [header]
            for row in range(n_points):
                values = ','.join(str(dv.plot_data[ch][row]) for ch in range(n_channels))
                lines.append(values)
            with open(path, 'w') as f:
                f.write('\n'.join(lines) + '\n')
            self.statusText.setText(f'CSV exported to {os.path.basename(path)}')
        except OSError as e:
            self.statusText.setText(f'Export failed: {e}')

    def _menu_export_png(self):
        dv = self.serialDataView
        if dv.graphWidget is None:
            self.statusText.setText('No graph to export (enable Show Graph)')
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
        # Tracks whether the previous received chunk ended in '\r'. Used to
        # suppress a '\n' that begins the next chunk when CRLF is split across
        # serial reads (otherwise Qt renders a blank line between messages).
        self._pending_cr = False

        self._binary_reader = BinaryStreamReader()
        self._frame_reader = FrameReader()

        self.serialData = QtWidgets.QTextEdit(self)
        self.serialData.setReadOnly(True)
        self.serialData.setFontFamily('Segoe UI')
        self.serialData.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.serialDataHex = QtWidgets.QTextEdit(self)
        self.serialDataHex.setReadOnly(True)
        self.serialDataHex.setFontFamily('Segoe UI')
        self.serialDataHex.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.label_data_flow = QtWidgets.QLabel('Data: HEX')
        self.label_data_flow.setFont(QtGui.QFont('Segoe UI', 12))
        self.label_data_flow.setIndent(5)

        self.label_sent_data = QtWidgets.QLabel('Data: ASCII')
        self.label_sent_data.setFont(QtGui.QFont('Segoe UI', 12))
        self.label_sent_data.setIndent(5)

        self.graph_mode = QCheckBox("Show Graph")
        self.graph_mode.setFont(QtGui.QFont('Segoe UI', 12))
        self.graph_mode.stateChanged.connect(self.graph_state_changed)

        self.graph_channels = QSpinBox(minimum=1, maximum=12, value=4, prefix="Ch: ")
        self.graph_channels.setFont(QtGui.QFont('Segoe UI', 12))
        self.graph_channels.valueChanged.connect(self._on_channels_changed)

        self.data_mode = QtWidgets.QComboBox()
        self.data_mode.addItems(['ASCII', 'Binary Stream', 'Custom Frame'])
        self.data_mode.setFont(QtGui.QFont('Segoe UI', 12))
        self.data_mode.setMinimumWidth(130)
        self.data_mode.currentIndexChanged.connect(self._on_mode_changed)

        self.plot_length_spin = QSpinBox(minimum=10, maximum=10000, value=DEFAULT_PLOT_LENGTH, prefix="Pts: ", singleStep=50)
        self.plot_length_spin.setFont(QtGui.QFont('Segoe UI', 12))
        self.plot_length_spin.valueChanged.connect(self._on_setting_changed)

        self.clear_button = QtWidgets.QPushButton('Clear ALL')
        self.clear_button.clicked.connect(self.clear_button_Clicked)
        self.clear_button.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)

        self.label = QLabel(self)
        self.label.setPixmap(create_connector_pixmap('#cc2222'))

        self.converter_label = QtWidgets.QLabel('Converter')
        self.converter_label.setFont(QtGui.QFont('Segoe UI', 12))
        self.converter_label.setIndent(5)

        self.convert_A_type = QtWidgets.QComboBox(self)
        self.convert_A_type.addItems(list(CONVERTERS.keys()))
        self.convert_A_type.setCurrentIndex(0)
        self.convert_A_type.setMinimumHeight(30)
        self.convert_A_type.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.convert_A_type.currentIndexChanged.connect(self.translate_data)

        self.convert_A_text = QtWidgets.QTextEdit(self)
        self.convert_A_text.setMaximumHeight(31)
        self.convert_A_text.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.convert_A_text.textChanged.connect(self.translate_data)
        self.convert_A_text.setFont(QtGui.QFont('Segoe UI', 12))

        self.convert_B_text = QtWidgets.QTextEdit(self)
        self.convert_B_text.setMaximumHeight(31)
        self.convert_B_text.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.convert_B_text.setFont(QtGui.QFont('Segoe UI', 12))

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

        # Single controls row: left = decoding, right = Pts + Show Graph
        controls = QtWidgets.QWidget()
        cl = QtWidgets.QHBoxLayout(controls)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(4)
        row_widgets = [
            self.data_mode, self.type_combo, self.delimiter_combo,
            self.delimiter_custom, self.graph_channels, self.endian_combo,
            self.sync_button, self.sync_word_edit, self.size_field_combo,
            self.frame_size_spin, self.checksum_check, self.plot_length_spin,
            self.graph_mode, self._frame_start_label, self._payload_size_label,
        ]
        row_font = QtGui.QFont('Segoe UI', 12)
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
        # Right: plot controls
        cl.addStretch()
        cl.addWidget(self.plot_length_spin)
        cl.addWidget(self.graph_mode)

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

        self.setLayout(QtWidgets.QGridLayout(self))
        self.layout().addWidget(controls,               0, 0, 1, 6)
        self.layout().addWidget(self.splitter,          1, 0, 1, 6)
        self.layout().addWidget(self.converter_label,   2, 1, 1, 1)
        self.layout().addWidget(self.label,             3, 0, 1, 1)
        self.layout().addWidget(self.convert_A_type,    3, 1, 1, 1)
        self.layout().addWidget(self.convert_A_text,    3, 2, 1, 1)
        self.layout().addWidget(self.convert_B_text,    3, 3, 1, 2)
        self.layout().addWidget(self.clear_button,      3, 5, 1, 1, alignment=QtCore.Qt.AlignRight)
        self.layout().setRowStretch(1, 1)
        self.layout().setContentsMargins(2, 2, 2, 2)

        self._load_settings()

    def _channel_name(self, index):
        """Return custom name for a channel, or default 'Ch N'."""
        return self.channel_names.get(index, f'Ch {index}')

    def _channel_color(self, index):
        """Return custom color for a channel, or default from PLOT_COLORS."""
        return self.channel_colors.get(index, PLOT_COLORS[index % len(PLOT_COLORS)])

    def _create_plot_lines(self):
        """Create plot lines based on the current channel count spinbox."""
        n = self.graph_channels.value()
        plot_length = self.plot_length_spin.value()
        self.plot_lines = []
        self.plot_data = []
        for i in range(n):
            color = self._channel_color(i)
            line = self.graphWidget.plotItem.plot(
                pen=pg.mkPen(color, width=2),
                name=self._channel_name(i)
            )
            arr = np.zeros(plot_length)
            line.setData(arr)
            self.plot_lines.append(line)
            self.plot_data.append(arr)

    def _on_channels_changed(self):
        """Rebuild plot lines when channel count changes while graph is active."""
        if self.graphWidget is not None:
            for line in self.plot_lines:
                self.graphWidget.plotItem.removeItem(line)
            if self.graphWidget.plotItem.legend is not None:
                self.graphWidget.plotItem.legend.clear()
            self._create_plot_lines()
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
            self.graphWidget.plotItem.setMenuEnabled(False)
            self.graphWidget.plotItem.vb.setMenuEnabled(False)
            self.graphWidget.setXRange(0, self.plot_length_spin.value())
            self.graphWidget.enableAutoRange(axis='y')
            self.graphWidget.addLegend()
            self.graphWidget.scene().sigMouseClicked.connect(self._on_plot_mouse_clicked)
            self._create_plot_lines()
            self.numberbuffer = []
            # Clear Graph button overlaid in lower-right corner
            self._clear_graph_btn = QtWidgets.QPushButton('Clear Graph', self.graphWidget)
            self._clear_graph_btn.setStyleSheet(
                'background-color: #ffffff; border: 1px solid #aaa; padding: 2px 8px;')
            self._clear_graph_btn.clicked.connect(self._clear_graph)
            self._clear_graph_btn.adjustSize()
            self.graphWidget.installEventFilter(self)
            self.splitter.insertWidget(0, self.graphWidget)
            self._position_clear_graph_btn()
        else:
            self.graphWidget.removeEventFilter(self)
            self.graphWidget.setParent(None)
            self.graphWidget.deleteLater()
            self.graphWidget = None
            self._clear_graph_btn = None
            self.plot_lines = []
            self.plot_data = []

    def _clear_graph(self):
        """Reset all plot data to zeros."""
        for arr in self.plot_data:
            arr[:] = 0
        for line in self.plot_lines:
            line.setData(self.plot_data[self.plot_lines.index(line)])

    def _position_clear_graph_btn(self):
        """Position the Clear Graph button in the lower-right of the graph."""
        if self._clear_graph_btn and self.graphWidget:
            btn = self._clear_graph_btn
            gw = self.graphWidget
            btn.move(gw.width() - btn.width() - 5, gw.height() - btn.height() - 5)

    def eventFilter(self, obj, event):
        if obj is self.graphWidget and event.type() == QtCore.QEvent.Resize:
            self._position_clear_graph_btn()
        return super().eventFilter(obj, event)

    def _on_plot_mouse_clicked(self, ev):
        """Right-click on a legend entry to rename or change color."""
        if ev.button() != QtCore.Qt.RightButton:
            return
        legend = self.graphWidget.plotItem.legend
        if legend is None:
            return
        pos = ev.scenePos()
        for i, (sample, label) in enumerate(legend.items):
            row_rect = sample.sceneBoundingRect().united(label.sceneBoundingRect())
            if row_rect.contains(pos):
                self._show_channel_context_menu(i, label, sample)
                ev.accept()
                break

    def _show_channel_context_menu(self, channel_index, label, sample):
        """Show context menu for a legend entry."""
        menu = QtWidgets.QMenu(self)
        rename_action = menu.addAction('Rename...')
        color_action = menu.addAction('Change Color...')
        reset_action = menu.addAction('Reset to Default')

        action = menu.exec_(QtGui.QCursor.pos())
        if action == rename_action:
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
                self.plot_lines[channel_index].setPen(pg.mkPen(hex_color, width=2))
                sample.item = self.plot_lines[channel_index]
                sample.update()
                self._save_settings()
        elif action == reset_action:
            self.channel_names.pop(channel_index, None)
            self.channel_colors.pop(channel_index, None)
            default_name = f'Ch {channel_index}'
            default_color = PLOT_COLORS[channel_index % len(PLOT_COLORS)]
            label.setText(default_name)
            self.plot_lines[channel_index].setPen(pg.mkPen(default_color, width=2))
            sample.item = self.plot_lines[channel_index]
            sample.update()
            self._save_settings()

    def _append_data_point(self, value, channel):
        """Append a data point to a plot channel using in-place array shift."""
        if channel >= len(self.plot_data):
            return
        arr = self.plot_data[channel]
        arr[:-1] = arr[1:]
        arr[-1] = value
        self.plot_lines[channel].setData(arr)

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

    def clear_button_Clicked(self):
        self.serialDataHex.clear()
        self.serialData.clear()
        self.convert_A_text.clear()
        self.convert_B_text.clear()
        self._binary_reader.sync()
        self._frame_reader.reset()

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

        self._apply_reader_settings()
        self._save_settings()

    def _on_setting_changed(self):
        """Apply current settings to readers."""
        self._apply_reader_settings()
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
            'delimiter': self.delimiter_combo.currentText(),
            'delimiter_custom': self.delimiter_custom.text(),
            'channel_names': {str(k): v for k, v in self.channel_names.items()},
            'channel_colors': {str(k): v for k, v in self.channel_colors.items()},
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
        }

    def _save_settings(self, path=None):
        """Persist current plot/decode settings to JSON file."""
        try:
            with open(path or SETTINGS_FILE, 'w') as f:
                json.dump(self._get_settings_dict(), f, indent=2)
        except OSError:
            pass

    def _load_settings(self, path=None):
        """Restore plot/decode settings from JSON file."""
        try:
            with open(path or SETTINGS_FILE, 'r') as f:
                s = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            return

        widgets = [self.data_mode, self.type_combo, self.endian_combo,
                   self.graph_channels, self.plot_length_spin,
                   self.delimiter_combo, self.delimiter_custom,
                   self.sync_word_edit, self.size_field_combo,
                   self.frame_size_spin, self.checksum_check]
        for w in widgets:
            w.blockSignals(True)

        try:
            self.data_mode.setCurrentText(s.get('mode', 'ASCII'))
            self.graph_channels.setValue(s.get('num_channels', 4))
            self.plot_length_spin.setValue(s.get('num_points', DEFAULT_PLOT_LENGTH))
            self.delimiter_combo.setCurrentText(s.get('delimiter', 'Auto'))
            self.delimiter_custom.setText(s.get('delimiter_custom', ''))

            saved_names = s.get('channel_names', {})
            self.channel_names = {int(k): v for k, v in saved_names.items()}
            saved_colors = s.get('channel_colors', {})
            self.channel_colors = {int(k): v for k, v in saved_colors.items()}

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
        lastData = self.serialDataHex.toPlainText().split('\n')[-1]
        lastLength = math.ceil(len(lastData) / 3)

        pairs = [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]
        append_lists = []
        if lastLength > 0 and lastLength < 16:
            t = pairs[:16 - lastLength]
            pairs = pairs[16 - lastLength:]
            line = ' '.join(t)
            if pairs:
                line += '\n'
            append_lists.append(line)

        for i in range(0, len(pairs), 16):
            chunk = pairs[i:i+16]
            line = ' '.join(chunk)
            if i + 16 < len(pairs):
                line += '\n'
            append_lists.append(line)

        for text in append_lists:
            self.serialDataHex.insertPlainText(text)
        self.serialDataHex.moveCursor(QtGui.QTextCursor.End)

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

        lastData = self.serialDataHex.toPlainText().split('\n')[-1]
        lastLength = math.ceil(len(lastData) / 3)

        appendLists = []
        splitedByTwoChar = re.split('(..)', appendText.encode().hex())[1::2]
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
            elif mode == 'ASCII':
                self.serialData.insertPlainText(displayText)
                for insertText in appendLists:
                    self.serialDataHex.insertPlainText(insertText.upper())
            elif mode == 'BINARY':
                self.serialData.insertPlainText(displayText)
                try:
                    hex_val = format(int(appendText, 2), 'X')
                    if len(hex_val) % 2:
                        hex_val = '0' + hex_val
                    self.serialDataHex.insertPlainText(hex_val)
                except ValueError:
                    self.serialDataHex.insertPlainText(appendText)
        else:
            for insertText in appendLists:
                self.serialDataHex.insertPlainText(insertText.upper())
            self.serialData.insertPlainText(displayText)

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


# Horizontal separator line
class HLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(self.HLine | self.Sunken)


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
        self.setFont(QtGui.QFont('Segoe UI', 10, 60))
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

        self.charMode = QtWidgets.QComboBox(self)
        self.charMode.addItems(['ASCII', 'HEX', 'BINARY'])
        self.charMode.setCurrentIndex(0)
        self.charMode.setMinimumHeight(30)
        self.charMode.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        self.lineEnding = QtWidgets.QComboBox(self)
        self.lineEnding.addItems([
            "No line ending",
            "LF '\\n', 0x0A",
            "CR '\\r', 0x0D",
            "Both LF CR '\\r\\n'",
        ])
        self.lineEnding.setCurrentIndex(1)
        self.lineEnding.setMinimumHeight(30)
        self.lineEnding.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        self.sendData = QtWidgets.QTextEdit(self)
        self.sendData.installEventFilter(self)
        self.sendData.setAcceptRichText(False)
        self.sendData.setMaximumHeight(31)
        self.sendData.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.sendData.textChanged.connect(self._strip_newlines)
        self.sendData.setFont(QtGui.QFont('Segoe UI', 12))

        self.sendButton = QtWidgets.QPushButton('Send')
        self.sendButton.clicked.connect(self.sendButtonClicked)
        self.sendButton.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        # Macro buttons (right-click to edit)
        macros = self._load_macros()
        self.macro_buttons = []
        for macro in macros:
            btn = MacroButton(macro["label"], macro["hex"], self.sendRaw, self)
            btn.macroChanged.connect(self._save_macros)
            self.macro_buttons.append(btn)

        self.setLayout(QtWidgets.QGridLayout(self))

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

    def _load_macros(self):
        """Load macro definitions from JSON file, or use defaults."""
        try:
            with open(MACROS_FILE, 'r') as f:
                macros = json.load(f)
                if isinstance(macros, list) and len(macros) == NUM_MACRO_BUTTONS:
                    return macros
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            pass
        return [dict(m) for m in DEFAULT_MACROS]

    def _save_macros(self):
        """Persist current macro definitions to JSON file."""
        macros = [{"label": btn.text(), "hex": btn.hex_data} for btn in self.macro_buttons]
        try:
            with open(MACROS_FILE, 'w') as f:
                json.dump(macros, f, indent=2)
        except OSError:
            pass


class ToolBar(QtWidgets.QToolBar):
    def __init__(self, parent):
        super().__init__('Serial Port', parent)
        self.setMovable(False)

        toolbar_font = QtGui.QFont('Segoe UI', 12)

        serial_label = QtWidgets.QLabel(' Serial Port: ')
        serial_label.setFont(QtGui.QFont('Segoe UI', 12))
        self.addWidget(serial_label)

        self.portOpenButton = QtWidgets.QPushButton('Open')
        self.portOpenButton.setCheckable(True)
        self.portOpenButton.setMinimumHeight(32)
        self.portOpenButton.setFont(toolbar_font)

        self.portScanButton = QtWidgets.QPushButton('Scan')
        self.portScanButton.setCheckable(True)
        self.portScanButton.clicked.connect(self.scan_button_Clicked)
        self.portScanButton.setMinimumHeight(32)
        self.portScanButton.setFont(toolbar_font)

        self.portNames = QtWidgets.QComboBox(self)
        self.portNames.addItems([port.portName() for port in QSerialPortInfo().availablePorts()])
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

    def scan_button_Clicked(self):
        self.portNames.clear()
        self.portNames.addItems([port.portName() for port in QSerialPortInfo().availablePorts()])

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
        return self.portNames.currentText()

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
    window.resize(screen.width() * 2 // 3, screen.height() * 3 // 4)
    window.show()
    app.exec()
