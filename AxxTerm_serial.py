# -*- coding: utf-8 -*-
import sys
import math
import re
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
    '#e6194b', '#3cb44b', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#9A6324',
    '#800000', '#469990', '#dcbeff', '#000075',
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
            self.serialDataView.appendSerialText(QtCore.QTextStream(data).readAll(), "read")

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


class SerialDataView(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.numberbuffer = []
        self.plot_lines = []
        self.plot_data = []
        self.graphWidget = None

        self.serialData = QtWidgets.QTextEdit(self)
        self.serialData.setReadOnly(True)
        self.serialData.setFontFamily('Segoe UI')
        self.serialData.setMinimumWidth(500)
        self.serialData.setMinimumHeight(300)
        self.serialData.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.serialDataHex = QtWidgets.QTextEdit(self)
        self.serialDataHex.setReadOnly(True)
        self.serialDataHex.setFontFamily('Segoe UI')
        self.serialDataHex.setMinimumWidth(500)
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

        self.plot_length_spin = QSpinBox(minimum=10, maximum=10000, value=DEFAULT_PLOT_LENGTH, prefix="Pts: ", singleStep=50)
        self.plot_length_spin.setFont(QtGui.QFont('Segoe UI', 12))

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

        # Graph controls container
        graph_controls = QtWidgets.QWidget()
        gc_layout = QtWidgets.QHBoxLayout(graph_controls)
        gc_layout.setContentsMargins(0, 0, 0, 0)
        gc_layout.addWidget(self.plot_length_spin)
        gc_layout.addWidget(self.graph_channels)
        gc_layout.addWidget(self.graph_mode)

        self.setLayout(QtWidgets.QGridLayout(self))
        self.layout().addWidget(self.label_data_flow,   1, 3, 1, 2)
        self.layout().addWidget(self.label_sent_data,   1, 0, 1, 3)
        self.layout().addWidget(graph_controls,         1, 5, 1, 1, alignment=QtCore.Qt.AlignRight)
        self.layout().addWidget(self.serialData,        2, 0, 1, 3)
        self.layout().addWidget(self.serialDataHex,     2, 3, 1, 3)
        self.layout().addWidget(self.label,             4, 0, 1, 1)
        self.layout().addWidget(self.converter_label,   3, 1, 1, 1)
        self.layout().addWidget(self.convert_A_type,    4, 1, 1, 1)
        self.layout().addWidget(self.convert_A_text,    4, 2, 1, 1)
        self.layout().addWidget(self.convert_B_text,    4, 3, 1, 2)
        self.layout().addWidget(self.clear_button,      4, 5, 1, 1, alignment=QtCore.Qt.AlignRight)
        self.layout().setContentsMargins(2, 2, 2, 2)

    def _create_plot_lines(self):
        """Create plot lines based on the current channel count spinbox."""
        n = self.graph_channels.value()
        plot_length = self.plot_length_spin.value()
        self.plot_lines = []
        self.plot_data = []
        for i in range(n):
            color = PLOT_COLORS[i % len(PLOT_COLORS)]
            line = self.graphWidget.plotItem.plot(
                pen=pg.mkPen(color, width=2),
                name=f'Ch {i}'
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

    def graph_state_changed(self):
        if self.graph_mode.isChecked():
            self.graphWidget = pg.PlotWidget(title="Plot")
            self.graphWidget.setBackground('#FFFFFFFF')
            self.graphWidget.setMinimumHeight(300)
            self.graphWidget.plotItem.getAxis('bottom').setPen(pg.mkPen(color='#000000'))
            self.graphWidget.plotItem.getAxis('left').setPen(pg.mkPen(color='#000000'))
            self.graphWidget.plotItem.showGrid(True, True, 0.3)
            self.graphWidget.setXRange(0, self.plot_length_spin.value())
            self.graphWidget.enableAutoRange(axis='y')
            self.graphWidget.addLegend()
            self._create_plot_lines()
            self.numberbuffer = []
            self.layout().addWidget(self.graphWidget, 0, 0, 1, 6)
        else:
            self.layout().removeWidget(self.graphWidget)
            self.graphWidget.deleteLater()
            self.graphWidget = None
            self.plot_lines = []
            self.plot_data = []

    def _append_data_point(self, value, channel):
        """Append a data point to a plot channel using in-place array shift."""
        if channel >= len(self.plot_data):
            return
        arr = self.plot_data[channel]
        arr[:-1] = arr[1:]
        arr[-1] = value
        self.plot_lines[channel].setData(arr)

    def _parse_plot_values(self, line):
        """Parse a line of serial data into numeric values.

        Supports:
          - Tab-separated:   1.0\\t2.0\\t3.0
          - Comma-separated: 1.0,2.0,3.0
          - Space-separated: 1.0 2.0 3.0
          - Labeled:         ch0:1.0\\tch1:2.0\\tch2:3.0
        """
        line = line.strip()
        if not line:
            return []
        if '\t' in line:
            fields = line.split('\t')
        elif ',' in line:
            fields = line.split(',')
        else:
            fields = line.split()
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
                    self.serialData.insertPlainText(bytes.fromhex(appendText).decode('ISO-8859-1'))
                except ValueError:
                    self.serialData.insertPlainText(appendText)
                self.serialDataHex.insertPlainText(appendText.upper())
            elif mode == 'ASCII':
                self.serialData.insertPlainText(appendText)
                for insertText in appendLists:
                    self.serialDataHex.insertPlainText(insertText.upper())
            elif mode == 'BINARY':
                self.serialData.insertPlainText(appendText)
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
            self.serialData.insertPlainText(appendText)

            if self.graph_mode.isChecked() and self.graphWidget is not None:
                for char in appendText:
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
        super().__init__(parent)

        self.portOpenButton = QtWidgets.QPushButton('Open')
        self.portOpenButton.setCheckable(True)
        self.portOpenButton.setMinimumHeight(32)

        self.portScanButton = QtWidgets.QPushButton('Scan')
        self.portScanButton.setCheckable(True)
        self.portScanButton.clicked.connect(self.scan_button_Clicked)
        self.portScanButton.setMinimumHeight(32)

        self.portNames = QtWidgets.QComboBox(self)
        self.portNames.addItems([port.portName() for port in QSerialPortInfo().availablePorts()])
        self.portNames.setMinimumHeight(30)

        self.baudRates = QtWidgets.QComboBox(self)
        self.baudRates.addItems([
            '9600', '14400', '19200', '28800', '31250', '38400', '51200',
            '56000', '57600', '76800', '115200', '128000', '230400', '256000', '921600'
        ])
        self.baudRates.setCurrentText('115200')
        self.baudRates.setMinimumHeight(30)

        self.dataBits = QtWidgets.QComboBox(self)
        self.dataBits.addItems(['5 bit', '6 bit', '7 bit', '8 bit'])
        self.dataBits.setCurrentIndex(3)
        self.dataBits.setMinimumHeight(30)

        self._parity = QtWidgets.QComboBox(self)
        self._parity.addItems(['No Parity', 'Even Parity', 'Odd Parity', 'Space Parity', 'Mark Parity'])
        self._parity.setCurrentIndex(0)
        self._parity.setMinimumHeight(30)

        self.stopBits = QtWidgets.QComboBox(self)
        self.stopBits.addItems(['One Stop', 'One And Half Stop', 'Two Stop'])
        self.stopBits.setCurrentIndex(0)
        self.stopBits.setMinimumHeight(30)

        self._flowControl = QtWidgets.QComboBox(self)
        self._flowControl.addItems(['No Flow Control', 'Hardware Control', 'Software Control'])
        self._flowControl.setCurrentIndex(0)
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
    window.show()
    app.exec()
