# -*- coding: utf-8 -*-
import sys
import math
import re
from PyQt5 import QtWidgets, QtCore, QtGui
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from PyQt5.QtSerialPort import QSerialPort, QSerialPortInfo
from PyQt5.QtGui import QPixmap, QTextCursor, QIcon
from PyQt5.QtWidgets import *
import struct
import numpy as np

plotlength = 100 #TO-DO make dynamic with user input field!

class SerialMonitor(QtWidgets.QMainWindow):
    def __init__(self):
        super(SerialMonitor, self).__init__()
        self.port = QSerialPort()
        self.serialDataView = SerialDataView(self)
        self.serialSendView = SerialSendView(self)

        self.setCentralWidget( QtWidgets.QWidget(self) )
        self.layout = QtWidgets.QVBoxLayout( self.centralWidget() )
        self.layout.addWidget(self.serialDataView)
        self.layout.addWidget(self.serialSendView)
        
        self.layout.setContentsMargins(3, 3, 3, 3)
        self.setWindowTitle('AxxTerm')
        self.setWindowIcon(QtGui.QIcon('dsub9_GREEN_30px.png'))

        ### Tool Bar ###
        self.toolBar = ToolBar(self)
        self.addToolBar(self.toolBar)

        ### Status Bar ###
        self.setStatusBar( QtWidgets.QStatusBar(self) )
        self.statusText = QtWidgets.QLabel(self)
        self.statusBar().addWidget( self.statusText )
        
        ### Signal Connect ###
        self.toolBar.portOpenButton.clicked.connect(self.portOpen)
        self.serialSendView.serialSendSignal.connect(self.sendFromPort)
        self.port.readyRead.connect(self.readFromPort)
        
    def portOpen(self, flag):
        
        if flag:
            self.port.setBaudRate( self.toolBar.baudRate() )
            self.port.setPortName( self.toolBar.portName() )
            self.port.setDataBits( self.toolBar.dataBit() )
            self.port.setParity( self.toolBar.parity() )
            self.port.setStopBits( self.toolBar.stopBit() )
            self.port.setFlowControl( self.toolBar.flowControl() )

            r = self.port.open(QtCore.QIODevice.ReadWrite)
            if not r:
                self.statusText.setText('Port open error')
                self.toolBar.portOpenButton.setChecked(False)
                self.toolBar.serialControlEnable(True)
            else:
                self.statusText.setText('Port opened')
                self.toolBar.serialControlEnable(False)
                self.pixmap_green = QPixmap('dsub9_GREEN_30px.png')
                self.serialDataView.label.setPixmap(self.pixmap_green)
                
        else:
            self.port.close()
            self.statusText.setText('Port closed')
            self.toolBar.serialControlEnable(True)
            self.pixmap_red = QPixmap('dsub9_RED_30px.png')
            self.serialDataView.label.setPixmap(self.pixmap_red)
        
        
    def readFromPort(self):
        data = self.port.readAll()
        if len(data) > 0:
            self.serialDataView.appendSerialText( QtCore.QTextStream(data).readAll(), "read")

    def sendFromPort(self, text):
        if(self.serialSendView.charMode.currentText() == 'HEX'):
            if(self.serialSendView.lineEnding.currentIndex() == 1):
                text = text+'0A'
            elif(self.serialSendView.lineEnding.currentIndex() == 2):
                text = text+'0D'
            elif(self.serialSendView.lineEnding.currentIndex() == 3):
                text = text+'0D0A'
            try:
                self.port.write( bytes.fromhex(text))
                self.statusText.setText('')
            except:
                self.statusText.setText('Not a valid HEX string')
        elif(self.serialSendView.charMode.currentText() == 'ASCII'):
            if(self.serialSendView.lineEnding.currentIndex() == 1):
                text = text+'\n'
            elif(self.serialSendView.lineEnding.currentIndex() == 2):
                text = text+'\r'
            elif(self.serialSendView.lineEnding.currentIndex() == 3):
                text = text+'\r\n'
            try:
                self.port.write(text.encode())
                self.statusText.setText('')
            except:
                self.statusText.setText('Not a valid ASCII string')
        elif(self.serialSendView.charMode.currentText() == 'BINARY'):
            self.port.write(int(text, 2).to_bytes(2, byteorder='big'))   
        

        self.serialDataView.appendSerialText( text, "send", self.serialSendView.charMode.currentText())
        

class SerialDataView(QtWidgets.QWidget):
    def __init__(self, parent):
        super(SerialDataView, self).__init__(parent)

        self.numberbuffer = []

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
        self.label_data_flow.setFont( QtGui.QFont('Segoe UI',12) )
        self.label_data_flow.setIndent(5)

        self.label_sent_data = QtWidgets.QLabel('Data: ASCII')
        self.label_sent_data.setFont( QtGui.QFont('Segoe UI',12) )
        self.label_sent_data.setIndent(5)

        self.graph_mode = QCheckBox("Show Graph")
        self.graph_mode.setFont( QtGui.QFont('Segoe UI',12) )
        self.graph_mode.stateChanged.connect(self.graph_state_changed)

        self.graph_channels = QSpinBox(minimum=1, maximum=12, value=4, prefix="Channels: ")
        self.graph_channels.setFont( QtGui.QFont('Segoe UI',12) )
        
        self.clear_button = QtWidgets.QPushButton('Clear ALL')
        self.clear_button.clicked.connect(self.clear_button_Clicked)
        self.clear_button.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)

        self.label = QLabel(self)
        self.pixmap_red = QPixmap('dsub9_RED_30px.png')
        self.label.setPixmap(self.pixmap_red)

        self.converter_label = QtWidgets.QLabel('Converter')
        self.converter_label.setFont( QtGui.QFont('Segoe UI',12) )
        self.converter_label.setIndent(5)

        self.convert_A_type = QtWidgets.QComboBox(self)
        self.convert_A_type.addItems(['HEX --> ASCII', 'HEX --> DECIMAL', 'HEX --> BINARY', 'ASCII --> HEX', 'ASCII --> DECIMAL', 'ASCII --> BINARY', 'DECIMAL --> HEX', 'DECIMAL --> ASCII', 'DECIMAL --> BINARY', 'BINARY --> HEX', 'BINARY --> ASCII', 'BINARY --> DECIMAL'])
        self.convert_A_type.setCurrentIndex(0)
        self.convert_A_type.setMinimumHeight(30)
        self.convert_A_type.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        self.convert_A_text = QtWidgets.QTextEdit(self)
        self.convert_A_text.setMaximumHeight(31)
        self.convert_A_text.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.convert_A_text.textChanged.connect(self.translate_data_from_A)
        self.convert_A_text.setFont( QtGui.QFont('Segoe UI',12) )



        self.convert_B_text = QtWidgets.QTextEdit(self)
        self.convert_B_text.setMaximumHeight(31)
        self.convert_B_text.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.convert_B_text.setFont( QtGui.QFont('Segoe UI',12) )



        self.groupBox = QGroupBox("CONVERTER")


        self.setLayout( QtWidgets.QGridLayout(self) )
        self.layout().addWidget(self.label_data_flow,   1, 3, 1, 2)
        self.layout().addWidget(self.label_sent_data,   1, 0, 1, 3)
        self.layout().addWidget(self.graph_mode,        1, 5, 1, 1, alignment=QtCore.Qt.AlignRight)
        self.layout().addWidget(self.graph_channels,    1, 4, 1, 1, alignment=QtCore.Qt.AlignRight)
        self.layout().addWidget(self.serialData,        2, 0, 1, 3)
        self.layout().addWidget(self.serialDataHex,     2, 3, 1, 3)
        self.layout().addWidget(self.label,             4, 0, 1, 1)
        self.layout().addWidget(self.converter_label,   3, 1, 1, 1)
        self.layout().addWidget(self.convert_A_type,    4, 1, 1, 1)
        self.layout().addWidget(self.convert_A_text,    4, 2, 1, 1)
        self.layout().addWidget(self.convert_B_text,    4, 3, 1, 2)
        self.layout().addWidget(self.clear_button,      4, 5, 1, 1, alignment=QtCore.Qt.AlignRight)
        self.layout().setContentsMargins(2, 2, 2, 2)


    def graph_state_changed(self):
        if(self.graph_mode.isChecked()):
            self.graphWidget = pg.PlotWidget(title="Plot")
            self.graphWidget.setBackground('#FFFFFFFF')
            self.graphWidget.setMinimumHeight(300)

            self.graphWidget.plotItem.getAxis('bottom').setPen( pg.mkPen(color='#000000') )
            self.graphWidget.plotItem.getAxis('left').setPen( pg.mkPen(color='#000000') )
            self.graphWidget.plotItem.showGrid(True, True, 0.3)
            self.graphWidget.setXRange(0, plotlength)
            self.data = [ self.graphWidget.plotItem.plot(pen=pg.mkPen('r', width=2)), self.graphWidget.plotItem.plot(pen=pg.mkPen('b', width=2)), self.graphWidget.plotItem.plot(pen=pg.mkPen(color=(50,150,0), width=2)), self.graphWidget.plotItem.plot(pen=pg.mkPen('k', width=2))]
            self.data[0].setData( np.zeros(plotlength) )
            self.data[1].setData( np.zeros(plotlength) )
            self.data[2].setData( np.zeros(plotlength) )
            self.data[3].setData( np.zeros(plotlength) )
            self.numberbuffer = []

            self.layout().addWidget(self.graphWidget,   0, 0, 1, 6)
        else:
            self.layout().removeWidget(self.graphWidget)
            self.graphWidget.deleteLater()
            self.graphWidget = None


    def appendData(self, data, yNum):
        rolled = np.roll(self.data[yNum].yData, -1)
        rolled[-1] = data
        self.data[yNum].setData(rolled)          

    def translate_data_from_A(self):
        if(self.convert_A_type.currentText() == 'HEX --> ASCII'):
            try:
                self.convert_B_text.clear()
                last_value = self.convert_A_text.toPlainText()
                self.convert_B_text.insertPlainText(bytes.fromhex(last_value).decode('ISO-8859-1'))
            except:
                self.convert_B_text.insertPlainText("not valid")

        elif(self.convert_A_type.currentText() == 'HEX --> DECIMAL'):
            try:
                self.convert_B_text.clear()
                last_value = self.convert_A_text.toPlainText()
                self.convert_B_text.insertPlainText(str(int(last_value, 16)))
                
            except:
                self.convert_B_text.insertPlainText("not valid")
                
        elif(self.convert_A_type.currentText() == 'HEX --> BINARY'):
            try:
                self.convert_B_text.clear()
                last_value = self.convert_A_text.toPlainText()
                scale = 16 ## equals to hexadecimal
                num_of_bits = 8
                self.convert_B_text.insertPlainText(bin(int(last_value, scale))[2:].zfill(num_of_bits))
            except:
                self.convert_B_text.insertPlainText("not valid")

        elif(self.convert_A_type.currentText() == 'ASCII --> HEX'):
            try:
                self.convert_B_text.clear()
                last_value = self.convert_A_text.toPlainText()
                self.convert_B_text.insertPlainText("0x"+last_value.encode('ISO-8859-1').hex())
            except:
                self.convert_B_text.insertPlainText("not valid")

        elif(self.convert_A_type.currentText() == 'ASCII --> DECIMAL'):
            try:
                self.convert_B_text.clear()
                last_value = self.convert_A_text.toPlainText()
                self.convert_B_text.insertPlainText(str(ord(last_value.encode('ISO-8859-1'))))
            except:
                self.convert_B_text.insertPlainText("not valid")

        elif(self.convert_A_type.currentText() == 'ASCII --> BINARY'):
            try:
                self.convert_B_text.clear()
                last_value = self.convert_A_text.toPlainText()
                self.convert_B_text.insertPlainText(bin(int.from_bytes(last_value.encode('ISO-8859-1'), "big")))
            except:
                self.convert_B_text.insertPlainText("not valid")

        elif(self.convert_A_type.currentText() == 'DECIMAL --> HEX'):
            try:
                self.convert_B_text.clear()
                last_value = self.convert_A_text.toPlainText()
                self.convert_B_text.insertPlainText(hex(int(last_value)))
            except:
                self.convert_B_text.insertPlainText("not valid")
        
        elif(self.convert_A_type.currentText() == 'DECIMAL --> ASCII'):
            try:
                self.convert_B_text.clear()
                last_value = self.convert_A_text.toPlainText()
                self.convert_B_text.insertPlainText(chr(int(last_value)))
            except:
                self.convert_B_text.insertPlainText("not valid")

        elif(self.convert_A_type.currentText() == 'DECIMAL --> BINARY'):
            try:
                self.convert_B_text.clear()
                last_value = self.convert_A_text.toPlainText()
                num_of_bits = 8
                self.convert_B_text.insertPlainText(str("{0:b}".format(int(last_value)).zfill(num_of_bits)))
            except:
                self.convert_B_text.insertPlainText("not valid")

        elif(self.convert_A_type.currentText() == 'BINARY --> HEX'):
            try:
                self.convert_B_text.clear()
                last_value = self.convert_A_text.toPlainText()
                self.convert_B_text.insertPlainText(hex(int(last_value, 2)))
            except:
                self.convert_B_text.insertPlainText("not valid")

        elif(self.convert_A_type.currentText() == 'BINARY --> ASCII'):
            try:
                self.convert_B_text.clear()
                last_value = self.convert_A_text.toPlainText()
                self.convert_B_text.insertPlainText(chr(int(last_value, 2)))
            except:
                self.convert_B_text.insertPlainText("not valid")

        elif(self.convert_A_type.currentText() == 'BINARY --> DECIMAL'):
            try:
                self.convert_B_text.clear()
                last_value = self.convert_A_text.toPlainText()
                self.convert_B_text.insertPlainText(str(int(last_value, 2)))
            except:
                self.convert_B_text.insertPlainText("not valid")


    def clear_button_Clicked(self):
        self.serialDataHex.clear()
        self.serialData.clear()
        self.convert_A_text.clear()
        self.convert_B_text.clear()

    def appendSerialText(self, appendText, direction, mode="ASCII"):
        if(direction == "send"):
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
        lastLength = math.ceil( len(lastData) / 3 )
        
        appendLists = []
        splitedByTwoChar = re.split( '(..)', appendText.encode().hex() )[1::2]
        if lastLength > 0:
            t = splitedByTwoChar[ : 16-lastLength ] + ['\n']
            appendLists.append( ' '.join(t) )
            splitedByTwoChar = splitedByTwoChar[ 16-lastLength : ]

        appendLists += [ ' '.join(splitedByTwoChar[ i*16 : (i+1)*16 ] + ['\n']) for i in range( math.ceil(len(splitedByTwoChar)/16) ) ]
        if len(appendLists[-1]) < 47:
            appendLists[-1] = appendLists[-1][:-1]
        if(direction == "send"):
                if(mode == 'HEX'):
                    self.serialData.insertPlainText(bytes.fromhex(appendText).decode('ISO-8859-1'))
                    self.serialDataHex.insertPlainText(appendText.upper())
                    
                elif(mode == 'ASCII'):
                    self.serialData.insertPlainText(appendText)
                    for insertText in appendLists:
                        self.serialDataHex.insertPlainText(insertText.upper())
        else:

            for insertText in appendLists:
                self.serialDataHex.insertPlainText(insertText.upper())
            self.serialData.insertPlainText(appendText)
            
            if(self.graph_mode.isChecked()):
                for i in appendText:
                    if i == "\n":
                        try:
                            self.appendData(float(''.join(self.numberbuffer).strip().split("\t")[0].split(":")[1]), 0)
                            self.appendData(float(''.join(self.numberbuffer).strip().split("\t")[1].split(":")[1]), 1)
                            self.appendData(float(''.join(self.numberbuffer).strip().split("\t")[2].split(":")[1]), 2)
                            self.appendData(float(''.join(self.numberbuffer).strip().split("\t")[3].split(":")[1]), 3)    
                        except:
                            pass
                        self.numberbuffer = []
                    else:
                        self.numberbuffer.append(i)

        self.serialData.moveCursor(QtGui.QTextCursor.End)
        self.serialDataHex.moveCursor(QtGui.QTextCursor.End)


# creating VLine class
class HLine(QFrame):
  
    # a simple Vertical line
    def __init__(self):
  
        super(HLine, self).__init__()
        self.setFrameShape(self.HLine|self.Sunken)

class SerialSendView(QtWidgets.QWidget):

    serialSendSignal = QtCore.pyqtSignal(str)

    def __init__(self, parent):
        super(SerialSendView, self).__init__(parent)
        
        self.history = []
        self.i = 0

        self.charMode = QtWidgets.QComboBox(self)
        self.charMode.addItems(['ASCII', 'HEX', 'BINARY'])
        self.charMode.setCurrentIndex(0)
        self.charMode.setMinimumHeight(30)
        self.charMode.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        self.lineEnding = QtWidgets.QComboBox(self)
        self.lineEnding.addItems(["""No line ending""", """LF '\\n', 0x0A""", """CR '\\r'', 0x0D""", """Both LF CR '\\r\\n'"""])
        self.lineEnding.setCurrentIndex(1)
        self.lineEnding.setMinimumHeight(30)
        self.lineEnding.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        self.sendData = QtWidgets.QTextEdit(self)
        self.sendData.installEventFilter(self)
        self.sendData.setAcceptRichText(False)
        self.sendData.setMaximumHeight(31)
        self.sendData.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.sendData.textChanged.connect(self.some_event)
        self.sendData.setFont( QtGui.QFont('Segoe UI',12) )

        self.sendButton = QtWidgets.QPushButton('Send')
        self.sendButton.clicked.connect(self.sendButtonClicked)
        self.sendButton.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        
        self.macrobuttoncolor = 'color: white; background-color: #006600'

        self.send_button_1 = QtWidgets.QPushButton('0x7F')
        self.send_button_1.setFont( QtGui.QFont('Segoe UI', 10, 60))
        self.send_button_1.clicked.connect(self.send_button_1_Clicked)
        self.send_button_1.setStyleSheet(self.macrobuttoncolor)
        self.send_button_1.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        self.send_button_2 = QtWidgets.QPushButton('FF')
        self.send_button_2.setFont( QtGui.QFont('Segoe UI', 10, 60))
        self.send_button_2.clicked.connect(self.send_button_2_Clicked)
        self.send_button_2.setStyleSheet(self.macrobuttoncolor)
        self.send_button_2.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        
        self.send_button_3 = QtWidgets.QPushButton('FF')
        self.send_button_3.setFont( QtGui.QFont('Segoe UI', 10, 60))
        self.send_button_3.clicked.connect(self.send_button_3_Clicked)
        self.send_button_3.setStyleSheet(self.macrobuttoncolor)
        self.send_button_3.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        
        self.send_button_4 = QtWidgets.QPushButton('0xBB')
        self.send_button_4.setFont( QtGui.QFont('Segoe UI', 10, 60))
        self.send_button_4.clicked.connect(self.send_button_4_Clicked)
        self.send_button_4.setStyleSheet(self.macrobuttoncolor)
        self.send_button_4.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        self.send_button_5 = QtWidgets.QPushButton('__SHORTPRESS__')
        self.send_button_5.setFont( QtGui.QFont('Segoe UI', 10, 60))
        self.send_button_5.clicked.connect(self.send_button_5_Clicked)
        self.send_button_5.setStyleSheet(self.macrobuttoncolor)
        self.send_button_5.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        
        self.send_button_6 = QtWidgets.QPushButton('__LONGPRESS__')
        self.send_button_6.setFont( QtGui.QFont('Segoe UI', 10, 60))
        self.send_button_6.clicked.connect(self.send_button_6_Clicked)
        self.send_button_6.setStyleSheet(self.macrobuttoncolor)
        self.send_button_6.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)       

        self.send_button_7 = QtWidgets.QPushButton('$$$')
        self.send_button_7.setFont( QtGui.QFont('Segoe UI', 10, 60))
        self.send_button_7.clicked.connect(self.send_button_7_Clicked)
        self.send_button_7.setStyleSheet(self.macrobuttoncolor)
        self.send_button_7.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        
        self.send_button_8 = QtWidgets.QPushButton('__OTA__')
        self.send_button_8.setFont( QtGui.QFont('Segoe UI', 10, 60))
        self.send_button_8.clicked.connect(self.send_button_8_Clicked)
        self.send_button_8.setStyleSheet(self.macrobuttoncolor)
        self.send_button_8.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)  

        #self.setLayout( QtWidgets.QHBoxLayout(self) )
        self.setLayout( QtWidgets.QGridLayout(self) )

        self.layout().addWidget(HLine(),                0, 0, 1, 8) 
        self.layout().addWidget(self.send_button_1,     2, 0, 1, 1)
        self.layout().addWidget(self.send_button_2,     2, 1, 1, 1)
        self.layout().addWidget(self.send_button_3,     2, 2, 1, 1)
        self.layout().addWidget(self.send_button_4,     2, 3, 1, 1)
        self.layout().addWidget(self.send_button_5,     2, 4, 1, 1)
        self.layout().addWidget(self.send_button_6,     2, 5, 1, 1)
        self.layout().addWidget(self.send_button_7,     2, 6, 1, 1)
        self.layout().addWidget(self.send_button_8,     2, 7, 1, 1)
        self.layout().addWidget(self.charMode,          1, 0, 1, 1)
        
        self.layout().addWidget(self.sendData,          1, 1, 1, 5)
        self.layout().addWidget(self.lineEnding,        1, 6, 1, 1)
        self.layout().addWidget(self.sendButton,        1, 7, 1, 1)
        self.layout().setContentsMargins(1, 1, 1, 1)
    
    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress and obj is self.sendData:
            #if event.key() == QtCore.Qt.Key_Return and self.sendData.hasFocus():
            if((event.key() == 16777220 or event.key() == 16777221) and self.sendData.hasFocus()):
                self.serialSendSignal.emit( self.sendData.toPlainText() )
                self.history.append(self.sendData.toPlainText())
                self.sendData.clear()
                self.i = 0
            elif ((event.key() == 16777235) and self.sendData.hasFocus()):
                try:
                    self.i = self.i + 1
                    self.sendData.clear()
                    self.sendData.insertPlainText(self.history[-1*max(min(len(self.history), self.i), 1)])
                    
                except:
                    pass
            elif ((event.key() == 16777237) and self.sendData.hasFocus()):
                try:
                    self.i = self.i - 1
                    self.sendData.clear()
                    self.sendData.insertPlainText(self.history[-1*max(min(len(self.history), self.i), 1)])
                    
                except:
                    pass
            
        return super().eventFilter(obj, event)

    def some_event(self):
        try:
            last_value = self.sendData.toPlainText()[-1]
            if last_value == '\n':
                self.sendData.setPlainText(self.sendData.toPlainText()[:-1])
                self.sendData.moveCursor(QTextCursor.End)
        except IndexError:
            pass

    def sendRaw(self, raw_hex_data):
        oldmode = self.charMode.currentIndex()
        oldending = self.lineEnding.currentIndex()
        self.charMode.setCurrentText("HEX")
        self.lineEnding.setCurrentText("No line ending")
        self.serialSendSignal.emit(raw_hex_data)   
        self.charMode.setCurrentIndex(oldmode)
        self.lineEnding.setCurrentIndex(oldending)

    def sendButtonClicked(self):
        self.serialSendSignal.emit( self.sendData.toPlainText() )
        self.history.append(self.sendData.toPlainText())
        self.sendData.clear()
        self.i = 0

    def send_button_1_Clicked(self):
        self.sendRaw( '7F' )
        
    def send_button_2_Clicked(self):
        self.sendRaw( 'FF' )

    def send_button_3_Clicked(self):
        self.sendRaw('FF')

    def send_button_4_Clicked(self):
        self.sendRaw( 'BB' )

    def send_button_5_Clicked(self):
        self.sendRaw( '5f5f53484f525450524553535f5f0a' )

    def send_button_6_Clicked(self):
        self.sendRaw('5f5f4c4f4e4750524553535f5f0a')

    def send_button_7_Clicked(self):
        self.sendRaw('242424')  

    def send_button_8_Clicked(self):
        self.sendRaw('5F 5F 4F 54 41 5F 5F 0A')  

class ToolBar(QtWidgets.QToolBar):
    def __init__(self, parent):
        super(ToolBar, self).__init__(parent)
        
        self.portOpenButton = QtWidgets.QPushButton('Open')
        self.portOpenButton.setCheckable(True)
        self.portOpenButton.setMinimumHeight(32)

        self.portScanButton = QtWidgets.QPushButton('Scan')
        self.portScanButton.setCheckable(True)
        self.portScanButton.clicked.connect(self.scan_button_Clicked)
        self.portScanButton.setMinimumHeight(32)

        self.portNames = QtWidgets.QComboBox(self)
        self.portNames.addItems([ port.portName() for port in QSerialPortInfo().availablePorts() ])
        self.portNames.setMinimumHeight(30)

        self.baudRates = QtWidgets.QComboBox(self)
        self.baudRates.addItems([
            '9600', '14400', '19200', '28800', '31250', '38400', '51200', '56000', '57600', '76800', '115200', '128000', '230400', '256000', '921600'
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

        self.addWidget( self.portOpenButton )
        self.addWidget( self.portNames)
        self.addWidget( self.portScanButton )
        self.addWidget( self.baudRates)
        self.addWidget( self.dataBits)
        self.addWidget( self._parity)
        self.addWidget( self.stopBits)
        self.addWidget( self._flowControl)

    def scan_button_Clicked(self):
        self.portNames.clear()
        self.portNames.addItems([ port.portName() for port in QSerialPortInfo().availablePorts() ])

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
        if (self._parity.currentIndex()>0):
            return self._parity.currentIndex()+1
        else:
            return self._parity.currentIndex()

    def stopBit(self):
        return self.stopBits.currentIndex()

    def flowControl(self):
        return self._flowControl.currentIndex()

def _chunks(text, chunk_size):
    """Chunk text into chunk_size."""
    for i in range(0, len(text), chunk_size):
        yield text[i:i+chunk_size]

def str_to_hex(text):
    """Convert text to hex encoded bytes."""
    return ''.join('{:02x}'.format(ord(c)) for c in text)

def hex_to_raw(hexstr):
    """Convert a hex encoded string to raw bytes."""
    return ''.join(chr(int(x, 16)) for x in _chunks(hexstr, 2))
    
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app_icon = QIcon("dsub9_GREEN_30px.png")
    app.setWindowIcon(app_icon)
    window = SerialMonitor()
    window.show()
    app.exec()
