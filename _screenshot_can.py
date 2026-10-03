"""Screenshot harness for CAN mode: open a virtual bus, pump some traffic
through it, and save a PNG of the main window. Usage: python _screenshot_can.py out.png [fixed]"""
import sys
from PyQt5 import QtWidgets, QtCore
import AxxTerm

out = sys.argv[1] if len(sys.argv) > 1 else "screenshot_can.png"
fixed = len(sys.argv) > 2 and sys.argv[2] == 'fixed'

app = QtWidgets.QApplication([])
w = AxxTerm.SerialMonitor()
w.resize(1400, 860)
w.toolBar.modeCombo.setCurrentText('CAN')
w.toolBar.canInterfaces.setCurrentText('Virtual')
if fixed:
    w.canView.view_mode_combo.setCurrentText('Fixed')
w.show()
w.portOpen(True)
w.toolBar.portOpenButton.setChecked(True)

# The bus now opens on a worker thread; wait for it before pumping frames
import time
deadline = time.monotonic() + 10
while w._can_bus is None and time.monotonic() < deadline:
    app.processEvents()
    time.sleep(0.01)
assert w._can_bus is not None, 'CAN bus did not open'

peer = AxxTerm.pycan.Bus(interface='virtual', channel=0)
count = [0]


def pump():
    import random
    ids = [(0x18FEF100, True), (0xCF00400, True), (0x123, False), (0x7F, False)]
    can_id, ext = ids[count[0] % len(ids)]
    data = bytes((count[0] * 17 + i) & 0xFF for i in range(8 if ext else 2))
    peer.send(AxxTerm.pycan.Message(arbitration_id=can_id, is_extended_id=ext, data=data))
    count[0] += 1
    if count[0] == 12:
        w.sendCanFrame('18EF1234', True, '01 02 03 04')


timer = QtCore.QTimer()
timer.timeout.connect(pump)
timer.start(40)


def grab():
    timer.stop()
    w.grab().save(out)
    peer.shutdown()
    w.portOpen(False)
    w.close()
    app.quit()


QtCore.QTimer.singleShot(2500, grab)
app.exec_()
print("saved", out)
