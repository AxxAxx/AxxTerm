"""Screenshot harness: render the AxxTerm main window and save a PNG. Usage: python _screenshot.py out.png"""
import sys
from PyQt5 import QtWidgets, QtCore
import AxxTerm

out = sys.argv[1] if len(sys.argv) > 1 else "screenshot.png"
app = QtWidgets.QApplication([])
w = AxxTerm.SerialMonitor()
w.resize(1400, 860)
w.show()
def grab():
    w.grab().save(out)
    app.quit()
QtCore.QTimer.singleShot(1500, grab)
app.exec_()
print("saved", out)
