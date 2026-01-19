import sys
import numpy as np 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, 
    QPushButton, QLabel, QDoubleSpinBox, 
    QVBoxLayout, QHBoxLayout, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from Spectrometer_ExCode import AndorCameraController, KymeraController, SpectrometerController

class Acquisitionworker(QThread):
    image_ready = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, camera):
        super().__init__()
        self.camera = camera

    def run(self):
        try:
            img = self.camera.acquire_single()
            self.image_ready.emit(img)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectrometer Controller")
        self.camera = AndorCameraController()
        self._build_ui()
        self.status = self.statusBar()
        self.init_status_label()
        self.log_status("Ready")
        self.temp_label = QLabel("CCD Temp: -- °C")
        self.status.addPermanentWidget(self.temp_label)
        self.temp_timer = QTimer(self)
        self.temp_timer.timeout.connect(self.update_temperature)
        self.temp_timer.start(1000)
        self.keep_cooling = False
        print("Starting temperature timer")


    def init_status_label(self):
        self.status_label = QLabel("Ready")
        self.status.addPermanentWidget(self.status_label)
    
    def log_status(self, message, level="info"):
        colors = {
            "info": "black",
            "warn": "orange",
            "error": "red",
            "ok": "green"
        }
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {colors.get(level, 'black')};")
    
    def get_temperature(self):
        with self._lock:
            return self.camera.get_temperature()
    
    def maintain_cooling(self):
        if not self.cooler_enabled:
            return
        with self._lock:
            self.cam.set_cooler(True)
            if self.temperature_setpoint is not None:
                self.cam.set_temperature(self.temperature_setpoint)
    
    def update_temperature(self):
        if not self.camera.connected:
            self.temp_label.setText("CCD Temp: -- °C")
            self.temp_label.setStyleSheet("color: black;")
            return
        
        try:
            self.camera.maintain_cooling()
            temp = self.camera.get_temperature()

            if temp is None:
                self.temp_label.setText("CCD Temp: -- °C")
                self.temp_label.setStyleSheet("color: red;")
                return
            
            self.temp_label.setText(f"CCD Temp: {temp:.2f} °C")

            if temp < -60:
                color = "green"
            elif temp < -20:
                color = "orange"
            else:
                color = "red"
            self.temp_label.setStyleSheet(f"color: {color};")
            print("Temperature updated")

        except Exception as e:
            print("Temperature update error:", e)
            self.temp_label.setText("CCD Temp: error")
            self.temp_label.setStyleSheet("color: red;")
    
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
    
        self.connect_btn = QPushButton("Connect Camera")
        self.acquire_btn = QPushButton("Acquire Image")
        self.exposure_spin = QDoubleSpinBox()
        #Check range 
        self.exposure_spin.setRange(0.01, 1000)
        self.exposure_spin.setValue(0.1)
        self.exposure_spin.setSuffix(" s")

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)

        controls = QVBoxLayout()
        controls.addWidget(QLabel("Exposure"))
        controls.addWidget(self.exposure_spin)
        controls.addWidget(self.connect_btn)
        controls.addWidget(self.acquire_btn)
        controls.addStretch()

        main = QHBoxLayout()
        main.addLayout(controls)
        main.addWidget(self.canvas)

        central.setLayout(main)

        self.connect_btn.clicked.connect(self.connect_camera)
        self.acquire_btn.clicked.connect(self.acquire_image)

    def connect_camera(self):
        self.log_status("Connecting to camera...", level="info")
        try:
            self.camera.connect()
            self.camera.set_exposure(self.exposure_spin.value())
            self.camera.enable_cooling(-100)
            self.camera.keep_cooling = True
            self.log_status("Camera connected", level="ok")
        except Exception as e:
            self.log_status(f"Connection failed: {e}", level="error")
    
    def acquire_image(self):
        self.log_status("Acquiring image...", level="info")
        self.worker = Acquisitionworker(self.camera)
        self.worker.image_ready.connect(self.display_image)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(lambda:self.log_status("Image acquired", level="ok"))
        self.worker.start()

    def show_error(self, msg):
        self.log_status(f"Error: {msg}", level="error")
    
    def display_image(self, img):
        self.ax.clear()
        self.ax.imshow(img, cmap="gray")
        self.ax.set_title("Acquired Image")
        self.canvas.draw()
    
    def show_error(self, msg):
        QMessageBox.ciritical(self, "Acquisition Error", msg)
    
    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Exit", "Exit GUI?\n\nCooling with remain ON until power off.", 
        QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            event.accept()
        else: 
            event.ignore()
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())