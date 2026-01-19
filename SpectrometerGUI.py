import sys
import numpy as np 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, 
    QPushButton, QLabel, QDoubleSpinBox, 
    QVBoxLayout, QHBoxLayout, QMessageBox, QGroupBox, QComboBox
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

class KymeraPanel(QWidget):
    status_changed = pyqtSignal(str)

    def __init__(self, kymera_controller):
        super().__init__()
        self.kymera = kymera_controller
        self._build_ui()
    
    def _build_ui(self):
        main = QVBoxLayout()
        ctrl_group = QGroupBox("Kymera Control")
        ctrl_layout = QVBoxLayout()

        g_layout = QHBoxLayout()
        g_layout.addWidget(QLabel("Grating:"))
        self.grating_combo = QComboBox()
        ctrl_layout.addLayout(g_layout)
        g_layout.addWidget(self.grating_combo)
    
        wl_layout = QHBoxLayout()
        wl_layout.addWidget(QLabel("Central Wavelength"))
        self.wl_spin = QDoubleSpinBox()
        #CHECK RANGE
        self.wl_spin.setRange(200.0, 2000.0)
        self.wl_spin.setDecimals(2)
        self.wl_spin.setSingleStep(1.0)
        ctrl_layout.addLayout(wl_layout)
        wl_layout.addWidget(self.wl_spin)

        self.apply_btn = QPushButton("Apply")
        ctrl_layout.addWidget(self.apply_btn)

        ctrl_group.setLayout(ctrl_layout)
        main.addWidget(ctrl_group)

        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()

        self.status_grating = QLabel("Grating: --")
        self.status_wl = QLabel("Central Wavelength: -- nm")
        self.status_range = QLabel("Range: -- - -- nm")

        status_layout.addWidget(self.status_grating)
        status_layout.addWidget(self.status_wl)
        status_layout.addWidget(self.status_range)

        status_group.setLayout(status_layout)
        main.addWidget(status_group)

        main.addStretch()

        self.apply_btn.clicked.connect(self.apply_settings)
    
    def populate_gratings(self):
        self.grating_combo.clear()
        gratings = self.kymera.list_gratings()

        for i, name in enumerate(gratings):
            self.grating_combo.addItem(str(name), i)
    
    def apply_settings(self):
        try:
            idx = self.grating_combo.currentData()
            wl = self.wl_spin.value()

            self.kymera.set_grating(idx)
            self.kymera.set_central_wavelength(wl)

            self.refresh_status()
            self.status_changed.emit("Kymera settings applied")
        except Exception as e:
            self.status_changed.emit(f"Error applying Kymera settings: {e}")

    def refresh_status(self):
        try:
            g = self.kymera.get_grating()
            wl = self.kymera.get_central_wavelength()
            wl0, wl1 = self.kymera.get_wavelength_span()

            self.status_grating.setText(f"Grating: {g}")
            self.status_wl.setText(f"Central Wavelength: {wl:.2f} nm")
            self.status_range.setText(f"Range: {wl0:.1f} - {wl1:.1f} nm")
        except Exception as e:
            print("KymeraPanel.refresh_status error:", e)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectrometer Controller")
        self.camera = AndorCameraController()
        self.kymera = KymeraController()
        print("Before creating kymera panel:", hasattr(self, "kymera_panel"))
        self.kymera_panel = KymeraPanel(self.kymera)
        print("After creating kymera panel:", hasattr(self, "kymera_panel"))

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
        #self.kymera_panel = KymeraPanel(self.kymera)
        self.kymera_panel.setEnabled(False)
        self.kymera_panel.status_changed.connect(self.log_status)


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
        controls.addWidget(self.kymera_panel)
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

            self.kymera.setup_from_camera(self.camera.cam)

            self.kymera_panel.populate_gratings()
            self.kymera_panel.refresh_status()
            self.kymera_panel.setEnabled(True)

            self.log_status("Camera & Kymera connected", level="ok")
        except Exception as e:
            self.log_status(f"Camera & Kymera connection failed: {e}", level="error")
    
    def acquire_image(self):
        self.log_status("Acquiring image...", level="info")
        self.worker = Acquisitionworker(self.camera)
        self.worker.image_ready.connect(self.display_image)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(lambda:self.log_status("Image acquired", level="ok"))
        self.worker.start()
    
    def display_image(self, img):
        self.ax.clear()
        self.ax.imshow(img, cmap="gray")
        self.ax.set_title("Acquired Image")
        self.canvas.draw()
    
    def show_error(self, msg):
        QMessageBox.critical(self, "Acquisition Error", msg)
        self.log_status(msg, level="error")
    
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