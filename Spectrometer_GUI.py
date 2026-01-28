import sys 
from PyQt6.QtCore import pyqtSignal
import numpy as np 
from numpy.random import noncentral_chisquare
import pyqtgraph as pg
pg.setConfigOptions(antialias=True)
from PyQt6.QtWidgets import (
    QApplication, QWidget,  QPushButton, QVBoxLayout, QLabel, 
    QLineEdit, QMessageBox, QComboBox, QGroupBox, QHBoxLayout, QSpinBox
)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
import matplotlib.pyplot as plt 

from Spectrometer import (
    AndorCameraController, KymeraController, SpectrometerController
)

class AcquireWorker(QThread):
    finished = pyqtSignal(object, object, object)
    error = pyqtSignal(str)

    def __init__(self, spectrometer, laser_wl):
        super().__init__()
        self.spectrometer = spectrometer
        self.laser_wl = laser_wl

    def run(self):
        try:
            spectrum, wl, raman = self.spectrometer.acquire_spectrum(self.laser_wl)
            self.finished.emit(spectrum, wl, raman)
        except Exception as e:
            self.error.emit(str(e))

class SpectrometerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectrometer")

        self.cam = AndorCameraController()
        self.kymera = KymeraController()
        self.spec = SpectrometerController(self.cam, self.kymera)

        self.connect_btn = QPushButton("Connect")
        self.acquire_btn = QPushButton("Acquire Spectrum")

        self.laser_edit = QLineEdit("532")
        self.status_label = QLabel("Disconnected")

        self.last_spectrum = None
        self.last_wavelength = None
        self.last_raman = None

        self.xaxis_combo = QComboBox()
        self.xaxis_combo.addItems(["Raman shift (cm${-1}$)", "Wavelength (nm)"])
        self.xaxis_combo.currentIndexChanged.connect(self.update_plot_axis)

        self.plot_widget = pg.PlotWidget(title="Spectrum")
        self.plot_widget.setLabel("left", "Intensity (counts)")
        self.plot_widget.setLabel("bottom", "Raman shift (cm${-1}$)")
        self.plot_widget.showGrid(x=True, y=True)

        self.spectrum_curve = self.plot_widget.plot([], [])

        cooling_box = QGroupBox("Cooling")
        cooling_layout = QHBoxLayout()

        self.temp_setpoint = QSpinBox()
        self.temp_setpoint.setRange(-100, 30)
        self.temp_setpoint.setValue(-90)
        self.temp_setpoint.setSuffix("°C")

        self.cooler_on = QPushButton("Cooler ON")
        self.cooler_off = QPushButton("Cooler OFF")

        self.temp_status = QLabel("Temp: -- °C | Status: --")
        self.temp_status_label = QLabel("Temp: -- °C | Status: --")
        cooling_layout.addWidget(QLabel("Setpoint:"))
        cooling_layout.addWidget(self.temp_setpoint)
        cooling_layout.addWidget(self.cooler_on)
        cooling_layout.addWidget(self.cooler_off)

        cooling_box.setLayout(cooling_layout)

        layout.addWidget(cooling_box)
        layout.addWidget(self.temp_status_label)
        
        layout.addWidget(QVBoxLayout())
        layout.addWidget(QLabel("Laser wavelength (nm):"))
        layout.addWidget(self.laser_edit)
        layout.addWidget(self.connect_btn)
        layout.addWidget(self.acquire_btn)
        layout.addWidget(self.status_label)
        layout.addWidget(self.xaxis_combo)
        layout.addWidget(self.plot_widget, stretch=1)

        self.setLayout(layout)
        self.connect_btn.clicked.connect(self.connect_devices)
        self.acquire_btn.clicked.connect(self.acquire)
        self.cooler_on.clicked.connect(self.enable_cooling)
        self.cooler_off.clicked.connect(self.disable_cooling)

        self.temp_timer = QTimer()
        self.temp_timer.timeout.connect(self.update_temperature)
        self.temp_timer.start(1000)


        self.set_connected(False)

    def set_connected(self, state: bool):
        self.acquire_btn.setEnabled(state)
        self.cooler_on.setEnabled(state)
        self.cooler_off.setEnabled(state)
        self.temp_setpoint.setEnabled(state)
        
    def connect_devices(self):
        try:
            self.cam.connect()

            self.cam.set_readout_mode("image")
            self.cam.setup_image_mode()
            self.cam.set_roi(hbin=1, vbin=1)
            self.cam.set_exposure(0.1)

            self.kymera.setup_from_camera(self.cam.cam)

            self.kymera.set_grating(1)
            self.kymera.set_central_wavelength(532)

            self.status_label.setText("Connected")
            self.set_connected(True)
        except Exception as e:
            QMessageBox.critical(self, "Connection error", str(e))
            self.set_connected(False)
    
    def enable_cooling(self):
        try:
            temp = self.temp_setpoint.value()
            self.cam.set_temp(temp, enable_cooler=True)
        except Exception as e:
            QMessageBox.critical(self, "Cooling error", str(e))
    
    def disable_cooling(self):
        try:
            self.cam.set_cooler(False)
        except Exception as e:
            QMessageBox.critical(self, "Cooling error", str(e))
    
    def update_temperature(self):
        if not self.cam.connected:
            self.temp_status_label.setText("Temp: -- °C | Status: Disconnected")
            return 
        
        try:
            temp = self.cam.get_temperature()
            status = self.cam.get_temp_status()
            cooler = self.cam.cooler()
        
            self.temp_status_label.setText(f"Temp: {temp:.1f} °C | Status: {status} | Cooler: {'ON' if cooler else 'OFF'}")
        except Excpetion as e:
            pass 

    def acquire(self):
        try:
            self.set_connected(False)
            laser_wl = float(self.laser_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Input error", "Laser wavelength must be a number")
            return 
    
        self.status_label.setText("Acquiring...")
        self.worker = AcquireWorker(self.spec, laser_wl)
        self.worker.finished.connect(self.show_result)
        self.worker.error.connect(self.show_error)
        self.worker.start()
    
    def update_plot(self):
        if self.last_spectrum is None:
            return 
        
        if self.xaxis_combo.currentText() == "Raman shift)":
            x = self.last_raman
            self.plot_widget.setLabel("bottom", "Raman shift (cm${-1}$)")
            self.plot_widget.getViewBow().invertX(True)
        else:
            x = self.last_wavelength
            self.plot_widget.setLabel("bottom", "Wavelength (nm)")
            self.plot_widget.getViewBox().invertX(False)
        
        self.spectrum_curve.setData(x, self.last_spectrum)
    
    def update_plot_axis(self):
        self.update_plot()

    def show_result(self, spectrum, wl, raman):
        self.status_label.setText("Done")

        self.last_spectrum = spectrum
        self.last_wavelength = wl
        self.last_raman = raman

        self.update_plot()
        self.set_connected(True)
    
    def show_error(self, msg):
        QMessageBox.critical(self, "Acquisition error", msg)
        self.status_label.setText("Error")
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = SpectrometerGUI()
    gui.show()
    sys.exit(app.exec())
