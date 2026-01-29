import sys 
from PyQt6.QtCore import pyqtSignal
import numpy as np 
from numpy.random import noncentral_chisquare
import pyqtgraph as pg
pg.setConfigOptions(antialias=True)
from PyQt6.QtWidgets import (
    QApplication, QWidget,  QPushButton, QVBoxLayout, QGridLayout,QLabel, 
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

        layout = QVBoxLayout()
        layout.addWidget(cooling_box)
        layout.addWidget(self.temp_status_label)

        fan_box = QGroupBox("Fan")
        fan_layout = QHBoxLayout()
        self.fan_combo = QComboBox()
        self.fan_combo.addItems(["full", "low", "off"])
        
        fan_layout.addWidget(QLabel("Mode:"))
        fan_layout.addWidget(self.fan_combo)

        fan_box.setLayout(fan_layout)
        layout.addWidget(fan_box)

        trigger_box = QGroupBox("Trigger")
        trigger_layout = QHBoxLayout()

        self.trigger_combo = QComboBox()
        self.trigger_combo.addItems(["Internal", "Software"])

        trigger_layout.addWidget(QLabel("Mode:"))
        trigger_layout.addWidget(self.trigger_combo)
        trigger_box.setLayout(trigger_layout)

        layout.addWidget(trigger_box)

        self.roi_box = QGroupBox("ROI/Binning")
        roi_layout = QGridLayout()

        self.hbin_spin = QSpinBox()
        self.hbin_spin.setRange(1, 16)
        self.hbin_spin.setValue(1)

        self.vbin_spin = QSpinBox()
        self.vbin_spin.setRange(1, 16)
        self.vbin_spin.setValue(1)

        self.hstart_spin = QSpinBox()
        self.hstart_spin.setRange(0, 10000)
        self.hstart_spin.setValue(0)

        self.hend_spin = QSpinBox()
        self.hend_spin.setRange(0, 10000)
        self.hend_spin.setValue(0)

        self.vstart_spin = QSpinBox()
        self.vstart_spin.setRange(0, 10000)
        self.vstart_spin.setValue(0)

        self.vend_spin = QSpinBox()
        self.vend_spin.setRange(0, 10000)
        self.vend_spin.setValue(0)

        self.apply_roi_btn = QPushButton("Apply ROI")

        roi_layout.addWidget(QLabel("H Bin:"), 0, 0)
        roi_layout.addWidget(self.hbin_spin, 0, 1)
        roi_layout.addWidget(QLabel("V Bin:"), 0, 2)
        roi_layout.addWidget(self.vbin_spin, 0, 3)
        roi_layout.addWidget(QLabel("H Start:"), 1, 0)
        roi_layout.addWidget(self.hstart_spin, 1, 1)
        roi_layout.addWidget(QLabel("H End:"), 1, 2)
        roi_layout.addWidget(self.hend_spin, 1, 3)
        roi_layout.addWidget(QLabel("V Start:"), 2, 0)
        roi_layout.addWidget(self.vstart_spin, 2, 1)
        roi_layout.addWidget(QLabel("V End:"), 2, 2)
        roi_layout.addWidget(self.vend_spin, 2, 3)
        roi_layout.addWidget(self.apply_roi_btn, 3, 0, 1, 4)
        self.roi_box.setLayout(roi_layout)
        layout.addWidget(self.roi_box)

        self.single_box = QGroupBox("Single track")
        single_layout = QGridLayout()

        self.single_center_spin = QSpinBox()
        self.single_center_spin.setRange(0, 10000)
        self.single_center_spin.setValue(100)

        self.single_width_spin = QSpinBox()
        self.single_width_spin.setRange(1, 1000)
        self.single_width_spin.setValue(1)

        single_layout.addWidget(QLabel("Center:"), 0, 0)
        single_layout.addWidget(self.single_center_spin, 0, 1)
        single_layout.addWidget(QLabel("Width:"), 1, 0)
        single_layout.addWidget(self.single_width_spin, 1, 1)
        self.single_box.setLayout(single_layout)
        layout.addWidget(self.single_box)

        self.multi_box = QGroupBox("Multi track")
        multi_layout = QGridLayout()

        self.multi_number_spin = QSpinBox()
        self.multi_number_spin.setRange(1, 100)
        self.multi_number_spin.setValue(1)

        self.multi_height_spin = QSpinBox()
        self.multi_height_spin.setRange(1, 1000)
        self.multi_height_spin.setValue(1)

        self.multi_offset_spin = QSpinBox()
        self.multi_offset_spin.setRange(0, 1000)
        self.multi_offset_spin.setValue(0)

        multi_layout.addWidget(QLabel("Number:"), 0, 0)
        multi_layout.addWidget(self.multi_number_spin, 0, 1)
        multi_layout.addWidget(QLabel("Height:"), 1, 0)
        multi_layout.addWidget(self.multi_height_spin, 1, 1)
        multi_layout.addWidget(QLabel("Offset:"), 2, 0)
        multi_layout.addWidget(self.multi_offset_spin, 2, 1)
        self.multi_box.setLayout(multi_layout)
        layout.addWidget(self.multi_box)

        geom_box = QGroupBox("Readout mode")
        geom_layout = QHBoxLayout()
        self.geom_combo = QComboBox()
        #eventuall add continuous mode warning on
        self.geom_combo.addItems(["fvb", "image", "single_track", "multi_track"])

        geom_layout.addWidget(QLabel("Mode:"))
        geom_layout.addWidget(self.geom_combo)
        geom_box.setLayout(geom_layout)
        layout.addWidget(geom_box)

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
        self.apply_roi_btn.clicked.connect(self.apply_roi)
        self.fan_combo.currentTextChanged.connect(self.set_fan_mode)
        self.geom_combo.currentTextChanged.connect(self.set_geometry_mode)
        self.geom_combo.currentTextChanged.connect(self.update_geometry_ui)
        self.trigger_combo.currentTextChanged.connect(self.set_trigger_mode)

        self.temp_timer = QTimer()
        self.temp_timer.timeout.connect(self.update_temperature)
        self.temp_timer.start(1000)


        self.set_connected(False)

    def set_connected(self, state: bool):
        self.acquire_btn.setEnabled(state)
        self.cooler_on.setEnabled(state)
        self.cooler_off.setEnabled(state)
        self.temp_setpoint.setEnabled(state)
        self.apply_roi_btn.setEnabled(state)
        self.fan_combo.setEnabled(state)
        
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
            fan = self.cam.get_fan_mode()
        
            self.temp_status_label.setText(f"Temp: {temp:.1f} °C | Status: {status} | Cooler: {'ON' if cooler else 'OFF'} | Fan: {fan}")
        except Exception as e:
            pass 
    
    def set_fan_mode(self, mode):
        try:
            self.cam.set_fan_mode(mode)
        except Exception as e:
            QMessageBox.critical(self, "Fan error", str(e))
    
    def set_trigger_mode(self, text):
        if not self.cam.connected:
            return
        if text == "Internal":
            self.cam.set_internal_trigger()
        elif text == "Software":
            self.cam.set_software_trigger()
        else:
            QMessageBox.critical(self, "Trigger mode error", "Invalid trigger mode")
        self.status_label.setText(f"Trigger mode: {mode}")

    def apply_roi(self):
        try: 
            hbin = self.hbin_spin.value()
            vbin = self.vbin_spin.value()
            hstart = self.hstart_spin.value()
            hend = self.hend_spin.value() or None
            vstart = self.vstart_spin.value()
            vend = self.vend_spin.value() or None

            self.cam.set_roi(hbin=hbin, vbin=vbin, hstart=hstart, hend=hend, vstart=vstart, vend=vend)
            self.kymera.setup_from_camera(self.cam.cam)
            self.wavelength_nm = self.kymera.get_calibration_nm()
            self.status_label.setText("ROI applied")
        except Exception as e:
            QMessageBox.critical(self, "ROI error", str(e))
    
    def set_geometry_mode(self, mode):
        try:
            self.cam.set_readout_mode(mode)

            if mode == "single_track":
                center = self.single_center_spin.value()
                width = self.single_width_spin.value()
                self.cam.setup_single_mode(center, width)
            
            elif mode == "multi track":
                number = self.multi_number_spin.value()
                height = self.multi_height_spin.value()
                offset = self.multi_offset_spin.value()
                self.cam.setup_multi_mode(number, height, offset)
            
            elif mode == "image":
                hstart = self.hstart_spin.value()
                hend = self.hend_spin.value() or None
                vstart = self.vstart_spin.value()
                vend = self.vend_spin.value() or None
                hbin = self.hbin_spin.value()
                vbin = self.vbin_spin.value()
                self.cam.setup_image_mode(hstart=hstart, hend=hend, vstart=vstart, vend=vend, hbin=hbin, vbin=vbin)

            self.kymera.setup_from_camera(self.cam.cam)
            self.status_label.setText(f"Readout mode: {mode}")

        except Exception as e:
            QMessageBox.critical(self, "Readout mode error", str(e))
    
    def update_geometry_ui(self, mode):
        self.single_box.setVisible(mode == "single_track")
        self.multi_box.setVisible(mode == "multi_track")
        self.roi_box.setVisible(mode == "image")

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
