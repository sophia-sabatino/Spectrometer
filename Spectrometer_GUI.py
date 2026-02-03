import sys 
from PyQt6.QtCore import pyqtSignal
import numpy as np 
from numpy.random import noncentral_chisquare
import pyqtgraph as pg
pg.setConfigOptions(antialias=True)
from PyQt6.QtWidgets import (
    QApplication, QWidget,  QPushButton, QVBoxLayout, QGridLayout,QLabel, 
    QLineEdit, QMessageBox, QComboBox, QGroupBox, QHBoxLayout, QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt
import matplotlib.pyplot as plt 

from Spectrometer import (
    AndorCameraController, KymeraController, SpectrometerController
)

class AcquireWorker(QThread):
    finished = pyqtSignal(object, object, object)
    error = pyqtSignal(str)

    def __init__(self, spec, laser_wl, trigger_mode="int", pixel_width=26.0):
        super().__init__()
        self.spec = spec
        self.laser_wl = laser_wl
        self.trigger_mode = trigger_mode
        self.pixel_width = pixel_width

    def run(self):
        try:
            if self.trigger_mode == "software":
                spectrum, wl, raman = self.spec.acquire_spectrum_software(self.laser_wl)
            else:
                spectrum, wl, raman = self.spec.acquire_spectrum(self.laser_wl)
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
        self.temp_setpoint.setValue(-80)
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

        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        controls_layout = QVBoxLayout()
        controls_layout.addWidget(cooling_box)
        controls_layout.addWidget(self.temp_status_label)

        fan_box = QGroupBox("Fan")
        fan_layout = QHBoxLayout()
        self.fan_combo = QComboBox()
        self.fan_combo.addItems(["full", "low", "off"])
        
        fan_layout.addWidget(QLabel("Mode:"))
        fan_layout.addWidget(self.fan_combo)

        fan_box.setLayout(fan_layout)
        controls_layout.addWidget(fan_box)

        trigger_box = QGroupBox("Trigger")
        trigger_layout = QHBoxLayout()

        self.trigger_combo = QComboBox()
        self.trigger_combo.addItems(["Internal", "Software"])

        trigger_layout.addWidget(QLabel("Mode:"))
        trigger_layout.addWidget(self.trigger_combo)
        trigger_box.setLayout(trigger_layout)

        controls_layout.addWidget(trigger_box)

        grating_box = QGroupBox("Grating")
        grating_layout = QHBoxLayout()
        self.grating_combo = QComboBox()

        try:
            gratings = self.kymera.list_gratings()
        except:
            gratings = ["Grating 0", "Grating 1", "Grating 2"]
        
        for i, g in enumerate(gratings):
            self.grating_combo.addItem(f"{i}: {g}")
        
        grating_layout.addWidget(QLabel("Select:"))
        grating_layout.addWidget(self.grating_combo)
        self.grating_combo.setCurrentIndex(0)
        grating_box.setLayout(grating_layout)
        controls_layout.addWidget(grating_box)

        slit_box = QGroupBox("Entrance slit")
        slit_layout = QHBoxLayout()

        self.slit_spin = QDoubleSpinBox()
        self.slit_spin.setRange(5, 300)
        self.slit_spin.setSingleStep(5)
        self.slit_spin.setValue(50)
        self.slit_spin.setSuffix("µm")
        self.set_slit_btn = QPushButton("Set")
        slit_layout.addWidget(QLabel("Width:"))
        slit_layout.addWidget(self.slit_spin)
        slit_layout.addWidget(self.set_slit_btn)
        slit_box.setLayout(slit_layout)
        controls_layout.addWidget(slit_box)

        pixel_width_box = QGroupBox("Pixel width")
        pixel_width_layout = QHBoxLayout()
        self.pixel_width_display = QLineEdit()
        self.pixel_width_display.setReadOnly(True)
        self.pixel_width_display.setAlignment(Qt.AlignmentFlag.AlignCenter)

        pixel_width_layout.addWidget(QLabel("Pixel Width (m):"))
        pixel_width_layout.addWidget(self.pixel_width_display)
        pixel_width_box.setLayout(pixel_width_layout)
        controls_layout.addWidget(pixel_width_box)

        exposure_box = QGroupBox("Exposure time")
        exposure_layout = QHBoxLayout()
        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setRange(0.001, 1000)
        self.exposure_spin.setValue(0.1)
        self.exposure_spin.setSuffix("s")
        self.set_exposure_btn = QPushButton("Set")
        exposure_layout.addWidget(QLabel("Exposure time (s):"))
        exposure_layout.addWidget(self.exposure_spin)
        exposure_layout.addWidget(self.set_exposure_btn)
        exposure_box.setLayout(exposure_layout)
        controls_layout.addWidget(exposure_box)

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
        controls_layout.addWidget(self.roi_box)

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
        controls_layout.addWidget(self.single_box)

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
        controls_layout.addWidget(self.multi_box)

        geom_box = QGroupBox("Readout mode")
        geom_layout = QHBoxLayout()
        self.geom_combo = QComboBox()
        #eventuall add continuous mode warning on
        self.geom_combo.addItems(["fvb", "image", "single_track", "multi_track"])

        geom_layout.addWidget(QLabel("Mode:"))
        geom_layout.addWidget(self.geom_combo)
        geom_box.setLayout(geom_layout)
        controls_layout.addWidget(geom_box)

        acq_box = QGroupBox("Aquisition Mode")
        acq_layout = QVBoxLayout()
        self.acq_mode_combo = QComboBox()
        self.acq_mode_combo.addItems(["single", "accumulate", "kinetic", "continuous"])

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        mode_layout.addWidget(self.acq_mode_combo)
        acq_box.setLayout(mode_layout)

        self.kinetics_params_box = QGroupBox("Kinetics Parameters")
        kinetic_layout = QHBoxLayout()
        self.kinetic_frames_spin = QSpinBox()
        self.kinetic_frames_spin.setRange(1, 1000)
        self.kinetic_frames_spin.setValue(10)
        self.kinetic_cycle_spin = QDoubleSpinBox()
        self.kinetic_cycle_spin.setRange(0.001, 1000)
        self.kinetic_cycle_spin.setDecimals(3)
        self.kinetic_cycle_spin.setValue(0.1)
        kinetic_layout.addWidget(QLabel("Frames:"))
        kinetic_layout.addWidget(self.kinetic_frames_spin)
        kinetic_layout.addWidget(QLabel("Cycle time (s):"))
        kinetic_layout.addWidget(self.kinetic_cycle_spin)
        self.kinetics_params_box.setLayout(kinetic_layout)
        acq_layout.addWidget(self.kinetics_params_box)

        self.accum_params_box = QGroupBox("Accumulate Parameters")
        accum_layout = QHBoxLayout()
        self.accum_num_spin = QSpinBox()
        self.accum_num_spin.setRange(1, 1000)
        self.accum_num_spin.setValue(5)
        self.accum_cycle_spin = QDoubleSpinBox()
        self.accum_cycle_spin.setRange(0.001, 1000)
        self.accum_cycle_spin.setDecimals(3)
        self.accum_cycle_spin.setValue(0.1)
        accum_layout.addWidget(QLabel("Accumulations:"))
        accum_layout.addWidget(self.accum_num_spin)
        accum_layout.addWidget(QLabel("Cycle time (s):"))
        accum_layout.addWidget(self.accum_cycle_spin)
        self.accum_params_box.setLayout(accum_layout)
        acq_layout.addWidget(self.accum_params_box)

        self.cont_params_box = QGroupBox("Continuous Parameters")
        cont_layout = QHBoxLayout()
        self.cont_cycle_spin = QDoubleSpinBox()
        self.cont_cycle_spin.setRange(0.001, 1000)
        self.cont_cycle_spin.setDecimals(3)
        self.cont_cycle_spin.setValue(0.1)
        cont_layout.addWidget(QLabel("Cycle time (s):"))
        cont_layout.addWidget(self.cont_cycle_spin)
        self.cont_params_box.setLayout(cont_layout)
        acq_layout.addWidget(self.cont_params_box)

        acq_box.setLayout(acq_layout)
        controls_layout.addWidget(acq_box)


        wl_box = QGroupBox("Central wavelength")
        wl_layout = QHBoxLayout()
        self.center_wl_spin = QDoubleSpinBox()
        self.center_wl_spin.setRange(200, 1200)
        self.center_wl_spin.setDecimals(2)
        self.center_wl_spin.setValue(600)
        self.center_wl_spin.setSuffix("nm")
        self.set_wl_btn = QPushButton("Set")
        wl_layout.addWidget(QLabel("Wavelength:"))
        wl_layout.addWidget(self.center_wl_spin)
        wl_layout.addWidget(self.set_wl_btn)
        wl_box.setLayout(wl_layout)
        controls_layout.addWidget(wl_box)

        plot_layout = QVBoxLayout()
        plot_layout.addWidget(QLabel("Laser wavelength (nm):"))
        plot_layout.addWidget(self.laser_edit)
        plot_layout.addWidget(self.connect_btn)
        plot_layout.addWidget(self.acquire_btn)
        plot_layout.addWidget(self.status_label)
        plot_layout.addWidget(self.xaxis_combo)
        plot_layout.addWidget(self.plot_widget, stretch=1)

        main_layout.addLayout(controls_layout, stretch=0)
        main_layout.addLayout(plot_layout, stretch=1)

        #self.setLayout(main_layout)
        self.connect_btn.clicked.connect(self.connect_devices)
        self.acquire_btn.clicked.connect(self.acquire)
        self.cooler_on.clicked.connect(self.enable_cooling)
        self.cooler_off.clicked.connect(self.disable_cooling)
        self.apply_roi_btn.clicked.connect(self.apply_roi)
        self.fan_combo.currentTextChanged.connect(self.set_fan_mode)
        self.geom_combo.currentTextChanged.connect(self.set_geometry_mode)
        self.geom_combo.currentTextChanged.connect(self.update_geometry_ui)
        self.trigger_combo.currentTextChanged.connect(self.set_trigger_mode)
        self.grating_combo.currentIndexChanged.connect(self.set_grating_from_gui)
        self.set_wl_btn.clicked.connect(self.set_central_wavelength_from_gui)
        self.set_slit_btn.clicked.connect(self.set_slit_from_gui)
        self.acq_mode_combo.currentTextChanged.connect(self.update_acquisition_ui)
        self.set_exposure_btn.clicked.connect(self.set_exposure_from_gui)

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

            current_grating = self.kymera.get_grating()
            self.grating_combo.setCurrentIndex(current_grating-1)

            current_wl = self.kymera.get_central_wavelength()
            self.center_wl_spin.setValue(current_wl)

            current_slit = self.kymera.get_slit_width_um()
            self.slit_spin.setValue(current_slit)
            self.pixel_width_display.setText(f"{self.kymera.get_acq_pixel_width():.2f}")

            current_exp = self.cam.get_exposure()
            self.exposure_spin.setValue(current_exp)

            self.cam.set_readout_mode("fvb")
            self.geom_combo.setCurrentText("fvb")
            self.update_geometry_ui("fvb")
            
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
    
    #getting an error here 
    def set_grating_from_gui(self, index):
        if not self.cam.connected:
            return
        
        try:
            index = int(index)
            self.kymera.set_grating(index+1)
            self.kymera.setup_from_camera(self.cam.cam)
            self.status_label.setText(f"Grating: {index}")
        
        except Exception as e:
            QMessageBox.critical(self, "Grating error", str(e))
    
    def set_slit_from_gui(self):
        if not self.cam.connected:
            return
        
        try:
            width = float(self.slit_spin.value())
            self.kymera.set_slit_width_um(width)
            self.kymera.setup_from_camera(self.cam.cam)
            self.status_label.setText(f"Slit = {width:.1f} µm")
            if self.last_spectrum is not None:
                self.update_plot()
        
        except Exception as e:
            QMessageBox.critical(self, "Slit error", str(e))
    
    def set_exposure_from_gui(self):
        if not self.cam.connected:
            return
        try:
            exp = self.exposure_spin.value()
            self.cam.set_exposure(exp)
            self.status_label.setText(f"Exposure time: {exp:.3f} s")
        except Exception as e:
            QMessageBox.critical(self, "Exposure error", str(e))

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
    
    def update_acquisition_ui(self, mode):
        self.kinetics_params_box.setVisible(mode == "kinetic")
        self.accum_params_box.setVisible(mode == "accumulate")
        self.cont_params_box.setVisible(mode == "continuous")
    
    def set_central_wavelength_from_gui(self):
        if not self.cam.connected: 
            return
        
        try:
            wl = self.center_wl_spin.value()
            self.kymera.set_central_wavelength(wl*10e-9)
            self.kymera.setup_from_camera(self.cam.cam)
            self.status_label.setText(f"Central wavelength: {wl:.2f} nm")
            if self.last_spectrum is not None:
                self.update_plot()
        
        except Exception as e:
            QMessageBox.critical(self, "Wavelength error", str(e))

    def acquire(self):
        try:
            self.set_connected(False)
            laser_wl = float(self.laser_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Input error", "Laser wavelength must be a number")
            return 
    
        trigger_text = self.trigger_combo.currentText()
        trigger_mode = "software" if trigger_text == "Software" else "int"

        if self.acq_mode_combo.currentText() == "single":
            self.cam.set_acquisition_mode("single")
        elif self.acq_mode_combo.currentText() == "accumulate":
            self.cam.set_acquisition_mode("accum")
        elif self.acq_mode_combo.currentText() == "kinetic":
            self.cam.set_acquisition_mode("kinetic")
        elif self.acq_mode_combo.currentText() == "continuous":
            self.cam.set_acquisition_mode("cont")
        else:
            QMessageBox.critical(self, "Acquisition mode error", "Invalid acquisition mode")
            return
        
        if self.acq_mode_combo.currentText() == "kinetic":
            frames = self.kinetic_frames_spin.value()
            cycle = self.kinetic_cycle_spin.value()
            self.cam.setup_kinetic_mode(num_frames=frames, cycle_time=cycle)
        elif self.acq_mode_combo.currentText() == "accumulate":
            num_accum = self.accum_num_spin.value()
            cycle = self.accum_cycle_spin.value()
            self.cam.setup_accum_mode(num_accum=num_accum, cycle_time=cycle)
        elif self.acq_mode_combo.currentText() == "continuous":
            cycle = self.cont_cycle_spin.value()
            self.cam.setup_cont_mode(cycle_time=cycle)

        self.status_label.setText("Acquiring...")
        pixel_width = float(self.pixel_width_display.text())
        self.worker = AcquireWorker(self.spec, laser_wl, trigger_mode=trigger_mode, pixel_width=pixel_width)
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
