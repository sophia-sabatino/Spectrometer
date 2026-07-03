import numpy as np
from pylablib.devices import Andor
from pylablib.devices.Andor import Shamrock
from pylablib.devices.Andor import AndorSDK2Camera
from pybirch.Instruments.base import BaseMeasurementInstrument 

class AndorSpectrometerDriver(BaseMeasurementInstrument):
    display_mode = "spectrum"

    def __init__(self, name="Andor Spectrometer"):
        super().__init__(name)
        
        self.camera = AndorCameraController()
        self.kymera = KymeraController()

        self.data_columns = np.array([])
        self.data_units = np.array([])
    
    def _connect_impl(self):
        try:
            self.camera.connect()
            self.kymera.setup_from_camera(self.camera.cam)
            return True
        except Exception:
            return False 
    
    def _initialize_impl(self):
        wl = self.kymera.get_calibration_nm()

        self.data_columns = np.array([f"{x:.3f}" for x in wl])
        self.data_units = np.array(["counts"] * len(wl))
    
    def _perform_measurement_impl(self):
        image = self.camera.acquire_single()
        spectrum = image.mean(axis=0)
        return np.array([spectrum])
    
    @property
    def settings(self):
        return {
            "exposure": self.camera.get_exposure(),
            "grating": self.kymera.get_grating(),
            "center_wavelength": self.kymera.get_central_wavelength(),
            "temperature_setpoint": self.camera.temperature_setpoint,
        }
    
    @settings.setter
    def settings(self, settings):
        if "exposure" in settings:
            self.camera.set_exposure(float(settings["exposure"]))
        if "grating" in settings:
            self.kymera.set_grating(settings["grating"])
        if "center_wavelength" in settings:
            self.kymera.set_central_wavelength(float(settings["center_wavelength"]))
        if "temperature_setpoint" in settings:
            self.camera.set_temp(float(settings["temperature_setpoint"]))


    def _shutdown_impl(self):
        try:
            self.camera.shutdown()
            self.kymera.disconnect()
        finally:
            super()._shutdown_impl()
