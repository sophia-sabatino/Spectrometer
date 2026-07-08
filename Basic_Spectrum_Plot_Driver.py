import numpy as np
from pylablib.devices import Andor
from pylablib.devices.Andor import Shamrock
from pybirch.Instruments.base import BaseMeasurementInstrument
import matplotlib.pyplot as plt


settings_schema = {
    "type": "object",
    "properties": {
        "exposure": {
            "type": "number",
            "title": "Exposure (s)",
            "default": 0.1,
            "minimum": 0.001,
        },
        "center_wavelength": {
            "type": "number",
            "title": "Center Wavelength (nm)",
            "default": 500.0,
        },
    },
}


class AndorSpectrometerDriver(BaseMeasurementInstrument):

    display_mode = "spectrum"
    settings_schema = settings_schema

    def __init__(self, name="Andor Spectrometer"):
        super().__init__(name)

        self.cam = None
        self.spec = None

    ####################################################################
    # Connection
    ####################################################################

    def _connect_impl(self):

        self.cam = Andor.AndorSDK2Camera()
        self.spec = Shamrock.ShamrockSpectrograph()

        self.spec.setup_pixels_from_camera(self.cam)

        return True

    ####################################################################
    # Initialization
    ####################################################################

    def _initialize_impl(self):

        wl = self.spec.get_calibration() * 1e9

        self.data_columns = np.array(
            [f"{w:.3f}" for w in wl]
        )

        self.data_units = np.array(
            ["counts"] * len(wl)
        )

    ####################################################################
    # Settings
    ####################################################################

    @property
    def settings(self):
        return {
            "exposure": self.cam.get_exposure(),
            "center_wavelength": self.spec.get_wavelength() * 1e9,
        }

    @settings.setter
    def settings(self, settings):

        if "exposure" in settings:
            self.cam.set_exposure(
                float(settings["exposure"])
            )

        if "center_wavelength" in settings:
            self.spec.set_wavelength(
                float(settings["center_wavelength"]) * 1e-9
            )

            wl = self.spec.get_calibration() * 1e9

            self.data_columns = np.array(
                [f"{w:.3f}" for w in wl]
            )

    ####################################################################
    # Measurement
    ####################################################################

    def _perform_measurement_impl(self):

        image = self.cam.snap()

        spectrum = np.asarray(image).sum(axis=0)

        wl = self.spec.get_calibration() * 1e9

        plt.figure(figsize=(8,5))
        plt.plot(wl, spectrum)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Counts")
        plt.title("Single Acquisition Spectrum")
        plt.grid(True)
        plt.show()

        return np.array([spectrum])

    ####################################################################
    # Shutdown
    ####################################################################

    def _shutdown_impl(self):

        if self.cam is not None:
            self.cam.close()

        if self.spec is not None:
            self.spec.close()

        super()._shutdown_impl()