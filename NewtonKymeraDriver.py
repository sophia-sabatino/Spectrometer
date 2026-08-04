import numpy as np
import pylablib as pll

pll.par["devices/dlls/andor_sdk2"] = r"C:\Program Files\Andor SOLIS"
pll.par["devices/dlls/andor_shamrock"] = r"C:\Program Files\Andor SOLIS"

from pylablib.devices import Andor
from pybirch.Instruments.base import BaseMeasurementInstrument
import time
import threading

class NewtonKymeraDriver(BaseMeasurementInstrument):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hardware_lock = threading.RLock()

        self.cam = None
        self.spec = None

        # Local settings are stored even before the hardware is connected.
        self._readout_mode = "fvb"
        self._temperature_setpoint = -80.0
        self._cooler_enabled = True
        self._fan_mode = "full"
        self._exposure = 0.05
        self._grating = 1
        self._center_wavelength_nm = 633.0
        self._acquisition_mode = "single"
        self._num_accumulations = 1
        self._accumulation_cycle_time = 0.0

        self._wavelength_nm = np.array([], dtype=float)

        self.data_columns = []
        self.data_units = []

    def _connect_impl(self):
        try:
            print("Opening Newton camera...")

            self.cam = Andor.AndorSDK2Camera(idx=0)

            print("Camera opened")
            print("Camera information:", self.cam.get_device_info())
            print("Detector size:", self.cam.get_detector_size())
            print("Pixel size:", self.cam.get_pixel_size())

            print("Opening Kymera spectrograph...")

            self.spec = Andor.ShamrockSpectrograph(idx=0)

            print("Spectrograph opened")
            print("Initial grating:", self.spec.get_grating())
            print(
                "Initial center wavelength:",
                self.spec.get_wavelength() * 1e9,
                "nm",
            )

            self._configure_camera()
            self._configure_spectrograph()
            self._update_calibration()

            print("Newton/Kymera initialization complete")

        except Exception:
            # Prevent SDK resources from remaining locked when connection
            # fails partway through.
            self._close_hardware()
            raise

    def _configure_camera(self):
        if self.cam is None:
            raise RuntimeError("Camera is not connected")

        self.cam.set_fan_mode("full")
        self.cam.set_temperature(-80, enable_cooler=True)

        # Internal trigger and single acquisition are the simplest starting
        # configuration.
        self.cam.set_trigger_mode("int")
        self.cam.set_acquisition_mode("single")
        self.cam.set_exposure(self._exposure)

        # Full vertical binning produces one spectrum row.
        self.cam.set_read_mode("fvb")

        print("Fan mode:", self.cam.get_fan_mode())
        print("Cooler enabled:", self.cam.is_cooler_on())
        print("Temperature setpoint: -80 °C")
        print("Current temperature:", self.cam.get_temperature(), "°C")
        print("Temperature status:", self.cam.get_temperature_status())
        print("Camera exposure:", self.cam.get_exposure(), "s")
        print("Camera read mode:", self.cam.get_read_mode())
        print("Camera acquisition mode:", self.cam.get_acquisition_mode())

    def _configure_spectrograph(self):
        if self.spec is None:
            raise RuntimeError("Spectrograph is not connected")

        self.spec.set_grating(self._grating)

        # PyLabLib spectrograph wavelengths are in meters.
        self.spec.set_wavelength(
            self._center_wavelength_nm * 1e-9
        )

        print("Selected grating:", self.spec.get_grating())
        print(
            "Center wavelength:",
            self.spec.get_wavelength() * 1e9,
            "nm",
        )

        # Supply the Newton detector geometry to the Kymera calibration
        # routines.
        self.spec.setup_pixels_from_camera(self.cam)

        print(
            "Configured detector pixels:",
            self.spec.get_number_pixels(),
        )
        print(
            "Configured pixel width:",
            self.spec.get_pixel_width(),
            "m",
        )


    def _update_calibration(self):
        if self.cam is None or self.spec is None:
            raise RuntimeError(
                "Camera and spectrograph must both be connected"
            )

        # Repeat this step before calibration in case the camera geometry
        # has changed.
        self.spec.setup_pixels_from_camera(self.cam)

        wavelength_m = np.asarray(
            self.spec.get_calibration(),
            dtype=float,
        )

        if wavelength_m.ndim != 1 or wavelength_m.size == 0:
            raise RuntimeError(
                "The Kymera returned an invalid wavelength calibration"
            )

        self._wavelength_nm = wavelength_m * 1e9

        # Each wavelength pixel becomes one PyBirch output column.
        #
        # String labels are usually safer than raw floating-point column
        # names.
        self.data_columns = [
            f"{wavelength:.4f}"
            for wavelength in self._wavelength_nm
        ]

        self.data_units = [
            "counts"
            for _ in self.data_columns
        ]

        print(
            "Calibration:",
            len(self._wavelength_nm),
            "pixels from",
            f"{self._wavelength_nm[0]:.3f}",
            "to",
            f"{self._wavelength_nm[-1]:.3f}",
            "nm",
        )

        peak_pixel_test = len(self._wavelength_nm) // 2

        print("Calibration length:", len(self._wavelength_nm))
        print(
            "Wavelength range:",
            self._wavelength_nm[0],
            "to",
            self._wavelength_nm[-1],
            "nm",
        )
        print(
            "Middle pixel wavelength:",
            self._wavelength_nm[peak_pixel_test],
            "nm",
        )

    def _single_timeout(self):
        exposure = float(self.cam.get_exposure())
        return max(exposure + 5.0, 10.0)

    def _configure_acquisition_mode(self):
        if self._acquisition_mode not in ("single", "accum"):
            raise ValueError(
                "Acquisition mode must be 'single' or 'accum'"
            )

        # This method now only validates/reports the logical PyBirch mode.
        # It does not reconfigure the camera.
        print("PyBirch logical mode:", self._acquisition_mode)
        print(
            "Software accumulation frames:",
            1 if self._acquisition_mode == "single"
            else self._num_accumulations,
        )

    def _acquire_one_raw(self):
        if self.cam is None:
            raise RuntimeError("Camera is not connected")

        with self._hardware_lock:
            exposure = float(self.cam.get_exposure())
            timeout = max(exposure + 5.0, 10.0)

            print(
                f"Calling snap: exposure={exposure:.4f} s, "
                f"timeout={timeout:.2f} s"
            )

            frame = self.cam.snap(timeout=timeout)

            return np.asarray(frame, dtype=np.float64)

    def _acquire_raw(self):
        if self._acquisition_mode == "single":
            num_frames = 1
        elif self._acquisition_mode == "accum":
            num_frames = int(self._num_accumulations)
        else:
            raise RuntimeError(
                f"Unsupported acquisition mode: "
                f"{self._acquisition_mode!r}"
            )

        if num_frames < 1:
            raise ValueError(
                "Number of accumulation frames must be at least 1"
            )

        with self._hardware_lock:
            accumulated = None

            for index in range(num_frames):
                exposure = float(self.cam.get_exposure())
                timeout = max(exposure + 5.0, 10.0)

                frame = np.asarray(
                    self.cam.snap(timeout=timeout),
                    dtype=np.float64,
                )

                if accumulated is None:
                    accumulated = np.zeros_like(
                        frame,
                        dtype=np.float64,
                    )

                if frame.shape != accumulated.shape:
                    raise RuntimeError(
                        "Frame shape changed during accumulation: "
                        f"expected {accumulated.shape}, "
                        f"got {frame.shape}"
                    )

                accumulated += frame

                print(
                    f"Acquired frame {index + 1}/{num_frames}: "
                    f"min={frame.min():.1f}, "
                    f"max={frame.max():.1f}, "
                    f"mean={frame.mean():.1f}"
                )

                if (
                    index < num_frames - 1
                    and self._accumulation_cycle_time > 0
                ):
                    time.sleep(
                        self._accumulation_cycle_time
                    )

            return accumulated

    def _perform_measurement_impl(self):
        if self.cam is None or self.spec is None:
            raise RuntimeError(
                "Camera and spectrograph are not connected"
            )

        image = self._acquire_raw()
        image = np.squeeze(image)

        if image.ndim == 1:
            spectrum = image

        elif image.ndim == 2:
            spectrum = image.sum(axis=0)

        else:
            raise RuntimeError(
                f"Unexpected image shape: {image.shape}"
            )

        spectrum = np.asarray(
            spectrum,
            dtype=np.float64,
        ).ravel()

        if spectrum.size != self._wavelength_nm.size:
            raise RuntimeError(
                "Spectrum/calibration length mismatch: "
                f"spectrum={spectrum.size}, "
                f"calibration={self._wavelength_nm.size}"
            )

        return np.array([spectrum])

    @property
    def settings(self):
        values = {
            "exposure": self._exposure,
            "acquisition_mode": self._acquisition_mode,
            "num_accumulations": self._num_accumulations,
            "accumulation_cycle_time": self._accumulation_cycle_time,
            "readout_mode": self._readout_mode,
            "grating": self._grating,
            "center_wavelength": self._center_wavelength_nm,
            "temperature_setpoint": self._temperature_setpoint,
            "current_temperature": None,
            "cooler_enabled": self._cooler_enabled,
            "fan_mode": self._fan_mode,
        }

        if self.cam is not None:
            with self._hardware_lock:
                values["exposure"] = float(
                    self.cam.get_exposure()
                )
                values["readout_mode"] = (
                    self.cam.get_read_mode()
                )
                values["current_temperature"] = float(
                    self.cam.get_temperature()
                )
                values["cooler_enabled"] = bool(
                    self.cam.is_cooler_on()
                )
                values["fan_mode"] = (
                    self.cam.get_fan_mode()
                )

            # Do not do this:
            # values["acquisition_mode"] = self.cam.get_acquisition_mode()

        if self.spec is not None:
            values["grating"] = int(self.spec.get_grating())
            values["center_wavelength"] = float(
                self.spec.get_wavelength() * 1e9
            )

        return values

    @settings.setter
    def settings(self, values):
        if not isinstance(values, dict):
            raise TypeError("Settings must be supplied as a dictionary")

        calibration_changed = False

        with self._hardware_lock:
            # -------------------------------------------------------------
            # Exposure
            # -------------------------------------------------------------
            if "exposure" in values:
                exposure = float(values["exposure"])

                if exposure <= 0:
                    raise ValueError(
                        "Exposure must be greater than zero"
                    )

                if exposure != self._exposure:
                    self._exposure = exposure

                    if self.cam is not None:
                        self.cam.set_exposure(exposure)

                        actual = float(self.cam.get_exposure())
                        print(
                            "Exposure changed:",
                            f"requested={exposure}, actual={actual}"
                        )

            # -------------------------------------------------------------
            # Readout
            # -------------------------------------------------------------
            if "readout_mode" in values:
                mode = str(values["readout_mode"])

                if mode not in ("fvb", "image"):
                    raise ValueError(
                        "Readout mode must be 'fvb' or 'image'"
                    )

                if mode != self._readout_mode:
                    self._readout_mode = mode

                    if self.cam is not None:
                        self.cam.set_read_mode(mode)

                    calibration_changed = True

            # -------------------------------------------------------------
            # Cooling
            # -------------------------------------------------------------
            if "temperature_setpoint" in values:
                temperature = float(values["temperature_setpoint"])

                if temperature != self._temperature_setpoint:
                    self._temperature_setpoint = temperature

                    if self.cam is not None:
                        self.cam.set_temperature(
                            temperature,
                            enable_cooler=self._cooler_enabled,
                        )

            if "cooler_enabled" in values:
                enabled = bool(values["cooler_enabled"])

                if enabled != self._cooler_enabled:
                    self._cooler_enabled = enabled

                    if self.cam is not None:
                        self.cam.set_cooler(enabled)

            if "fan_mode" in values:
                mode = str(values["fan_mode"])

                if mode not in ("full", "low", "off"):
                    raise ValueError(
                        "Fan mode must be 'full', 'low', or 'off'"
                    )

                if mode != self._fan_mode:
                    self._fan_mode = mode

                    if self.cam is not None:
                        self.cam.set_fan_mode(mode)

            # -------------------------------------------------------------
            # Spectrograph
            # -------------------------------------------------------------
            if "grating" in values:
                grating = int(values["grating"])

                if grating != self._grating:
                    self._grating = grating

                    if self.spec is not None:
                        self.spec.set_grating(grating)

                    calibration_changed = True

            if "center_wavelength" in values:
                wavelength_nm = float(values["center_wavelength"])

                if wavelength_nm <= 0:
                    raise ValueError(
                        "Center wavelength must be greater than zero"
                    )

                if wavelength_nm != self._center_wavelength_nm:
                    self._center_wavelength_nm = wavelength_nm

                    if self.spec is not None:
                        self.spec.set_wavelength(
                            wavelength_nm * 1e-9
                        )

                    calibration_changed = True

            # -------------------------------------------------------------
            # Logical software accumulation
            # -------------------------------------------------------------
            if "acquisition_mode" in values:
                mode = str(values["acquisition_mode"])

                if mode not in ("single", "accum"):
                    raise ValueError(
                        "Acquisition mode must be 'single' or 'accum'"
                    )

                self._acquisition_mode = mode

            if "num_accumulations" in values:
                num_acc = int(values["num_accumulations"])

                if num_acc < 1:
                    raise ValueError(
                        "Number of accumulations must be at least 1"
                    )

                self._num_accumulations = num_acc

            if "accumulation_cycle_time" in values:
                cycle_time = float(
                    values["accumulation_cycle_time"]
                )

                if cycle_time < 0:
                    raise ValueError(
                        "Accumulation cycle time cannot be negative"
                    )

                self._accumulation_cycle_time = cycle_time

            if calibration_changed:
                self._update_calibration()

        self._configure_acquisition_mode()

    def get_status(self):
        if self.cam is None:
            return {
                "connected": False,
                "temperature": None,
                "temperature_status": None,
                "cooler": False,
                "fan_mode": None,
            }

        with self._hardware_lock:
            return {
                "connected": True,
                "temperature": self.cam.get_temperature(),
                "temperature_status":
                    self.cam.get_temperature_status(),
                "cooler": self.cam.is_cooler_on(),
                "fan_mode": self.cam.get_fan_mode(),
                "grating": (
                    self.spec.get_grating()
                    if self.spec is not None
                    else None
                ),
                "center_wavelength_nm": (
                    self.spec.get_wavelength() * 1e9
                    if self.spec is not None
                    else None
                ),
            }
        
    def estimate_measure_time(self):
        exposure = self._exposure

        if self.cam is not None:
            try:
                exposure = float(self.cam.get_exposure())
            except Exception:
                pass

        frames = (
            1
            if self._acquisition_mode == "single"
            else self._num_accumulations
        )

        delays = max(frames - 1, 0)

        return (
            frames * (exposure + 2.0)
            + delays * self._accumulation_cycle_time
        )

    def _close_hardware(self):
        # Close the Kymera first because it communicates through the Newton.
        if self.spec is not None:
            try:
                self.spec.close()
            except Exception as exc:
                print("Error closing spectrograph:", exc)
            finally:
                self.spec = None

        if self.cam is not None:
            try:
                self.cam.close()
            except Exception as exc:
                print("Error closing camera:", exc)
            finally:
                self.cam = None

    def _disconnect_impl(self):
        self._close_hardware()
        print("Newton and Kymera closed")