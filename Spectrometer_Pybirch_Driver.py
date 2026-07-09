import numpy as np
from pylablib.devices import Andor
from pylablib.devices.Andor import Shamrock
from pylablib.devices.Andor import AndorSDK2Camera
from pybirch.Instruments.base import BaseMeasurementInstrument 
import matplotlib.pyplot as plt
import threading

settings_schema = {
    "type": "object",
    "x-ui-sections": [
        {"name": "acquisition", "label": "Acquisition"},
        {"name": "spectrograph", "label": "Spectrograph"},
        {"name": "cooling", "label": "Cooling"},
    ],
    "properties": {
        "exposure": {
            "type": "number", "title": "Exposure", "default": 0.1,
            "minimum": 0.001, "x-ui-unit": "s", "x-ui-group": "acquisition",
        },
        "grating": {
            "type": "integer", "title": "Grating", "default": 0,
            "enum": [0, 1, 2], "x-ui-group": "spectrograph",
        },
        "center_wavelength": {
            "type": "number", "title": "Center Wavelength", "default": 500.0,
            "x-ui-unit": "nm", "x-ui-group": "spectrograph",
        },
        "temperature_setpoint": {
            "type": "number", "title": "Temperature Setpoint", "default": -60.0,
            "x-ui-unit": "°C", "x-ui-group": "cooling",
        },
    },
    "acquisition_mode": {
        "type": "string",
        "title": "Acquisition Mode",
        "enum": [
            "single",
            "accum"
        ],
        "default": "single",
        "x-ui-group": "acquisition",
    },

    "num_accumulations": {
        "type": "integer",
        "title": "Accumulations",
        "default": 1,
        "minimum": 1,
        "x-ui-group": "acquisition",
    },

    "accumulation_cycle_time": {
        "type": "number",
        "title": "Accumulation Cycle Time",
        "default": 0.0,
        "minimum": 0.0,
        "x-ui-unit": "s",
        "x-ui-group": "acquisition",
    },
}

class AndorCameraController:
    def __init__(self):
        self.cam = None
        self.connected = False

        # Cached state (GUI mirrors this)
        self.exposure = 0.1
        self.acquisition_mode = "single"
        self.hbin = 1
        self.vbin = 1
        self.temperature_setpoint = None
        self.cooler_enabled = False
        self.trigger_mode = "internal"
        self.kinetics_frame = 1
        self.kinetis_cycle_time = None
        self.accum_number = 1
        self.accum_cycle_time = 0.0

        self._lock = threading.Lock()

    # Connection control
    def connect(self):
        "Open camera connection"
        if not self.connected:
            self.cam = Andor.AndorSDK2Camera()
            self.cam.set_fan_mode("low")
            self.connected = True

    def disconnect(self):
        "Close camera connection"
        if self.connected:
            self.cam.close()
            self.connected = False
    
    # New cooling control
    def cooler(self):
        return self.cam.is_cooler_on()
    
    def set_cooler(self, on=True):
        if on:
            self.cam.set_fan_mode("full")
        self.cam.set_cooler(on)
        self.cooler_enabled = on
    
    def get_temp_status(self):
        return self.cam.get_temperature_status()
    
    def set_temp(self, temp, enable_cooler=True):
        self.cam.set_temperature(temp, enable_cooler)
        self.temperature_setpoint = temp
        self.cooler_enabled = enable_cooler

    def get_temperature(self):
        return self.cam.get_temperature()
    
    def update_fan_auto(self, threshold=10):
        print("cooler:", self.cooler_enabled, self.temperature_setpoint)
        if not self.cooler_enabled or self.temperature_setpoint is None:
            return
        current_temp = self.get_temperature()
        set_temp = self.temperature_setpoint
        if abs(current_temp - set_temp) < threshold:
            self.cam.set_fan_mode("full")
        else:
            self.cam.set_fan_mode("low")

    # Readout / ROI
    def set_roi(self, hbin=1, vbin=1,
                hstart=0, hend=None,
                vstart=0, vend=None):
        with self._lock:
            self.cam.set_roi(
                hstart=hstart, hend=hend,
                vstart=vstart, vend=vend,
                hbin=hbin, vbin=vbin
            )
            self.hbin = hbin
            self.vbin = vbin
    
    def get_fan_mode(self):
        return self.cam.get_fan_mode()
    
    def set_fan_mode(self, mode):
        if mode not in ["full", "low", "off"]:
            raise ValueError("Fan must must be 'full', 'low', or 'off'")
        with self._lock:
            self.cam.set_fan_mode(mode)
    
    def get_readout_mode(self):
        return self.cam.get_read_mode()
    
    def set_readout_mode(self, mode):
        if mode not in ["fvb", "single_track", "multi_track", "image", "cont"]:
            raise ValueError("Incorrect readout mode")
        with self._lock:
            self.cam.set_read_mode(mode)
        
    def setup_single_mode(self, center=0, width=1):
        self.cam.setup_single_track_mode(center, width)
    
    def get_single_mode_parameters(self):
        return self.cam.get_single_track_mode_parameters()
    
    def setup_multi_mode(self, number=1, height=1, offset=0):
        self.cam.setup_multi_track_mode(number, height, offset)
    
    def get_multi_mode_parameters(self):
        return self.cam.get_multi_track_mode_parameters()
    
    def setup_image_mode(self, hstart=0, hend=None, vstart=0, vend=None, hbin=1, vbin=1):
        self.cam.setup_image_mode(hstart, hend, vstart, vend, hbin, vbin)
    
    def get_image_mode_parameters(self):
        return self.cam.get_image_mode_parameters()
    
    def set_fvb(self):
        with self._lock:
            self.cam.set_roi(vbin="full")
            self.vbin = "full"
    
    def get_all_vsspeeds(self):
        return self.cam.get_all_vsspeeds()
    
    def set_vsspeed(self, speed):
        self.cam.set_vsspeed(speed)
    
    def get_max_vsspeed(self):
        return self.cam.get_max_vsspeed()
    
    #Trigger control
    def get_trigger_mode(self):
        return self.cam.get_trigger_mode()
    
    def set_trigger_mode(self, mode="int"):
        if mode not in ["int", "software"]:
            raise ValueError(
                "Trigger mode must be 'int', 'software'"
            )
        
        with self._lock:
            self.cam.set_trigger_mode(mode)
            self.trigger_mode = mode
    
    def set_internal_trigger(self):
        self.set_trigger_mode("int")
    
    def set_software_trigger(self):
        self.set_trigger_mode("software")
    
    # Acquisition settings
    def set_exposure(self, exposure):
        with self._lock:
            self.cam.set_exposure(exposure)
            self.exposure = exposure
    
    def get_exposure(self):
        return self.cam.get_exposure()
    
    def get_acquisition_mode(self):
        return self.cam.get_acquisition_mode()

    def set_acquisition_mode(self, mode="single"):
        with self._lock:
            self.cam.set_acquisition_mode(mode)
            self.acquisition_mode = mode
    
    def start_acquisition(self):
        self.cam.start_acquisition()
    
    def stop_acquisition(self):
        self.cam.stop_acquisition()
    
    def get_newest_image(self):
        return self.cam.read_newest_image()
    
    def setup_accum_mode(self, num_acc, cycle_time_acc=0.0):
        with self._lock:
            self.cam.set_acquisition_mode("accum")
            self.cam.setup_accum_mode(num_acc, cycle_time_acc)

            self.acquisition_mode = "accum"
            self.accum_number = num_acc
            self.accum_cycle_time = cycle_time_acc
    
    def get_accum_mode_parameters(self):
        return self.cam.get_accum_mode_parameters()
    
    def setup_kinetic_mode(self, num_cycle, cycle_time=0.0, num_acc=1, cycle_time_acc=0, num_prescan=0):
        self.cam.setup_kinetic_mode(num_cycle, cycle_time, num_acc, cycle_time_acc, num_prescan)
    
    def get_kinetic_mode_parameters(self):
        return self.cam.get_kinetic_mode_parameters()
    
    def setup_cont_mode(self, cycle_time=0):
        self.cam.setup_cont_mode(cycle_time)
    
    def get_cont_mode_parameters(self):
        return self.cam.get_cont_mode_parameters()
    
        
    # Acquisition
    def abort(self):
        self.cam.clear_acquisition()

    def acquire_single(self):
        """Blocking single acquisition"""
        with self._lock:
            return self.cam.snap()
    
    def acquire_software_triggered(self, timeout=10):
        with self._lock:
            self.cam.set_trigger_mode("software")
            self.cam.start_acquisition()
            self.cam.send_software_trigger()
            self.cam.wait_for_frame(timeout=timeout)
            image = self.cam.read_newest_image()
            return image
    
    def acquire_accum(self):
        with self._lock:

            self.cam.start_acquisition()

            self.cam.wait_for_frame()

            return self.cam.read_newest_image()


    def get_status(self):
        return {
            "connected": self.connected,
            "temperature": self.get_temperature(),
            "cooler": self.cooler_enabled,
            "hbin": self.hbin,
            "vbin": self.vbin,
            "exposure": self.exposure,
            "acqusition_mode": self.acquisition_mode,
            "trigger_mode": self.trigger_mode,
        }
    
    def configure_readout(
            self,
            mode,
            **kwargs):

        if mode == "fvb":

            self.setup_fvb_mode()

        elif mode == "single_track":

            self.setup_single_track(
                kwargs["center"],
                kwargs["width"]
            )

        elif mode == "multi_track":

            self.setup_multi_track(
                kwargs["number"],
                kwargs["height"],
                kwargs["offset"]
            )

        elif mode == "image":

            self.setup_image_mode(
                hstart=kwargs.get("hstart",0),
                hend=kwargs.get("hend",None),
                vstart=kwargs.get("vstart",0),
                vend=kwargs.get("vend",None),
                hbin=kwargs.get("hbin",1),
                vbin=kwargs.get("vbin",1),
            )

        else:
            raise ValueError(f"Unknown readout mode '{mode}'")

    # Safety
    def shutdown(self):
        try:
            self.disconnect()
        except Exception:
            pass

class KymeraController:
    def __init__(self, device_index=0):
        self.spec = Shamrock.ShamrockSpectrograph(device_index)
        self._wl_cache = None
    
    def disconnect(self):
        self.spec.close()

    def setup_from_camera(self, camera):
        self.spec.setup_pixels_from_camera(camera)
        self._wl_cache = None

    def set_grating(self, index):
        self.spec.set_grating(index)
        self._wl_cache = None

    #sets central wavelength
    def set_central_wavelength(self, wl_nm):
        self.spec.set_wavelength(wl_nm * 1e-9)
        self._wl_cache = None
    
    def get_number_pixels(self):
        return self.spec.get_number_pixels()

    def get_calibration_nm(self):
        if self._wl_cache is None:
            wl_m = self.spec.get_calibration()  # meters
            self._wl_cache = wl_m * 1e9        # convert to nm
        return self._wl_cache
    
    def get_grating(self):
        return self.spec.get_grating()
    
    def get_grating_offset(self):
        return self.spec.get_grating_offset()
    
    def set_grating_offset(self, offset):
        self.spec.set_grating_offset(offset)
        self._wl_cache = None
    
    def get_central_wavelength(self):
        return self.spec.get_wavelength() * 1e9
    
    def focus_mirror_present(self):
        return self.spec.is_focus_mirror_present()
    
    def get_focus_mirror_position(self):
        return self.spec.get_focus_mirror_position()
    
    #does not work 
    def set_focus_mirror_position(self, pos):
        position = int(pos)
        self.spec.set_focus_mirror_position(position)
    
    def get_focus_mirror_max(self):
        return self.spec.get_focus_mirror_position_max()
    
    def get_slit_width_um(self, slit="input_side"):
        width_m = self.spec.get_slit_width(slit)
        return width_m * 1e6

    #fixed!! :)
    def set_slit_width_um(self, width_um, slit="input_side"):
        width_m = width_um * 1e-6
        self.spec.set_slit_width(slit, width_m)
    
    def list_gratings(self):
        try:
            info = self.spec.get_greating_info()
            return [g['name'] for g in info]
        except AttributeError:
            return [0, 1, 2]
    
    def get_wavelength_span(self):
        wl = self.get_calibration_nm()
        return wl[0], wl[-1]
    
    def get_acq_pixel_width(self):
        return self.spec.get_pixel_width() * 1e6
    
    #preset in GUI to 26 um
    def set_acq_pixel_width(self, width_um):
        self.spec.set_pixel_width(width_um)
    
    def get_status(self):
        return {
            "grating": self.get_grating(),
            "central_wavelength_nm": self.get_central_wavelength(),
            "wavelength_range_nm": self.get_wavelength_span()
        }
    
    def get_flipper(self, flipper):
        return self.spec.is_flipper_present(flipper)
    
    def get_flipper(self, flipper):
        return self.spec.get_flipper_port(flipper)
    
    def set_flipper(self, flipper, port):
        self.spec.set_flipper_port(flipper, port)

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
        mode = self.camera.acquisition_mode

        if mode == "single":
            img = self.camera.acquire_single()

        elif mode == "accum":
            img = self.camera.acquire_accum()

        else:
            raise RuntimeError(f"Unsupported acquisition mode: {mode}")

        if img is None:
            raise RuntimeError("Camera returned no image (acquisition failed)")

        spectrum = np.asarray(img).sum(axis=0)

        wl = self.kymera.get_calibration_nm()
        plt.figure(figsize=(8,5)) 
        plt.plot(wl, spectrum) 
        plt.xlabel("Wavelength (nm)") 
        plt.ylabel("Counts") 
        plt.title("Single Acquisition Spectrum") 
        plt.grid(True) 
        plt.show()

        if spectrum.shape[0] != self.data_columns.shape[0]:
            raise RuntimeError(
                f"Spectrum length ({spectrum.shape[0]}) does not match "
                f"data_columns length ({self.data_columns.shape[0]}) — check "
                f"binning/ROI vs. spectrograph pixel setup, or re-run "
                f"_initialize_impl after changing grating/center wavelength."
            )

        return np.array([spectrum])   # shape (1, N) — N = len(data_columns)
            
    def get_status(self):
        return {
            "temperature": self.camera.get_temperature(),
            "cooler": self.camera.cooler(),
        }
    
    @property
    def settings(self):
        return {
            "exposure": self.camera.get_exposure(),
            "trigger_mode": self.camera.get_trigger_mode(),
            "grating": self.kymera.get_grating(),
            "center_wavelength": self.kymera.get_central_wavelength(),
            "readout_mode": self.camera.get_readout_mode(),
            "temperature_setpoint": self.camera.temperature_setpoint,
            "input_port": self.kymera.get_flipper("input"),
            "output_port": self.kymera.get_flipper("output"),
            "readout_mode": self.camera.get_readout_mode(),
            "acquisition_mode": self.camera.acquisition_mode,
            "num_accumulations": self.camera.accum_number,
            "accumulation_cycle_time": self.camera.accum_cycle_time,
        }
    
    @settings.setter
    def settings(self, settings):
        if "exposure" in settings:
            self.camera.set_exposure(float(settings["exposure"]))
        if "trigger_mode" in settings:
            self.camera.set_trigger_mode(settings["trigger_mode"])
        if "grating" in settings:
            self.kymera.set_grating(settings["grating"])
        if "center_wavelength" in settings:
            self.kymera.set_central_wavelength(float(settings["center_wavelength"]))
        if "readout_mode" in settings:
            self.camera.set_readout_mode(settings["readout_mode"])
        if "temperature_setpoint" in settings:
            self.camera.set_temp(float(settings["temperature_setpoint"]))
        if "input_port" in settings:
            self.kymera.set_flipper("input", settings["input_port"])
        if "output_port" in settings:
            self.kymera.set_flipper("output", settings["output_port"])
        if "readout_mode" in settings:
            self.camera.set_readout_mode(settings["readout_mode"])
        if "acquisition_mode" in settings:

            mode = settings["acquisition_mode"]

            if mode == "single":
                self.camera.set_acquisition_mode("single")

            elif mode == "accum":
                self.camera.setup_accum_mode(
                    num_acc=int(settings.get("num_accumulations", 1)),
                    cycle_time_acc=float(
                        settings.get("accumulation_cycle_time", 0.0)
                    )
                )
    
    def estimate_measure_time(self):
        try:
            return float(self.camera.get_exposure())
        except Exception:
            return None


    def _shutdown_impl(self):
        try:
            self.camera.shutdown()
            self.kymera.disconnect()
        finally:
            super()._shutdown_impl()
