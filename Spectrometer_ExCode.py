import sys 
from astropy.io import fits 
from pylablib.devices import Andor
from pylablib.devices.Andor import Shamrock
import threading
import time
import matplotlib.pyplot as plt
import os 
from astropy.io import fits
import numpy as np 

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

        self._lock = threading.Lock()
        self.keep_cooling = False
        self.temperature_setpoint = None

    # Connection control
    def connect(self):
        "Open camera connection"
        if not self.connected:
            self.cam = Andor.AndorSDK2Camera()
            self.connected = True

    def disconnect(self):
        "Close camera connection"
        if self.connected:
            self.cam.close()
            self.connected = False


    # Cooling control
    def enable_cooling(self, temperature):
        with self._lock:
            self.cam.set_cooler(True)
            self.cam.set_temperature(temperature)
            self.cooler_enabled = True
            self.temperature_setpoint = temperature

    def disable_cooling(self):
        with self._lock:
            self.cam.set_cooler(False)
            self.cooler_enabled = False

    def get_temperature(self):
        return self.cam.get_temperature()
    
    def maintain_cooling(self):
        if not getattr(self, "keep_cooling", False):
            return

        with self._lock:
            self.cam.set_cooler(True)

            if self.temperature_setpoint is not None:
                self.cam.set_temperature(self.temperature_setpoint)
    
    def enable_cooling(self, setpoint):
        with self._lock:
            self.cam.set_cooler(True)
            self.cam.set_temperature(setpoint)
            self.temperature_setpoint = setpoint
            self.keep_cooling = True
            
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
    
    def set_fvb(self):
        with self._lock:
            self.cam.set_roi(vbin="full")
            self.vbin = "full"
    
    #Trigger control
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

    def set_acquisition_mode(self, mode="single"):
        with self._lock:
            self.cam.set_acquisition_mode(mode)
            self.acquisition_mode = mode
        
    # Acquisition
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
    
    """def set_kinetic_mode(self):
        with self._lock:
            self.cam.set_acquisition_mode("kinetic")
            self.acquisition_mode = "kinetic"

    
    def acquire_kinetic(self, num_frames, mode="kinetic", timeout=10):
        with self._lock:
            if mode not in ["kinetic", "fast_kinetic"]:
                raise ValueError("Mode must be 'kinetic' or 'fast kinetic'")
            self.cam.set_acquisition_mode(mode)
            self.acquisition_mode = mode
            
            self.cam.set_trigger_mode("int")
            
            self.cam.start_acquisition()

            frames = []

            for i in range(num_frames):
                self.cam.wait_for_frame(timeout=timeout)
                img = self.cam.read_newest_image()

                if img is None:
                    raise RuntimeError(f"Frame {i} is None")

                frames.append(img)

            return frames"""

    
    #File Saving 
    def save_image(self, image, filename=None, directory=None, save_preview=True):
        if directory is None:
            directory = os.getcwd()
        os.makedirs(directory, exist_ok=True)
        
        if filename is None:
            filename = f"image_{time.strftime('%Y%m%d-%H%M%S')}.fits"
        
        full_path = os.path.join(directory, filename)
        
        hdr = fits.Header()
        hdr['EXPOSURE'] = self.exposure 
        hdr['H_BIN'] = self.hbin
        hdr['V_BIN'] = self.vbin
        hdr['TEMP_SET'] = self.temperature_setpoint
        hdr['COOLER'] = self.cooler_enabled
        hdr['ACQ_MODE'] = self.acquisition_mode
        hdr['DATE'] = time.strftime("%Y-%m-%dT%H:%M:%S")
        hdu = fits.PrimaryHDU(image, header=hdr)
        hdu.writeto(full_path, overwrite=True)
        
        if save_preview:
            preview_file = full_path.replace('.fits', '.png')
            plt.imsave(preview_file, image, cmap='gray')
            
        return full_path

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

    def setup_from_camera(self, camera):
        self.spec.setup_pixels_from_camera(camera)
        self._wl_cache = None

    def set_grating(self, index):
        self.spec.set_grating(index)
        self._wl_cache = None

    def set_central_wavelength(self, wl_nm):
        self.spec.set_wavelength(wl_nm * 1e-9)
        self._wl_cache = None

    def get_calibration_nm(self):
        if self._wl_cache is None:
            wl_m = self.spec.get_calibration()  # meters
            self._wl_cache = wl_m * 1e9        # convert to nm
        return self._wl_cache
    
    def get_grating(self):
        return self.spec.get_greating()
    
    def get_central_wavelength(self):
        return self.spec.get_wavelength() * 1e9
    
    def get_slit_width_um(self, slit="input_side"):
        width_m = self.spec.get_slit_width(slit)
        return width_m * 1e6

    #not working 
    """def set_slit_width_um(self, width_um, slit="input_side"):
        width_m = width_um * 1e6
        self.spec.set_slit_width(slit, width_m)"""
    
    def list_gratings(self):
        try:
            info = self.spec.get_greating_info()
            return [g['name'] for g in info]
        except AttributeError:
            return [0, 1, 2]
    
    def get_wavelength_span(self):
        wl = self.get_calibration_nm()
        return wl[0], wl[-1]
    
    def get_status(self):
        return {
            "grating": self.get_grating(),
            "central_wavelength_nm": self.get_central_wavelength(),
            "wavelength_range_nm": self.get_wavelength_span()
        }

class SpectrometerController:
    def __init__(self, camera_controller, kymera_controller):
        self.camera = camera_controller
        self.kymera = kymera_controller
        self._lock = threading.Lock()
    
    def connect(self):
        self.camera.connect()
        self.kymera.connect()
    
    def disconnect(self):
        self.camera.shutdown()
        self.kymera.disconnect()
    
    def acquire_image(self):
        with self._lock:
            return self.camera.acquire_single()
    
    def get_wavelength_axis(self):
        return self.kymera.get_calibration_nm()
    
    def extract_spectrum(self, image, axis=0):
        return np.sum(image, axis=axis)
    
    def acquire_spectrum(self, axis=0):
        img = self.acquire_image()
        spectrum = self.extract_spectrum(img, axis=axis)
        wl = self.get_wavelength_axis()
        return spectrum, wl, img
    
    def save_spectrum_csv(self, spectrum, wavelength_nm, filename=None):
        if filename is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"spectrum_{ts}.csv"

        metadata = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "exposure_s": getattr(self.camera, "exposure", "unknown"),
            "acquisition_mode": getattr(self.camera, "acquisition_mode", "unknown"),
            "trigger_mode": getattr(self.camera, "trigger_mode", "unknown"),
            "grating": getattr(self.kymera.spec, "get_grating", lambda: "unknown")(),
            "center_wavelength_nm": getattr(
                self.kymera.spec, "get_wavelength", lambda: None
            )(),
            "num_pixels": len(wavelength_nm),
        }

        if metadata["center_wavelength_nm"] not in ("unknown", None):
            metadata["center_wavelength_nm"] *= 1e9

        with open(filename, "w") as f:
            # Metadata header
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")

            # Column header
            f.write("wavelength_nm,intensity\n")

            # Data
            for wl, val in zip(wavelength_nm, spectrum):
                f.write(f"{wl},{val}\n")

        return os.path.abspath(filename)
    
    def get_status(self):
        status = self.cam_ctrl.get_status()
        status.update({
            "wavelength_range_nm": self.get_wavelength_axis()[[0, -1]] if self.kymera_ctrl else None
        })
        return status

    def plot_spectrum(self, spectrum, wl=None):
        if wl is None:
            wl = self.get_wavelength_axis()
        plt.plot(wl, spectrum)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity (a.u.)")
        plt.show()

"""controller = AndorCameraController()

try:
    controller.connect()
    controller.enable_cooling(-90)
    controller.set_exposure(0.05)        # exposure time
    controller.set_acquisition_mode("single")        # sets acquisition mode = "kinetics"
    controller.cam.set_trigger_mode("int")  # internal trigger required


    # Take image
    image = controller.acquire_single()

    # Display it
    plt.imshow(image, cmap="gray")
    plt.colorbar()
    plt.show()
    
    #controller.save_image(image)

finally:
    controller.shutdown()"""

"""camera_ctrl = AndorCameraController()
kymera_ctrl = KymeraController()

camera_ctrl.connect()
camera_ctrl.enable_cooling(-90)  # Example temperature
camera_ctrl.set_exposure(0.5)    # Example exposure (s)

kymera_ctrl.setup_from_camera(camera_ctrl.cam)
kymera_ctrl.set_grating(1)           # Set grating index
kymera_ctrl.set_central_wavelength(500)  # 500 nm

spec_ctrl = SpectrometerController(camera_ctrl, kymera_ctrl)

spectrum, wavelength_nm, image = spec_ctrl.acquire_spectrum()

print("Image shape:", image.shape)
print("Wavelength range (nm):", wavelength_nm[0], "→", wavelength_nm[-1])

spec_ctrl.plot_spectrum(spectrum, wl=wavelength_nm)

camera_ctrl.shutdown()"""






# In[ ]:




