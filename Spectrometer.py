import sys 
from astropy.io import fits 
from pylablib.devices import Andor
from pylablib.devices.Andor import AndorSDK2Camera
from pylablib.devices.Andor import Shamrock
import threading
import time
import matplotlib.pyplot as plt
import os 
from astropy.io import fits
import numpy as np 
#for cosmic ray
"""from pathlib import Path
from astropy.nddata import CCDData
from astropy.nddata import block_replicate
from astropy import units as u
import ccdproc as ccdp
from photutils.segmentation import detect_sources 
from convenience_functions import show_image, display_cosmic_rays"""

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
    
    def setup_accum_mode(self, num_acc, cycle_time_acc=0):
        self.cam.setup_accum_mode(num_acc, cycle_time_acc)
    
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
    
    def wavelength_to_raman_shift(self, wl_nm, laser_nm):
        wl_nm = np.array(wl_nm)
        return(1/laser_nm - 1/wl_nm) * 1e7
        
    def extract_spectrum(self, image, axis=0):
        return np.sum(image, axis=axis)
    
    """def acquire_spectrum(self, axis=0):
        img = self.acquire_image()
        spectrum = self.extract_spectrum(img, axis=axis)
        wl = self.get_wavelength_axis()
        return spectrum, wl, img"""
    
    def acquire_spectrum(self, laser_wl):
        image = self.acquire_image()
        spectrum = image.mean(axis=0)
        wl = self.kymera.get_calibration_nm()
        raman = self.wavelength_to_raman_shift(wl, laser_wl)
        return spectrum, wl, raman  
    
    def acquire_spectrum_software(self, laser_wl, pixel_width=26.0):
        image = self.camera.acquire_software_triggered()
        spectrum = image.mean(axis=0)
        wl = self.kymera.get_calibration_nm()
        raman = self.wavelength_to_raman_shift(wl, laser_wl)
        return spectrum, wl, raman
    
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

    def plot_wavelength_spectrum(self, spectrum, wl=None):
        if wl is None:
            wl = self.get_wavelength_axis()
        plt.plot(wl, spectrum)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity (counts)")
        plt.title("Spectrum (Wavelength)")
        plt.show()
    
    def plot_raman_spectrum(self, raman, spectrum):
        plt.plot(raman, spectrum)
        plt.xlabel("Raman Shift (cm$^{-1}$)")
        plt.ylabel("Intensity (counts)")
        plt.title("Spectrum (Raman)")
        plt.gca().invert_xaxis()
        plt.show()






