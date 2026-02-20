from flask import Flask, request, jsonify
import threading
import time

from Spectrometer import AndorCameraController, KymeraController, SpectrometerController 
 

app = Flask(__name__)

camera = AndorCameraController()
camera.connect()
kymera = KymeraController(device_index=0)
kymera.setup_from_camera(camera.cam)
spec = SpectrometerController(camera, kymera)


@app.route("/api/camera/status")
def camera_status():
    return jsonify(camera.get_status())

@app.route("/api/camera/exposure", methods=["POST"])
def set_exposure():
    exp = float(request.json["exposure"])
    camera.set_exposure(exp)
    return jsonify({"exposure": exp})

@app.route("/api/camera/exposure")
def get_exposure():
    return jsonify({"exposure": camera.get_exposure()})

@app.route("/api/camera/cooler", methods=["POST"])
def set_cooler():
    on = bool(request.json["on"])
    camera.set_cooler(on)
    return jsonify({"cooler": on})

@app.route("/api/camera/temperature")
def get_temperature():
    return jsonify({
        "temperature": camera.get_temperature(),
        "status": camera.get_temp_status()
    })

@app.route("/api/camera/roi", methods=["POST"])
def set_roi():
    data = request.json
    camera.set_roi(
        hbin=data.get("hbin", 1),
        vbin=data.get("vbin", 1),
        hstart=data.get("hstart", 0),
        hend=data.get("hend"),
        vstart=data.get("vstart", 0),
        vend=data.get("vend")
    )
    return jsonify({"status": "roi set"})

@app.route("/api/kymera/status")
def kymera_status():
    return jsonify(kymera.get_status())

@app.route("/api/kymera/grating", methods=["POST"])
def set_grating():
    idx = int(request.json["index"])
    kymera.set_grating(idx)
    return jsonify({"grating": idx})

@app.route("/api/kymera/central_wavelength", methods=["POST"])
def set_central_wavelength():
    wl = float(request.json["wavelength_nm"])
    kymera.set_central_wavelength(wl)
    return jsonify({"central_wavelength_nm": wl})

@app.route("/api/kymera/central_wavelength")
def get_central_wavelength():
    return jsonify({"central_wavelength_nm": kymera.get_central_wavelength()})

@app.route("/api/kymera/slit_width", methods=["POST"])
def set_slit_width():
    width = float(request.json["width_um"])
    kymera.set_slit_width_um(width)
    return jsonify({"slit_width_um": width})

@app.route("/api/kymera/slit_width")
def get_slit_width():
    return jsonify({"slit_width_um": kymera.get_slit_width_um()})

@app.route("/api/spectrum/acquire", methods=["POST"])
def acquire_spectrum():
    laser_wl = float(request.json["laser_wavelength_nm"])
    spectrum, wl, raman = spec.acquire_spectrum(laser_wl)
    return jsonify({
        "wavelength_nm": wl.tolist(),
        "raman_shift": raman.tolist(),
        "intensity": spectrum.tolist()
    })

@app.route("/api/spectrum/acquire_async", methods=["POST"])
def acquire_spectrum_async():
    laser_wl = float(request.json["laser_wavelength_nm"])
    def worker():
        global last_spectrum
        last_spectrum = spec.acquire_spectrum(laser_wl)
    
    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"status": "acquisition started"})

@app.route("/api/spectrum/last")
def get_last_spectrum():
    if "last_spectrum" in globals():
        return jsonify({"error": "no spectrum acquired yet"}), 404
    
    spectrum, wl, raman = last_spectrum
    return jsonify({
        "wavelength_nm": wl.tolist(),
        "raman_shift": raman.tolist(),
        "intensity": spectrum.tolist()
    })

@app.route("/api/shutdown", methods=["POST"])
def shutdown():
    camera.shutdown()
    kymera.disconnect()
    return jsonify({"status": "shutdown complete"})

if __name__ == "__main__":
    app.run(debug=True)

