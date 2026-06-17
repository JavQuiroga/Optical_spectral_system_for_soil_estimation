import gxipy as gx
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def save_png_uint8(img: np.ndarray, out_path: Path):
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())

    if mx <= mn:
        img8 = np.zeros_like(img, dtype=np.uint8)
    else:
        img8 = ((img - mn) * (255.0 / (mx - mn))).clip(0, 255).astype(np.uint8)

    try:
        import imageio.v2 as imageio
        imageio.imwrite(str(out_path), img8)
        print("PNG guardado:", out_path)
    except Exception as e:
        print("No pude guardar PNG. Instala imageio.")
        print("Error:", e)

    return img8


def set_exposure_time(cam, exposure_time_us: float):
    remote = cam.get_remote_device_feature_control()

    if remote.is_implemented("ExposureAuto") and remote.is_writable("ExposureAuto"):
        remote.get_enum_feature("ExposureAuto").set("Off")

    if remote.is_implemented("ExposureTime") and remote.is_writable("ExposureTime"):
        remote.get_float_feature("ExposureTime").set(float(exposure_time_us))

    print(f"Tiempo de exposición configurado a {exposure_time_us} microsegundos.")


def main(exposure_time_us: float = 3000):
    out_dir = Path("test_capture")
    out_dir.mkdir(exist_ok=True)

    dm = gx.DeviceManager()
    num, info = dm.update_all_device_list()
    print("Cámaras detectadas:", num)

    if num == 0:
        raise SystemExit("No detecté cámara. Revisa Galaxy Viewer / red / firewall.")

    cam = dm.open_device_by_index(1)

    try:
        cam.stream_on()

        set_exposure_time(cam, exposure_time_us)

        print("Capturando imagen...")
        raw = cam.data_stream[0].get_image()

        if raw is None:
            raise RuntimeError("get_image() devolvió None.")

        img = raw.get_numpy_array()

        if img is None:
            raise RuntimeError("No pude convertir la imagen a numpy array.")

        print("Frame capturado:")
        print("Shape:", img.shape)
        print("Dtype:", img.dtype)
        print("Min/Max:", img.min(), img.max())

        # Guardar imagen cruda en NPY
        npy_path = out_dir / "frame_raw.npy"
        np.save(npy_path, img)
        print("NPY guardado:", npy_path)

        # Guardar PNG visual
        png_path = out_dir / "frame_preview.png"
        img8 = save_png_uint8(img, png_path)

        # Mostrar visualización
        plt.figure(figsize=(8, 5))
        plt.imshow(img8, cmap="gray")
        plt.colorbar(label="Intensidad escalada 8-bit")
        plt.title(f"Frame capturado | Exposure = {exposure_time_us} us")
        plt.tight_layout()
        plt.show()

    finally:
        cam.stream_off()
        cam.close_device()
        print("OK captura y guardado.")


if __name__ == "__main__":
    main(exposure_time_us=3000)