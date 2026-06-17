from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================
# CONFIGURACION
# =========================

# Puedes usar:
#   preparado_muestra\sample_cube_y_xscan_lambda.npy
#   salida_cubo_dividido_spectralon\cube_y_xscan_lambda_divided_by_spectralon.npy
CUBE_NPY = Path(r"preparado_muestra\sample_cube_y_xscan_lambda.npy")

OUTPUT_DIR = Path(r"firma_extraida_muestra")

# Preview para seleccionar ROI.
# Si None, usa todas las bandas.
PREVIEW_BAND_START = None
PREVIEW_BAND_STOP = None

# Archivos opcionales de eje espectral.
SPECTRAL_COLS_NPY = Path(r"preparado_muestra\spectral_pixel_columns.npy")
WAVELENGTHS_NPY = Path(r"preparado_muestra\wavelengths_nm_approx.npy")

# Nombre de salida
SIGNATURE_NAME = "firma_muestra_promedio.npy"


def robust_limits(img: np.ndarray, low: float = 1, high: float = 99) -> tuple[float, float]:
    vmin, vmax = np.percentile(img, [low, high])
    if vmax <= vmin:
        vmin = float(np.min(img))
        vmax = float(np.max(img))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def make_preview(cube: np.ndarray) -> np.ndarray:
    b1 = 0 if PREVIEW_BAND_START is None else PREVIEW_BAND_START
    b2 = cube.shape[2] if PREVIEW_BAND_STOP is None else PREVIEW_BAND_STOP

    if b1 < 0 or b2 > cube.shape[2] or b1 >= b2:
        raise ValueError(f"Preview bands invalidas: {b1}:{b2} para {cube.shape[2]} bandas")

    return cube[:, :, b1:b2].mean(axis=2)


def select_ellipse_mask(gray: np.ndarray, title: str) -> np.ndarray:
    vmin, vmax = robust_limits(gray)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(gray, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title + "\nSelecciona 2 puntos: esquina sup-izq y esquina inf-der de la elipse")
    ax.set_xlabel("x_scan / frames")
    ax.set_ylabel("y_slit")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    points = plt.ginput(2, timeout=0)
    plt.close(fig)

    if len(points) != 2:
        raise RuntimeError("Debes seleccionar exactamente 2 puntos.")

    (x1, y1), (x2, y2) = points
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])

    cy = (y_min + y_max) / 2
    cx = (x_min + x_max) / 2
    ry = max((y_max - y_min) / 2, 1e-6)
    rx = max((x_max - x_min) / 2, 1e-6)

    y_grid, x_grid = np.indices(gray.shape)
    mask = ((y_grid - cy) / ry) ** 2 + ((x_grid - cx) / rx) ** 2 <= 1

    print("Pixeles seleccionados:", int(mask.sum()))
    return mask


def mean_signature(cube: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.shape != cube.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} no coincide con cubo {cube.shape[:2]}")
    if mask.sum() == 0:
        raise ValueError("La mascara no contiene pixeles.")
    return cube[mask, :].mean(axis=0).astype(np.float32)


def save_roi_preview(out_dir: Path, gray: np.ndarray, mask: np.ndarray) -> None:
    vmin, vmax = robust_limits(gray)
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(gray, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
    ax.contour(mask, colors="red", linewidths=2)
    ax.set_title("ROI muestra seleccionada")
    ax.set_xlabel("x_scan / frames")
    ax.set_ylabel("y_slit")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "roi_muestra.png", dpi=150)
    plt.show()

    np.save(out_dir / "preview_roi_base.npy", gray.astype(np.float32))


def plot_signature(out_dir: Path, signature: np.ndarray) -> None:
    signature = signature.reshape(-1)

    if SPECTRAL_COLS_NPY.exists():
        x = np.load(SPECTRAL_COLS_NPY).reshape(-1)
        x_label = "Columna espectral del sensor"
        out_name = "firma_muestra_vs_columnas.png"
    elif WAVELENGTHS_NPY.exists():
        x = np.load(WAVELENGTHS_NPY).reshape(-1)
        x_label = "Longitud de onda aproximada (nm)"
        out_name = "firma_muestra_vs_wavelengths_aprox.png"
    else:
        x = np.arange(signature.size)
        x_label = "Indice de banda"
        out_name = "firma_muestra_vs_indice.png"

    if x.size != signature.size:
        x = np.arange(signature.size)
        x_label = "Indice de banda"
        out_name = "firma_muestra_vs_indice.png"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, signature)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Intensidad / valor promedio")
    ax.set_title("Firma promedio de la muestra")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / out_name, dpi=150)
    plt.show()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cube = np.load(CUBE_NPY, mmap_mode="r")
    if cube.ndim != 3:
        raise ValueError(f"Se esperaba cubo 3D (y, x_scan, lambda), llego {cube.shape}")

    print("Cubo:", cube.shape, cube.dtype)

    preview = make_preview(cube)
    mask = select_ellipse_mask(preview, "ROI muestra")
    signature = mean_signature(cube, mask)

    np.save(OUTPUT_DIR / SIGNATURE_NAME, signature.reshape(-1, 1))
    np.save(OUTPUT_DIR / "mask_muestra.npy", mask)
    save_roi_preview(OUTPUT_DIR, preview, mask)
    plot_signature(OUTPUT_DIR, signature)

    metadata = {
        "cube_npy": str(CUBE_NPY),
        "cube_shape": tuple(int(v) for v in cube.shape),
        "signature_shape": tuple(int(v) for v in signature.reshape(-1, 1).shape),
        "axis_order": "(y_slit, x_scan, lambda_pixel)",
        "operation": "mean cube[mask, :] over selected ROI",
    }
    np.save(OUTPUT_DIR / "metadata_signature.npy", metadata)

    print("Firma guardada:", OUTPUT_DIR / SIGNATURE_NAME)


if __name__ == "__main__":
    main()
