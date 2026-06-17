from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================
# CONFIGURACION
# =========================

SPECTRALON_NPY = Path(r"Blanco_espectralon\cube_20260505_091738.npy")
OUTPUT_DIR = Path(r"preparado_spectralon")

# raw.shape = (y_sensor, x_sensor, frames)
SLIT_Y1 = 500
SLIT_Y2 = 750

SPECTRAL_X1 = 280
SPECTRAL_X2 = 365

PREVIEW_X1 = 280
PREVIEW_X2 = 365

FLIP_SCAN_AXIS = True
FLIP_SLIT_AXIS = False
FLIP_LAMBDA_AXIS = False

WHITE_REDUCTION = "median"  # "median" o "mean"


def robust_limits(img: np.ndarray, low: float = 1, high: float = 99) -> tuple[float, float]:
    vmin, vmax = np.percentile(img, [low, high])
    if vmax <= vmin:
        vmin = float(np.min(img))
        vmax = float(np.max(img))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def load_npy_checked(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Archivo vacio: {path}")
    try:
        return np.load(path, mmap_mode="r")
    except ValueError as exc:
        raise ValueError(f"No pude leer {path}; el .npy parece incompleto/corrupto.") from exc


def reconstruct_cube(path: Path) -> np.ndarray:
    raw = load_npy_checked(path)
    print(f"{path.name} raw shape:", raw.shape)

    if raw.ndim != 3:
        raise ValueError(f"Se esperaba raw 3D (y_sensor, x_sensor, frames), llego {raw.shape}")

    raw_crop = raw[SLIT_Y1:SLIT_Y2, SPECTRAL_X1:SPECTRAL_X2, :]
    raw_crop = raw_crop.astype(np.float32, copy=False)

    cube = np.transpose(raw_crop, (0, 2, 1))  # (y_slit, x_scan, lambda_pixel)

    if FLIP_SLIT_AXIS:
        cube = np.flip(cube, axis=0)
    if FLIP_SCAN_AXIS:
        cube = np.flip(cube, axis=1)
    if FLIP_LAMBDA_AXIS:
        cube = np.flip(cube, axis=2)

    print(f"Cubo Spectralon shape:", cube.shape)
    return cube.astype(np.float32, copy=False)


def get_preview(cube: np.ndarray) -> np.ndarray:
    i1 = PREVIEW_X1 - SPECTRAL_X1
    i2 = PREVIEW_X2 - SPECTRAL_X1
    if i1 < 0 or i2 > cube.shape[2] or i1 >= i2:
        raise ValueError(f"Preview {PREVIEW_X1}:{PREVIEW_X2} fuera del recorte espectral.")
    return cube[:, :, i1:i2].mean(axis=2)


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


def white_signature_from_roi(cube: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.shape != cube.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} no coincide con cubo {cube.shape[:2]}")

    pixels = cube[mask, :]
    if pixels.shape[0] == 0:
        raise ValueError("La ROI del Spectralon no contiene pixeles.")

    if WHITE_REDUCTION == "median":
        signature = np.median(pixels, axis=0)
    elif WHITE_REDUCTION == "mean":
        signature = np.mean(pixels, axis=0)
    else:
        raise ValueError("WHITE_REDUCTION debe ser 'median' o 'mean'")

    return signature.astype(np.float32)


def save_preview(out_dir: Path, cube: np.ndarray, mask: np.ndarray | None = None) -> None:
    preview = get_preview(cube)
    vmin, vmax = robust_limits(preview)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(preview, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
    if mask is not None:
        ax.contour(mask, colors="red", linewidths=2)
    ax.set_title("Preview cubo Spectralon reconstruido")
    ax.set_xlabel("x_scan / frames")
    ax.set_ylabel("y_slit")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "preview_spectralon_cube.png", dpi=150)
    plt.show()

    np.save(out_dir / "preview_spectralon_cube.npy", preview.astype(np.float32))


def save_raw_slit_diagnostic(out_dir: Path, raw_path: Path, frame_index: int | None = None) -> None:
    raw = load_npy_checked(raw_path)
    if frame_index is None:
        frame_index = raw.shape[2] // 2

    frame_index = int(np.clip(frame_index, 0, raw.shape[2] - 1))
    frame = raw[:, :, frame_index]
    vmin, vmax = robust_limits(frame, low=1, high=99.8)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(frame, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
    ax.axhline(SLIT_Y1, color="red", linestyle="--", linewidth=1.2, label="slit y")
    ax.axhline(SLIT_Y2, color="red", linestyle="--", linewidth=1.2)
    ax.axvline(SPECTRAL_X1, color="lime", linestyle="--", linewidth=1.2, label="rango espectral")
    ax.axvline(SPECTRAL_X2, color="lime", linestyle="--", linewidth=1.2)
    ax.axvline(PREVIEW_X1, color="cyan", linestyle="--", linewidth=1.0, label="preview")
    ax.axvline(PREVIEW_X2, color="cyan", linestyle="--", linewidth=1.0)
    ax.set_title(f"Frame raw Spectralon {frame_index}: rojo=slit, verde=espectral, cyan=preview")
    ax.set_xlabel("x_sensor / lambda_pixel")
    ax.set_ylabel("y_sensor")
    ax.legend(loc="best")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostico_frame_raw_spectralon_slit.png", dpi=150)
    plt.show()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cube = reconstruct_cube(SPECTRALON_NPY)
    save_raw_slit_diagnostic(OUTPUT_DIR, SPECTRALON_NPY)
    preview = get_preview(cube)
    mask = select_ellipse_mask(preview, "ROI Spectralon / blanco")
    white_signature = white_signature_from_roi(cube, mask)

    spectral_cols = np.arange(SPECTRAL_X1, SPECTRAL_X2, dtype=np.int32)
    if FLIP_LAMBDA_AXIS:
        spectral_cols = spectral_cols[::-1]

    wavelengths_nm_approx = np.linspace(400, 1700, cube.shape[2], dtype=np.float32)

    np.save(OUTPUT_DIR / "spectralon_cube_y_xscan_lambda.npy", cube)
    np.save(OUTPUT_DIR / "white_signature_spectralon.npy", white_signature.reshape(-1, 1))
    np.save(OUTPUT_DIR / "white_mask_spectralon.npy", mask)
    np.save(OUTPUT_DIR / "spectral_pixel_columns.npy", spectral_cols.reshape(-1, 1))
    np.save(OUTPUT_DIR / "wavelengths_nm_approx.npy", wavelengths_nm_approx.reshape(-1, 1))
    save_preview(OUTPUT_DIR, cube, mask)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(spectral_cols, white_signature)
    ax.set_xlabel("Columna espectral del sensor")
    ax.set_ylabel("Intensidad Spectralon")
    ax.set_title("Firma blanca extraida del Spectralon")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "white_signature_spectralon.png", dpi=150)
    plt.show()

    metadata = {
        "spectralon_npy": str(SPECTRALON_NPY),
        "spectralon_cube_shape": tuple(int(v) for v in cube.shape),
        "axis_order": "(y_slit, x_scan, lambda_pixel)",
        "slit_y_crop": [SLIT_Y1, SLIT_Y2],
        "spectral_x_crop": [SPECTRAL_X1, SPECTRAL_X2],
        "white_reduction": WHITE_REDUCTION,
        "note": "Este script solo reconstruye Spectralon y extrae firma blanca.",
    }
    np.save(OUTPUT_DIR / "metadata_spectralon.npy", metadata)

    print("Spectralon preparado en:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
