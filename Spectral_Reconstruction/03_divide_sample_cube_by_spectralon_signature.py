from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================
# CONFIGURACION
# =========================

SAMPLE_DIR = Path(r"preparado_muestra")
SPECTRALON_DIR = Path(r"preparado_spectralon")
OUTPUT_DIR = Path(r"salida_cubo_dividido_spectralon")

SAMPLE_CUBE_NPY = SAMPLE_DIR / "sample_cube_y_xscan_lambda.npy"
WHITE_SIGNATURE_NPY = SPECTRALON_DIR / "white_signature_spectralon.npy"

SPECTRAL_COLS_NPY = SAMPLE_DIR / "spectral_pixel_columns.npy"
WAVELENGTHS_NPY = SAMPLE_DIR / "wavelengths_nm_approx.npy"

# None usa todas las bandas para preview.
PREVIEW_BAND_START = None
PREVIEW_BAND_STOP = None

EPS_PERCENTILE = 1
EPS_FACTOR = 0.01


def robust_limits(img: np.ndarray, low: float = 1, high: float = 99) -> tuple[float, float]:
    vmin, vmax = np.percentile(img, [low, high])
    if vmax <= vmin:
        vmin = float(np.min(img))
        vmax = float(np.max(img))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def normalize_cube_by_signature(cube: np.ndarray, white_signature: np.ndarray) -> np.ndarray:
    white_signature = white_signature.reshape(-1).astype(np.float32)

    if cube.shape[2] != white_signature.size:
        raise ValueError(f"Bandas incompatibles: cubo={cube.shape[2]}, blanco={white_signature.size}")

    eps = max(float(np.percentile(np.abs(white_signature), EPS_PERCENTILE)) * EPS_FACTOR, 1e-6)
    denominator = np.maximum(white_signature, eps)
    return (cube / denominator[None, None, :]).astype(np.float32)


def make_preview(cube: np.ndarray) -> np.ndarray:
    b1 = 0 if PREVIEW_BAND_START is None else PREVIEW_BAND_START
    b2 = cube.shape[2] if PREVIEW_BAND_STOP is None else PREVIEW_BAND_STOP

    if b1 < 0 or b2 > cube.shape[2] or b1 >= b2:
        raise ValueError(f"Preview bands invalidas: {b1}:{b2} para {cube.shape[2]} bandas")

    return cube[:, :, b1:b2].mean(axis=2)


def save_preview(path: Path, cube: np.ndarray, title: str) -> None:
    preview = make_preview(cube)
    vmin, vmax = robust_limits(preview)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(preview, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x_scan / frames")
    ax.set_ylabel("y_slit")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()

    np.save(path.with_suffix(".npy"), preview.astype(np.float32))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sample_cube = np.load(SAMPLE_CUBE_NPY, mmap_mode="r")
    white_signature = np.load(WHITE_SIGNATURE_NPY)

    print("Sample cube:", sample_cube.shape, sample_cube.dtype)
    print("White signature:", white_signature.shape, white_signature.dtype)

    cube_normalized = normalize_cube_by_signature(sample_cube, white_signature)

    print("Cubo dividido:", cube_normalized.shape, cube_normalized.dtype)
    print("Min/max:", float(np.nanmin(cube_normalized)), float(np.nanmax(cube_normalized)))

    np.save(OUTPUT_DIR / "cube_y_xscan_lambda_divided_by_spectralon.npy", cube_normalized)

    if SPECTRAL_COLS_NPY.exists():
        np.save(OUTPUT_DIR / "spectral_pixel_columns.npy", np.load(SPECTRAL_COLS_NPY))
    if WAVELENGTHS_NPY.exists():
        np.save(OUTPUT_DIR / "wavelengths_nm_approx.npy", np.load(WAVELENGTHS_NPY))

    np.save(OUTPUT_DIR / "white_signature_used.npy", white_signature.reshape(-1, 1))

    save_preview(
        OUTPUT_DIR / "preview_cube_divided_by_spectralon.png",
        cube_normalized,
        "Preview cubo muestra dividido por firma Spectralon",
    )

    metadata = {
        "sample_cube_npy": str(SAMPLE_CUBE_NPY),
        "white_signature_npy": str(WHITE_SIGNATURE_NPY),
        "normalized_cube_shape": tuple(int(v) for v in cube_normalized.shape),
        "axis_order": "(y_slit, x_scan, lambda_pixel)",
        "operation": "sample_cube / white_signature_spectralon",
        "note": "Este script solo divide. Muestra y Spectralon se preparan en scripts separados.",
    }
    np.save(OUTPUT_DIR / "metadata_divide.npy", metadata)

    print("Salida guardada en:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
