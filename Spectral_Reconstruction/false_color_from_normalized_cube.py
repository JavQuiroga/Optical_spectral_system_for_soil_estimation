from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================
# CONFIGURACION
# =========================

# Puede ser un cubo preparado, calibrado o normalizado, siempre que tenga forma:
# (y_slit, x_scan, bandas)
CUBE_NPY = Path(r"preparado_muestra_colorckeher\sample_cube_y_xscan_lambda.npy")
OUTPUT_DIR = Path(r"falso_color_colorckeher")

# Archivos opcionales. Si existen, se usan solo para rotular la imagen.
SPECTRAL_COLS_NPY = CUBE_NPY.parent / "spectral_pixel_columns.npy"
WAVELENGTHS_NPY = CUBE_NPY.parent / "wavelengths_nm.npy"
WAVELENGTHS_APPROX_NPY = CUBE_NPY.parent / "wavelengths_nm_approx.npy"

# Bandas automaticas como fraccion del eje espectral.
# B toma la zona baja, G la media y R la alta.
BAND_FRACTIONS = {
    "R": 0.,
    "G": 0.4,
    "B": 0.,
}

# Promedia bandas vecinas para bajar ruido. Usa 1 si quieres una sola banda exacta.
BAND_WINDOW = 7

# Ajuste visual por percentiles para reducir ruido/outliers.
P_LOW = 1
P_HIGH = 99

# Correccion gamma visual. Menor que 1 aclara sombras; 1 deja lineal.
GAMMA = 0.85


def robust_scale(channel: np.ndarray, p_low: float = P_LOW, p_high: float = P_HIGH) -> np.ndarray:
    finite = np.isfinite(channel)
    if not np.any(finite):
        return np.zeros_like(channel, dtype=np.float32)

    vmin, vmax = np.percentile(channel[finite], [p_low, p_high])
    if vmax <= vmin:
        vmin = float(np.nanmin(channel))
        vmax = float(np.nanmax(channel))
    if vmax <= vmin:
        return np.zeros_like(channel, dtype=np.float32)

    scaled = np.clip((channel - vmin) / (vmax - vmin), 0, 1)
    if GAMMA != 1:
        scaled = np.power(scaled, GAMMA)
    return scaled.astype(np.float32)


def fraction_to_band_index(fraction: float, n_bands: int) -> int:
    if n_bands <= 0:
        raise ValueError("El cubo no tiene bandas espectrales.")
    fraction = float(np.clip(fraction, 0, 1))
    return int(round(fraction * (n_bands - 1)))


def band_window_mean(cube: np.ndarray, center_idx: int, window: int = BAND_WINDOW) -> np.ndarray:
    half = max(int(window) // 2, 0)
    b1 = max(center_idx - half, 0)
    b2 = min(center_idx + half + 1, cube.shape[2])
    return cube[:, :, b1:b2].astype(np.float32, copy=False).mean(axis=2)


def load_vector(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path).reshape(-1)


def band_label(idx: int, spectral_cols: np.ndarray | None, wavelengths: np.ndarray | None) -> str:
    parts = [f"banda {idx}"]
    if spectral_cols is not None and idx < spectral_cols.size:
        parts.append(f"x={int(spectral_cols[idx])}")
    if wavelengths is not None and idx < wavelengths.size:
        parts.append(f"{float(wavelengths[idx]):.1f} nm")
    return ", ".join(parts)


def make_false_color(cube: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    n_bands = cube.shape[2]
    indices = {
        name: fraction_to_band_index(frac, n_bands)
        for name, frac in BAND_FRACTIONS.items()
    }

    r = robust_scale(band_window_mean(cube, indices["R"]))
    g = robust_scale(band_window_mean(cube, indices["G"]))
    b = robust_scale(band_window_mean(cube, indices["B"]))

    return np.dstack([r, g, b]), indices


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cube = np.load(CUBE_NPY, mmap_mode="r")
    if cube.ndim != 3:
        raise ValueError(f"Se esperaba cubo 3D (y_slit, x_scan, bandas), llego {cube.shape}")

    spectral_cols = load_vector(SPECTRAL_COLS_NPY)
    wavelengths = load_vector(WAVELENGTHS_NPY)
    if wavelengths is None:
        wavelengths = load_vector(WAVELENGTHS_APPROX_NPY)

    print("Cubo:", cube.shape, cube.dtype)

    rgb, indices = make_false_color(cube)

    labels = {
        name: band_label(idx, spectral_cols, wavelengths)
        for name, idx in indices.items()
    }

    title = (
        "Falso color adaptativo\n"
        f"R: {labels['R']} | G: {labels['G']} | B: {labels['B']}"
    )

    out_png = OUTPUT_DIR / "false_color_adaptive.png"
    out_npy = OUTPUT_DIR / "false_color_adaptive.npy"

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(rgb, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("x_scan / frames")
    ax.set_ylabel("y_slit")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    np.save(out_npy, rgb.astype(np.float32))

    metadata = {
        "cube_npy": str(CUBE_NPY),
        "cube_shape": tuple(int(v) for v in cube.shape),
        "band_fractions": BAND_FRACTIONS,
        "band_window": int(BAND_WINDOW),
        "selected_band_indices": {k: int(v) for k, v in indices.items()},
        "selected_labels": labels,
        "percentile_scale": [float(P_LOW), float(P_HIGH)],
        "gamma": float(GAMMA),
    }
    np.save(OUTPUT_DIR / "false_color_adaptive_metadata.npy", metadata)

    print("Bandas seleccionadas:")
    print("  R:", labels["R"])
    print("  G:", labels["G"])
    print("  B:", labels["B"])
    print("Guardado:", out_png)


if __name__ == "__main__":
    main()
