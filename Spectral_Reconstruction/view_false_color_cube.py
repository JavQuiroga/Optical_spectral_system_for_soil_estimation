from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================
# CONFIGURACION
# =========================

# Puedes apuntar a cualquiera de estos:
#   preparado_muestra\sample_cube_y_xscan_lambda.npy
#   preparado_spectralon\spectralon_cube_y_xscan_lambda.npy
#   salida_cubo_dividido_spectralon\cube_y_xscan_lambda_divided_by_spectralon.npy
CUBE_NPY = Path(r"preparado_muestra_colorckeher\sample_cube_y_xscan_lambda.npy")

OUTPUT_DIR = Path(r"falso_color")

# Modo para seleccionar bandas:
#   "index"    -> R_BAND/G_BAND/B_BAND son indices internos del cubo: 0..n_bands-1
#   "original" -> R_BAND/G_BAND/B_BAND son columnas originales x_sensor
BAND_MODE = "index"

# Si BAND_MODE = "index", ejemplo para un cubo con 64 bandas: 0, 32, 63.
# Si BAND_MODE = "original", ejemplo: columnas originales 245, 275, 308.
R_BAND = 236
G_BAND = 210
B_BAND = 188

# Si usas BAND_MODE="original", define el recorte espectral con el que fue creado el cubo.
SPECTRAL_X1 = 245
SPECTRAL_X2 = 309

# Escalado por canal para visualizacion.
P_LOW = 1
P_HIGH = 99

# Si True, muestra una cuadricula de bandas individuales para ayudarte a escoger.
SHOW_BAND_GRID = True
GRID_BANDS = 12


def robust_scale(channel: np.ndarray, p_low: float = P_LOW, p_high: float = P_HIGH) -> np.ndarray:
    vmin, vmax = np.percentile(channel, [p_low, p_high])
    if vmax <= vmin:
        vmin = float(np.min(channel))
        vmax = float(np.max(channel))
    if vmax <= vmin:
        return np.zeros_like(channel, dtype=np.float32)
    return np.clip((channel - vmin) / (vmax - vmin), 0, 1).astype(np.float32)


def band_to_index(band: int, n_bands: int) -> int:
    if BAND_MODE == "index":
        idx = int(band)
    elif BAND_MODE == "original":
        idx = int(band) - SPECTRAL_X1
    else:
        raise ValueError("BAND_MODE debe ser 'index' u 'original'")

    if idx < 0 or idx >= n_bands:
        if BAND_MODE == "original":
            raise ValueError(
                f"Banda original {band} fuera del recorte {SPECTRAL_X1}:{SPECTRAL_X2}. "
                f"Columnas validas: {SPECTRAL_X1}..{SPECTRAL_X2 - 1}"
            )
        raise ValueError(f"Indice {idx} fuera del cubo. Indices validos: 0..{n_bands - 1}")

    return idx


def show_band_grid(cube: np.ndarray, out_dir: Path) -> None:
    n_bands = cube.shape[2]
    count = min(GRID_BANDS, n_bands)
    indices = np.linspace(0, n_bands - 1, count, dtype=int)

    cols = 4
    rows = int(np.ceil(count / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes, indices):
        img = cube[:, :, idx]
        vmin, vmax = np.percentile(img, [P_LOW, P_HIGH])
        if vmax <= vmin:
            vmin, vmax = float(np.min(img)), float(np.max(img))
        ax.imshow(img, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
        if BAND_MODE == "original":
            title = f"idx {idx} | col {SPECTRAL_X1 + idx}"
        else:
            title = f"idx {idx}"
        ax.set_title(title)
        ax.axis("off")

    for ax in axes[count:]:
        ax.axis("off")

    fig.suptitle("Bandas individuales para escoger falso color")
    fig.tight_layout()
    fig.savefig(out_dir / "band_grid.png", dpi=150)
    plt.show()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cube = np.load(CUBE_NPY, mmap_mode="r")
    if cube.ndim != 3:
        raise ValueError(f"Se esperaba cubo 3D (y, x_scan, lambda), llego {cube.shape}")

    print("Cubo:", cube.shape)
    print("Bandas disponibles:", cube.shape[2])

    if SHOW_BAND_GRID:
        show_band_grid(cube, OUTPUT_DIR)

    r_idx = band_to_index(R_BAND, cube.shape[2])
    g_idx = band_to_index(G_BAND, cube.shape[2])
    b_idx = band_to_index(B_BAND, cube.shape[2])

    print(f"R idx={r_idx}, G idx={g_idx}, B idx={b_idx}")

    r = robust_scale(cube[:, :, r_idx])
    g = robust_scale(cube[:, :, g_idx])
    b = robust_scale(cube[:, :, b_idx])

    rgb = np.dstack([r, g, b])

    if BAND_MODE == "original":
        label = f"R{R_BAND}_G{G_BAND}_B{B_BAND}"
        title = f"Falso color columnas originales R={R_BAND}, G={G_BAND}, B={B_BAND}"
    else:
        label = f"Ridx{R_BAND}_Gidx{G_BAND}_Bidx{B_BAND}"
        title = f"Falso color indices R={R_BAND}, G={G_BAND}, B={B_BAND}"

    plt.figure(figsize=(12, 7))
    plt.imshow(rgb, aspect="auto")
    plt.title(title)
    plt.xlabel("x_scan / frames")
    plt.ylabel("y_slit")
    plt.tight_layout()
    plt.show()

    out_png = OUTPUT_DIR / f"false_color_{label}.png"
    out_npy = OUTPUT_DIR / f"false_color_{label}.npy"

    plt.imsave(out_png, rgb)
    np.save(out_npy, rgb.astype(np.float32))

    print("Guardado:", out_png)


if __name__ == "__main__":
    main()
