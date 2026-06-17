from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================
# CONFIGURACION
# =========================

SAMPLE_NPY = Path(r"Soil_721\cube_20260430_220712.npy")
SPECTRALON_NPY = Path(r"Blanco_espectralon\cube_20260505_091738.npy")

OUTPUT_DIR = Path(r"salida_spectralon_normalizado")

# raw.shape = (y_sensor, x_sensor, frames)
SLIT_Y1 = 300
SLIT_Y2 = 870

SPECTRAL_X1 = 312
SPECTRAL_X2 = 450

PREVIEW_X1 = 312
PREVIEW_X2 = 450

# Preview especifico para seleccionar el Spectralon.
# Si el preview del Spectralon se ve mal, ajusta estas columnas sin tocar
# el preview de la muestra.
SPECTRALON_PREVIEW_X1 = 312
SPECTRALON_PREVIEW_X2 = 450

# Ranges diagnosticos para ver donde aparece mejor el Spectralon reconstruido.
DIAGNOSTIC_PREVIEW_RANGES = [
    (120, 160),
    (160, 200),
    (200, 240),
    (240, 280),
    (280, 320),
    (320, 340),
]

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

    # raw_crop: (y_slit, lambda_pixel, frame)
    # cube: (y_slit, x_scan, lambda_pixel)
    cube = np.transpose(raw_crop, (0, 2, 1))

    if FLIP_SLIT_AXIS:
        cube = np.flip(cube, axis=0)
    if FLIP_SCAN_AXIS:
        cube = np.flip(cube, axis=1)
    if FLIP_LAMBDA_AXIS:
        cube = np.flip(cube, axis=2)

    print(f"{path.name} cube shape:", cube.shape)
    return cube


def get_preview(cube: np.ndarray, x1: int = PREVIEW_X1, x2: int = PREVIEW_X2) -> np.ndarray:
    i1 = x1 - SPECTRAL_X1
    i2 = x2 - SPECTRAL_X1

    if i1 < 0 or i2 > cube.shape[2] or i1 >= i2:
        raise ValueError(f"Preview {x1}:{x2} fuera del recorte espectral.")

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

    # ginput devuelve: horizontal=x_scan, vertical=y_slit
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


def white_signature_from_roi(white_cube: np.ndarray, white_mask: np.ndarray) -> np.ndarray:
    if white_mask.shape != white_cube.shape[:2]:
        raise ValueError(f"Mask shape {white_mask.shape} no coincide con cubo {white_cube.shape[:2]}")

    pixels = white_cube[white_mask, :]
    if pixels.shape[0] == 0:
        raise ValueError("La ROI del Spectralon no contiene pixeles.")

    if WHITE_REDUCTION == "median":
        signature = np.median(pixels, axis=0)
    elif WHITE_REDUCTION == "mean":
        signature = np.mean(pixels, axis=0)
    else:
        raise ValueError("WHITE_REDUCTION debe ser 'median' o 'mean'")

    return signature.astype(np.float32)


def normalize_cube_by_signature(cube: np.ndarray, white_signature: np.ndarray) -> np.ndarray:
    if cube.shape[2] != white_signature.size:
        raise ValueError(f"Bandas incompatibles: cubo={cube.shape[2]}, blanco={white_signature.size}")

    eps = max(float(np.percentile(np.abs(white_signature), 1)) * 0.01, 1e-6)
    denominator = np.maximum(white_signature, eps)
    return (cube / denominator[None, None, :]).astype(np.float32)


def save_preview(path: Path, cube: np.ndarray, title: str, x1: int = PREVIEW_X1, x2: int = PREVIEW_X2) -> None:
    preview = get_preview(cube, x1=x1, x2=x2)
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


def save_raw_frame_diagnostic(path: Path, raw_path: Path, frame_index: int | None = None) -> None:
    raw = load_npy_checked(raw_path)
    if frame_index is None:
        frame_index = raw.shape[2] // 2

    frame_index = int(np.clip(frame_index, 0, raw.shape[2] - 1))
    frame = raw[:, :, frame_index]
    vmin, vmax = robust_limits(frame, low=1, high=99.8)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(frame, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
    ax.axhline(SLIT_Y1, color="red", linestyle="--", linewidth=1)
    ax.axhline(SLIT_Y2, color="red", linestyle="--", linewidth=1)
    ax.axvline(SPECTRAL_X1, color="lime", linestyle="--", linewidth=1)
    ax.axvline(SPECTRAL_X2, color="lime", linestyle="--", linewidth=1)
    ax.axvline(SPECTRALON_PREVIEW_X1, color="cyan", linestyle="--", linewidth=1)
    ax.axvline(SPECTRALON_PREVIEW_X2, color="cyan", linestyle="--", linewidth=1)
    ax.set_title(f"Frame raw Spectralon {frame_index}: rojo=slit, verde=rango espectral, cyan=preview")
    ax.set_xlabel("x_sensor / lambda_pixel")
    ax.set_ylabel("y_sensor")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()


def save_spectralon_diagnostics(out_dir: Path, spectralon_cube: np.ndarray) -> None:
    diag_dir = out_dir / "diagnosticos_spectralon"
    diag_dir.mkdir(parents=True, exist_ok=True)

    for x1, x2 in DIAGNOSTIC_PREVIEW_RANGES:
        try:
            preview = get_preview(spectralon_cube, x1=x1, x2=x2)
        except ValueError:
            continue

        vmin, vmax = robust_limits(preview)
        out = diag_dir / f"preview_spectralon_cols_{x1}_{x2}.png"
        plt.imsave(out, preview, cmap="gray", vmin=vmin, vmax=vmax)
        np.save(out.with_suffix(".npy"), preview.astype(np.float32))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sample_cube = reconstruct_cube(SAMPLE_NPY)
    spectralon_cube = reconstruct_cube(SPECTRALON_NPY)

    save_raw_frame_diagnostic(OUTPUT_DIR / "diagnostico_frame_raw_spectralon.png", SPECTRALON_NPY)
    save_spectralon_diagnostics(OUTPUT_DIR, spectralon_cube)

    spectralon_preview = get_preview(
        spectralon_cube,
        x1=SPECTRALON_PREVIEW_X1,
        x2=SPECTRALON_PREVIEW_X2,
    )
    spectralon_mask = select_ellipse_mask(spectralon_preview, "ROI Spectralon / blanco")
    white_signature = white_signature_from_roi(spectralon_cube, spectralon_mask)

    print("White signature shape:", white_signature.shape)
    print("White signature min/max:", float(np.min(white_signature)), float(np.max(white_signature)))

    cube_normalized = normalize_cube_by_signature(sample_cube, white_signature)

    print("Cubo normalizado shape:", cube_normalized.shape)
    print("Cubo normalizado min/max:", float(np.nanmin(cube_normalized)), float(np.nanmax(cube_normalized)))

    spectral_cols = np.arange(SPECTRAL_X1, SPECTRAL_X2, dtype=np.int32)
    if FLIP_LAMBDA_AXIS:
        spectral_cols = spectral_cols[::-1]

    wavelengths_nm_approx = np.linspace(400, 1700, cube_normalized.shape[2], dtype=np.float32)

    np.save(OUTPUT_DIR / "cube_y_xscan_lambda_normalized_by_spectralon.npy", cube_normalized)
    np.save(OUTPUT_DIR / "white_signature_spectralon.npy", white_signature.reshape(-1, 1))
    np.save(OUTPUT_DIR / "white_mask_spectralon.npy", spectralon_mask)
    np.save(OUTPUT_DIR / "spectral_pixel_columns.npy", spectral_cols.reshape(-1, 1))
    np.save(OUTPUT_DIR / "wavelengths_nm_approx.npy", wavelengths_nm_approx.reshape(-1, 1))

    save_preview(
        OUTPUT_DIR / "preview_spectralon.png",
        spectralon_cube,
        f"Preview cubo Spectralon columnas {SPECTRALON_PREVIEW_X1}:{SPECTRALON_PREVIEW_X2}",
        x1=SPECTRALON_PREVIEW_X1,
        x2=SPECTRALON_PREVIEW_X2,
    )
    save_preview(
        OUTPUT_DIR / "preview_normalized_by_spectralon.png",
        cube_normalized,
        "Preview cubo muestra normalizado por Spectralon",
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(spectral_cols, white_signature)
    ax.set_xlabel("Columna espectral del sensor")
    ax.set_ylabel("Intensidad Spectralon")
    ax.set_title("Firma blanca del Spectralon")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "white_signature_spectralon.png", dpi=150)
    plt.show()

    print("Guardado en:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
