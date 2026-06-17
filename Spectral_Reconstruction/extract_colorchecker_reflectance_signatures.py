from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath


# =========================
# CONFIGURACION POR DEFECTO
# =========================

COLORCHECKER_NPY = Path(r"Colorckeher\Cube_colorcheker1.npy")
WHITE_NPY = Path(r"Colorckeher\Cube_blancoSpectralon.npy")
BLACK_NPY = Path(r"Colorckeher\Cube_negro.npy")
OUTPUT_DIR = Path(r"firmas_colorchecker_reflectancia")

# raw.shape = (y_sensor, x_sensor, frames)
SLIT_Y1 = 300
SLIT_Y2 = 850
SPECTRAL_X1 = 650
SPECTRAL_X2 = 800

FLIP_SCAN_AXIS = True
FLIP_SLIT_AXIS = False
FLIP_LAMBDA_AXIS = False

PREVIEW_LOW_PERCENTILE = 1
PREVIEW_HIGH_PERCENTILE = 99
PREVIEW_MODE = "mean"  # "mean", "max" o "single"
PREVIEW_X1 = 650
PREVIEW_X2 = 800
PREVIEW_COL = 624
SIGNATURE_REDUCTION = "median"  # "median" o "mean"
DENOMINATOR_EPS = 1e-6


def load_npy_checked(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Archivo vacio: {path}")
    return np.load(path, mmap_mode="r")


def robust_limits(img: np.ndarray, low: float = 1, high: float = 99) -> tuple[float, float]:
    finite = np.isfinite(img)
    if not np.any(finite):
        return 0.0, 1.0
    vmin, vmax = np.percentile(img[finite], [low, high])
    if vmax <= vmin:
        vmin = float(np.nanmin(img))
        vmax = float(np.nanmax(img))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def validate_raw_cube(raw: np.ndarray, name: str) -> None:
    if raw.ndim != 3:
        raise ValueError(f"{name} debe ser 3D (y_sensor, x_sensor, frames), llego {raw.shape}")
    if not (0 <= SLIT_Y1 < SLIT_Y2 <= raw.shape[0]):
        raise ValueError(f"Recorte SLIT_Y invalido para {name}: {SLIT_Y1}:{SLIT_Y2}")
    if not (0 <= SPECTRAL_X1 < SPECTRAL_X2 <= raw.shape[1]):
        raise ValueError(f"Recorte SPECTRAL_X invalido para {name}: {SPECTRAL_X1}:{SPECTRAL_X2}")


def print_frame_note(color_raw: np.ndarray, white_raw: np.ndarray, black_raw: np.ndarray) -> None:
    frame_counts = {
        "ColorChecker": int(color_raw.shape[2]),
        "Spectralon": int(white_raw.shape[2]),
        "Negro": int(black_raw.shape[2]),
    }
    unique_counts = set(frame_counts.values())
    if len(unique_counts) > 1:
        print(
            "\nNota: los cubos tienen diferente numero de frames. "
            "Esto es valido para extraer firmas: cada ROI se promedia dentro "
            "de su propio cubo y todas las firmas quedan alineadas por bandas espectrales."
        )
        for name, count in frame_counts.items():
            print(f"  {name}: {count} frames")


def orient_spatial_image(img_y_frames: np.ndarray) -> np.ndarray:
    """Aplica los flips para mostrar y enmascarar como el cubo reconstruido."""
    img = img_y_frames
    if FLIP_SLIT_AXIS:
        img = np.flip(img, axis=0)
    if FLIP_SCAN_AXIS:
        img = np.flip(img, axis=1)
    return img


def undo_mask_orientation(mask: np.ndarray) -> np.ndarray:
    """Devuelve la mascara desde coordenadas visuales a coordenadas raw crop."""
    raw_mask = mask
    if FLIP_SCAN_AXIS:
        raw_mask = np.flip(raw_mask, axis=1)
    if FLIP_SLIT_AXIS:
        raw_mask = np.flip(raw_mask, axis=0)
    return raw_mask


def make_preview(
    raw: np.ndarray,
    preview_mode: str = PREVIEW_MODE,
    preview_x1: int = PREVIEW_X1,
    preview_x2: int = PREVIEW_X2,
    preview_col: int = PREVIEW_COL,
) -> np.ndarray:
    """Crea una vista espacial (y_crop, frames) para dibujar ROIs."""
    if preview_mode == "single":
        if not (0 <= preview_col < raw.shape[1]):
            raise ValueError(f"PREVIEW_COL fuera del ancho del cubo: {preview_col}")
        preview = raw[SLIT_Y1:SLIT_Y2, preview_col, :]
    else:
        if not (0 <= preview_x1 < preview_x2 <= raw.shape[1]):
            raise ValueError(f"Rango preview invalido: {preview_x1}:{preview_x2}")
        crop = raw[SLIT_Y1:SLIT_Y2, preview_x1:preview_x2, :]
        if preview_mode == "mean":
            preview = crop.mean(axis=1)
        elif preview_mode == "max":
            preview = crop.max(axis=1)
        else:
            raise ValueError("--preview-mode debe ser 'mean', 'max' o 'single'")
    return orient_spatial_image(preview.astype(np.float32, copy=False))


def polygon_to_mask(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if points.shape[0] < 3:
        raise ValueError("La ROI necesita al menos 3 puntos.")
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    mask = MplPath(points).contains_points(coords).reshape(shape)
    if not np.any(mask):
        raise ValueError("La mascara quedo vacia. Selecciona un area mas grande.")
    return mask


def select_polygon_mask(preview: np.ndarray, title: str, color: str = "yellow") -> tuple[np.ndarray, np.ndarray]:
    vmin, vmax = robust_limits(preview, PREVIEW_LOW_PERCENTILE, PREVIEW_HIGH_PERCENTILE)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(preview, cmap="turbo", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title + "\nHaz clics alrededor de la ROI y presiona ENTER para cerrar")
    ax.set_xlabel("x_scan / frames")
    ax.set_ylabel("y_slit")
    fig.tight_layout()

    points = plt.ginput(n=-1, timeout=0)
    if len(points) < 3:
        plt.close(fig)
        raise RuntimeError("ROI cancelada o incompleta.")

    pts = np.asarray(points, dtype=np.float32)
    closed = np.vstack([pts, pts[0]])
    ax.plot(closed[:, 0], closed[:, 1], "-", color=color, linewidth=1.8)
    ax.text(
        float(np.mean(pts[:, 0])),
        float(np.mean(pts[:, 1])),
        str(int(len(points))),
        color=color,
        fontweight="bold",
        ha="center",
        bbox={"facecolor": "black", "alpha": 0.45, "pad": 2},
    )
    fig.canvas.draw_idle()
    plt.pause(0.35)
    plt.close(fig)

    mask_visual = polygon_to_mask(pts, preview.shape)
    return undo_mask_orientation(mask_visual), pts


def signature_from_mask(raw: np.ndarray, mask_raw_crop: np.ndarray) -> np.ndarray:
    """Calcula firma media/mediana sin construir todo el cubo reconstruido."""
    y_count = SLIT_Y2 - SLIT_Y1
    bands = SPECTRAL_X2 - SPECTRAL_X1
    if mask_raw_crop.shape[0] != y_count or mask_raw_crop.shape[1] != raw.shape[2]:
        raise ValueError(
            f"Mask shape {mask_raw_crop.shape} no coincide con "
            f"(y_crop, frames)=({y_count}, {raw.shape[2]})"
        )

    signature = np.empty(bands, dtype=np.float32)
    raw_y = raw[SLIT_Y1:SLIT_Y2, :, :]

    for i, sensor_x in enumerate(range(SPECTRAL_X1, SPECTRAL_X2)):
        plane = raw_y[:, sensor_x, :]
        values = plane[mask_raw_crop].astype(np.float32, copy=False)
        if SIGNATURE_REDUCTION == "median":
            signature[i] = np.nanmedian(values)
        elif SIGNATURE_REDUCTION == "mean":
            signature[i] = np.nanmean(values)
        else:
            raise ValueError("SIGNATURE_REDUCTION debe ser 'median' o 'mean'")

    if FLIP_LAMBDA_AXIS:
        signature = signature[::-1]
    return signature


def ask_patch_names(num_patches: int | None) -> list[str]:
    if num_patches is None:
        raw = input("Cuantas ROIs/parches del ColorChecker quieres seleccionar? [24]: ").strip()
        num_patches = 24 if raw == "" else int(raw)
    if num_patches <= 0:
        raise ValueError("El numero de parches debe ser mayor que cero.")
    return [f"patch_{i:02d}" for i in range(1, num_patches + 1)]


def save_signatures_csv(
    path: Path,
    spectral_cols: np.ndarray,
    wavelengths_nm: np.ndarray,
    white: np.ndarray,
    black: np.ndarray,
    raw_signatures: dict[str, np.ndarray],
    reflectance_signatures: dict[str, np.ndarray],
) -> None:
    fieldnames = ["band_index", "sensor_x", "wavelength_nm", "white_raw", "black_raw"]
    for name in raw_signatures:
        fieldnames.append(f"{name}_raw")
        fieldnames.append(f"{name}_reflectance")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(spectral_cols.size):
            row = {
                "band_index": i,
                "sensor_x": int(spectral_cols[i]),
                "wavelength_nm": float(wavelengths_nm[i]),
                "white_raw": float(white[i]),
                "black_raw": float(black[i]),
            }
            for name in raw_signatures:
                row[f"{name}_raw"] = float(raw_signatures[name][i])
                row[f"{name}_reflectance"] = float(reflectance_signatures[name][i])
            writer.writerow(row)


def plot_signatures(
    out_dir: Path,
    wavelengths_nm: np.ndarray,
    white: np.ndarray,
    black: np.ndarray,
    raw_signatures: dict[str, np.ndarray],
    reflectance_signatures: dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(wavelengths_nm, white, color="black", linewidth=1.8, label="white / Spectralon")
    ax.plot(wavelengths_nm, black, color="tab:blue", linewidth=1.8, label="black")
    for name, sig in raw_signatures.items():
        ax.plot(wavelengths_nm, sig, linewidth=1.2, alpha=0.8, label=name)
    ax.set_title("Firmas crudas")
    ax.set_xlabel("Longitud de onda aprox. (nm)")
    ax.set_ylabel("Intensidad")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "firmas_crudas_colorchecker.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for name, sig in reflectance_signatures.items():
        ax.plot(wavelengths_nm, sig, linewidth=1.4, label=name)
    ax.set_title("Reflectancia ColorChecker corregida")
    ax.set_xlabel("Longitud de onda aprox. (nm)")
    ax.set_ylabel("(I_color - I_negro) / (I_blanco - I_negro)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncols=2, fontsize=8)
    finite = np.concatenate([v[np.isfinite(v)] for v in reflectance_signatures.values()])
    if finite.size:
        lo, hi = np.percentile(finite, [1, 99])
        if hi > lo:
            ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    fig.tight_layout()
    fig.savefig(out_dir / "reflectancia_colorchecker.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Selecciona ROIs en ColorChecker, Spectralon y negro para extraer firmas de reflectancia."
    )
    parser.add_argument("--colorchecker-npy", type=Path, default=COLORCHECKER_NPY)
    parser.add_argument("--white-npy", type=Path, default=WHITE_NPY)
    parser.add_argument("--black-npy", type=Path, default=BLACK_NPY)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--num-patches", type=int, default=None)
    parser.add_argument("--start-nm", type=float, default=400.0)
    parser.add_argument("--end-nm", type=float, default=1700.0)
    parser.add_argument(
        "--preview-mode",
        choices=["mean", "max", "single"],
        default=PREVIEW_MODE,
        help="Como formar la imagen para seleccionar ROIs.",
    )
    parser.add_argument("--preview-x1", type=int, default=PREVIEW_X1)
    parser.add_argument("--preview-x2", type=int, default=PREVIEW_X2)
    parser.add_argument("--preview-col", type=int, default=PREVIEW_COL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    color_raw = load_npy_checked(args.colorchecker_npy)
    white_raw = load_npy_checked(args.white_npy)
    black_raw = load_npy_checked(args.black_npy)

    validate_raw_cube(color_raw, "ColorChecker")
    validate_raw_cube(white_raw, "Spectralon")
    validate_raw_cube(black_raw, "Negro")

    print("ColorChecker:", color_raw.shape, color_raw.dtype)
    print("Spectralon:", white_raw.shape, white_raw.dtype)
    print("Negro:", black_raw.shape, black_raw.dtype)
    print_frame_note(color_raw, white_raw, black_raw)
    print("Recorte y:", SLIT_Y1, SLIT_Y2)
    print("Recorte espectral x:", SPECTRAL_X1, SPECTRAL_X2)
    if args.preview_mode == "single":
        print("Vista ROI: columna espectral", args.preview_col)
    else:
        print("Vista ROI:", args.preview_mode, "en x", args.preview_x1, args.preview_x2)

    print("\nSelecciona ROI del blanco/Spectralon.")
    white_preview = make_preview(
        white_raw,
        args.preview_mode,
        args.preview_x1,
        args.preview_x2,
        args.preview_col,
    )
    white_mask, white_roi = select_polygon_mask(white_preview, "ROI blanco / Spectralon", color="white")
    white_signature = signature_from_mask(white_raw, white_mask)
    print("Pixeles ROI blanco:", int(np.count_nonzero(white_mask)))

    print("\nSelecciona ROI del negro.")
    black_preview = make_preview(
        black_raw,
        args.preview_mode,
        args.preview_x1,
        args.preview_x2,
        args.preview_col,
    )
    black_mask, black_roi = select_polygon_mask(black_preview, "ROI negro", color="black")
    black_signature = signature_from_mask(black_raw, black_mask)
    print("Pixeles ROI negro:", int(np.count_nonzero(black_mask)))

    patch_names = ask_patch_names(args.num_patches)
    color_preview = make_preview(
        color_raw,
        args.preview_mode,
        args.preview_x1,
        args.preview_x2,
        args.preview_col,
    )
    raw_signatures: dict[str, np.ndarray] = {}
    patch_rois: dict[str, np.ndarray] = {}
    patch_pixel_counts: dict[str, int] = {}

    for name in patch_names:
        print(f"\nSelecciona ROI del ColorChecker: {name}")
        mask, roi = select_polygon_mask(color_preview, f"ROI ColorChecker: {name}", color="yellow")
        raw_signatures[name] = signature_from_mask(color_raw, mask)
        patch_rois[name] = roi
        patch_pixel_counts[name] = int(np.count_nonzero(mask))
        print(f"Pixeles ROI {name}:", patch_pixel_counts[name])

    denominator = white_signature - black_signature
    denominator[np.abs(denominator) < DENOMINATOR_EPS] = np.nan

    reflectance_signatures = {
        name: (sig - black_signature) / denominator
        for name, sig in raw_signatures.items()
    }

    spectral_cols = np.arange(SPECTRAL_X1, SPECTRAL_X2, dtype=np.int32)
    if FLIP_LAMBDA_AXIS:
        spectral_cols = spectral_cols[::-1]
    wavelengths_nm = np.linspace(args.start_nm, args.end_nm, spectral_cols.size, dtype=np.float32)

    np.save(args.output_dir / "white_signature_raw.npy", white_signature.reshape(-1, 1))
    np.save(args.output_dir / "black_signature_raw.npy", black_signature.reshape(-1, 1))
    np.save(args.output_dir / "colorchecker_raw_signatures.npy", raw_signatures)
    np.save(args.output_dir / "colorchecker_reflectance_signatures.npy", reflectance_signatures)
    np.save(args.output_dir / "spectral_pixel_columns.npy", spectral_cols.reshape(-1, 1))
    np.save(args.output_dir / "wavelengths_nm_approx.npy", wavelengths_nm.reshape(-1, 1))

    np.save(
        args.output_dir / "roi_metadata.npy",
        {
            "colorchecker_npy": str(args.colorchecker_npy),
            "white_npy": str(args.white_npy),
            "black_npy": str(args.black_npy),
            "colorchecker_shape": tuple(int(v) for v in color_raw.shape),
            "white_shape": tuple(int(v) for v in white_raw.shape),
            "black_shape": tuple(int(v) for v in black_raw.shape),
            "slit_y_crop": [SLIT_Y1, SLIT_Y2],
            "spectral_x_crop": [SPECTRAL_X1, SPECTRAL_X2],
            "preview_mode": args.preview_mode,
            "preview_x_crop": [args.preview_x1, args.preview_x2],
            "preview_col": args.preview_col,
            "white_roi_points_visual": white_roi,
            "black_roi_points_visual": black_roi,
            "patch_roi_points_visual": patch_rois,
            "patch_pixel_counts": patch_pixel_counts,
            "signature_reduction": SIGNATURE_REDUCTION,
            "reflectance_formula": "(I_patch - I_black) / (I_white - I_black)",
        },
    )

    save_signatures_csv(
        args.output_dir / "colorchecker_signatures_reflectance.csv",
        spectral_cols,
        wavelengths_nm,
        white_signature,
        black_signature,
        raw_signatures,
        reflectance_signatures,
    )
    plot_signatures(
        args.output_dir,
        wavelengths_nm,
        white_signature,
        black_signature,
        raw_signatures,
        reflectance_signatures,
    )

    print("\nListo.")
    print("Salida:", args.output_dir)
    print("CSV:", args.output_dir / "colorchecker_signatures_reflectance.csv")
    print("Grafica reflectancia:", args.output_dir / "reflectancia_colorchecker.png")


if __name__ == "__main__":
    main()
