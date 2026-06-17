from __future__ import annotations

import argparse
import re
from io import BytesIO
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from characterize_spectral_curve import build_results_dataframe, ensure_2d_image


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR / "Barrido_monocromador"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "Characterize_results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Genera GIF/CSV de avance por bloque usando exactamente los puntos "
            "selected_x calculados por characterize_spectral_curve.py."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Carpeta raiz con bloques B1_..., B2_..., etc.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "spectral_progress_gifs",
        help="Carpeta donde se guardan los GIF y CSV generados.",
    )
    parser.add_argument(
        "--block",
        type=str,
        default=None,
        help="Bloque a visualizar, por ejemplo B5. Si no se indica, procesa todos.",
    )
    parser.add_argument("--poly-degree", type=int, default=3, help="Mismo grado usado en la caracterizacion.")
    parser.add_argument("--fps", type=float, default=6.0, help="Cuadros por segundo del GIF.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limita frames por bloque para pruebas rapidas.")
    parser.add_argument("--frame-step", type=int, default=1, help="Usa 1 de cada N imagenes para aligerar el GIF.")
    parser.add_argument(
        "--roi-half-width",
        type=int,
        default=45,
        help="Semiancho alrededor de selected_x solo para ubicar visualmente la banda Y.",
    )
    parser.add_argument(
        "--band-half-height",
        type=int,
        default=32,
        help="Semialto de la banda horizontal mostrada en la imagen.",
    )
    return parser


def normalize_for_display(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0

    vmin, vmax = np.percentile(finite, [2.0, 99.8])
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def estimate_visual_band(
    image: np.ndarray,
    selected_x: float,
    roi_half_width: int,
    band_half_height: int,
) -> dict[str, float]:
    image_2d = ensure_2d_image(image)
    height, width = image_2d.shape
    x_center = int(np.clip(round(selected_x), 0, width - 1))
    x0 = max(0, x_center - roi_half_width)
    x1 = min(width, x_center + roi_half_width + 1)
    roi_x = image_2d[:, x0:x1]

    corrected = np.maximum(roi_x - np.percentile(roi_x, 20.0, axis=1, keepdims=True), 0.0)
    row_profile = corrected.sum(axis=1)
    y_center = int(np.argmax(row_profile)) if row_profile.size else height // 2
    y0 = max(0, y_center - band_half_height)
    y1 = min(height, y_center + band_half_height + 1)

    roi = image_2d[y0:y1, x0:x1]
    if roi.size == 0:
        return {
            "band_y0": float(y0),
            "band_y1": float(y1),
            "local_peak_x": float(selected_x),
            "local_peak_y": float(y_center),
        }

    max_y, max_x = np.unravel_index(int(np.argmax(roi)), roi.shape)
    return {
        "band_y0": float(y0),
        "band_y1": float(y1 - 1),
        "local_peak_x": float(x0 + max_x),
        "local_peak_y": float(y0 + max_y),
    }


def render_frame(
    entry: dict[str, object],
    wavelengths_nm: np.ndarray,
    displacements_px: np.ndarray,
    frame_idx: int,
    total_frames: int,
) -> np.ndarray:
    fig, (ax_img, ax_plot) = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.2),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    image_2d = ensure_2d_image(np.load(Path(str(entry["path"]))), str(entry["path"]))
    vmin, vmax = normalize_for_display(image_2d)
    ax_img.imshow(image_2d, cmap="inferno", aspect="auto", vmin=vmin, vmax=vmax)

    selected_x = float(entry["selected_x"])
    ax_img.axvline(selected_x, color="lime", linestyle="-", linewidth=1.4, label="selected_x")
    ax_img.axhline(float(entry["band_y0"]), color="white", linestyle=":", linewidth=1.0, label="Banda visual")
    ax_img.axhline(float(entry["band_y1"]), color="white", linestyle=":", linewidth=1.0)
    ax_img.scatter(
        [float(entry["local_peak_x"])],
        [float(entry["local_peak_y"])],
        c=["cyan"],
        s=[18],
        linewidths=0,
        label="Max local",
    )
    ax_img.set_title(
        f"Imagen {frame_idx + 1}/{total_frames}\n"
        f"{float(entry['wavelength_nm']):.0f} nm | desplazamiento = {float(entry['displacement_px']):.2f} px"
    )
    ax_img.set_xlabel("Pixel X")
    ax_img.set_ylabel("Pixel Y")
    ax_img.legend(loc="upper right", framealpha=0.78)

    ax_plot.plot(wavelengths_nm, displacements_px, color="0.78", linewidth=1.0)
    ax_plot.plot(
        wavelengths_nm[: frame_idx + 1],
        displacements_px[: frame_idx + 1],
        color="tab:blue",
        marker="o",
        markersize=3.5,
        linewidth=1.4,
    )
    ax_plot.scatter([wavelengths_nm[frame_idx]], [displacements_px[frame_idx]], color="crimson", s=44)
    ax_plot.set_title("Dispersion espectral acumulada")
    ax_plot.set_xlabel("Longitud de onda [nm]")
    ax_plot.set_ylabel("Desplazamiento [px]")
    ax_plot.grid(True, alpha=0.3)
    ax_plot.set_xlim(float(np.min(wavelengths_nm)), float(np.max(wavelengths_nm)))

    y_span = float(np.ptp(displacements_px)) if displacements_px.size > 1 else 1.0
    y_margin = max(1.0, 0.06 * y_span)
    ax_plot.set_ylim(float(np.min(displacements_px)) - y_margin, float(np.max(displacements_px)) + y_margin)

    fig.suptitle(Path(str(entry["path"])).parent.name, fontsize=12)
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return imageio.imread(buffer)


def save_progress_csv(entries: list[dict[str, object]], output_csv: Path) -> None:
    columns = [
        "exposure_folder",
        "block",
        "filename",
        "wavelength_nm",
        "raw_x",
        "selected_x",
        "displacement_px",
        "fit_x",
        "residual_px",
        "fit_inlier",
        "peak_score",
        "adjusted_peak_score",
        "peak_prominence",
        "peak_width_px",
        "band_y0",
        "band_y1",
        "local_peak_x",
        "local_peak_y",
        "path",
    ]
    pd.DataFrame(entries)[columns].to_csv(output_csv, index=False)


def output_stem(entries: list[dict[str, object]]) -> str:
    first_path = Path(str(entries[0]["path"]))
    exposure = str(entries[0]["exposure_folder"])
    stem = first_path.parent.name
    if exposure != first_path.parent.parent.name:
        stem = f"{exposure}_{stem}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def save_block_progress(
    block_df: pd.DataFrame,
    output_dir: Path,
    fps: float,
    max_frames: int | None,
    frame_step: int,
    roi_half_width: int,
    band_half_height: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = block_df.sort_values("wavelength_nm", kind="stable").reset_index(drop=True)

    frame_step = max(1, int(frame_step))
    if frame_step > 1:
        ordered = ordered.iloc[::frame_step].reset_index(drop=True)
    if max_frames is not None:
        ordered = ordered.iloc[:max_frames].reset_index(drop=True)
    if ordered.empty:
        raise ValueError("No hay datos para generar el GIF.")

    entries = []
    for row in ordered.to_dict(orient="records"):
        image = ensure_2d_image(np.load(Path(str(row["path"]))), str(row["path"]))
        row.update(
            estimate_visual_band(
                image,
                selected_x=float(row["selected_x"]),
                roi_half_width=roi_half_width,
                band_half_height=band_half_height,
            )
        )
        entries.append(row)

    x0 = float(entries[0]["selected_x"])
    for entry in entries:
        entry["displacement_px"] = float(entry["selected_x"]) - x0

    wavelengths_nm = np.array([float(item["wavelength_nm"]) for item in entries], dtype=float)
    displacements_px = np.array([float(item["displacement_px"]) for item in entries], dtype=float)

    stem = output_stem(entries)
    output_gif = output_dir / f"{stem}_progress.gif"
    output_csv = output_dir / f"{stem}_progress.csv"
    save_progress_csv(entries, output_csv)

    frames = [
        render_frame(entry, wavelengths_nm, displacements_px, idx, len(entries))
        for idx, entry in enumerate(entries)
    ]
    duration_s = 1.0 / fps if fps > 0 else 0.2
    imageio.mimsave(output_gif, frames, duration=duration_s, loop=0)
    return output_gif, output_csv


def main() -> None:
    args = build_parser().parse_args()
    df, _block_coeffs = build_results_dataframe(args.root, degree=args.poly_degree, max_files=None)
    if df.empty:
        raise SystemExit(f"No se encontraron archivos .npy procesables dentro de {args.root}")

    if args.block:
        block_name = args.block.upper()
        df = df[df["block"].str.upper() == block_name]
        if df.empty:
            raise SystemExit(f"No se encontraron datos para el bloque {block_name}.")

    outputs = []
    for (_exposure, _block), block_df in df.groupby(["exposure_folder", "block"], sort=False):
        outputs.append(
            save_block_progress(
                block_df=block_df,
                output_dir=args.output_dir,
                fps=args.fps,
                max_frames=args.max_frames,
                frame_step=args.frame_step,
                roi_half_width=args.roi_half_width,
                band_half_height=args.band_half_height,
            )
        )

    for gif_path, csv_path in outputs:
        print(f"GIF generado: {gif_path}")
        print(f"CSV generado: {csv_path}")


if __name__ == "__main__":
    main()
