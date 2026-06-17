from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR / "Barrido_monocromador"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "Characterize_results"

MPL_CONFIG_DIR = SCRIPT_DIR / ".mplconfig"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR.resolve()))

import matplotlib.pyplot as plt

try:
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks, peak_widths
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Este script requiere scipy. Instala con: python -m pip install scipy") from exc


BLOCK_CONFIG = {
    "B1": {"start_nm": 400, "end_nm": 410, "step_nm": 2},
    "B2": {"start_nm": 411, "end_nm": 450, "step_nm": 2},
    "B3": {"start_nm": 451, "end_nm": 500, "step_nm": 2},
    "B4": {"start_nm": 501, "end_nm": 700, "step_nm": 2},
    "B5": {"start_nm": 701, "end_nm": 1000, "step_nm": 2},
    "B6": {"start_nm": 1001, "end_nm": 1300, "step_nm": 2},
    "B7": {"start_nm": 1301, "end_nm": 1700, "step_nm": 2},
}

BLOCK_DIR_PATTERN = re.compile(r"^(B\d+)_")
WAVELENGTH_PATTERN = re.compile(r"(?<!\d)(\d{3,4})\s*nm(?!\d)", re.IGNORECASE)
LEADING_INDEX_PATTERN = re.compile(r"^(\d+)")


@dataclass(frozen=True)
class CalibrationFile:
    path: Path
    exposure_folder: str
    block: str
    block_dir: Path
    wavelength_nm: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibra la curva espectral del sistema con barridos .npy y puede "
            "reconstruir arreglos usando esa calibracion."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    add_calibrate_args(parser)
    calibrate = subparsers.add_parser("calibrate", help="Genera la curva de calibracion espectral.")
    add_calibrate_args(calibrate)

    reconstruct = subparsers.add_parser(
        "reconstruct",
        help="Interpola el eje de dispersion de un .npy hacia un eje en nm.",
    )
    reconstruct.add_argument("--input-npy", type=Path, required=True, help="Imagen/cubo de entrada.")
    reconstruct.add_argument("--output-npy", type=Path, required=True, help="Cubo reconstruido de salida.")
    reconstruct.add_argument(
        "--calibration-npz",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "spectral_calibration_model.npz",
        help="Modelo .npz generado por calibrate.",
    )
    reconstruct.add_argument("--start-nm", type=float, default=None, help="Inicio del eje espectral de salida.")
    reconstruct.add_argument("--end-nm", type=float, default=None, help="Fin del eje espectral de salida.")
    reconstruct.add_argument("--step-nm", type=float, default=2.0, help="Paso del eje espectral de salida.")
    reconstruct.add_argument(
        "--spectral-axis",
        type=int,
        default=-1,
        help="Eje de pixeles de dispersion en el .npy de entrada. Por defecto: ultimo eje.",
    )
    return parser


def add_calibrate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Carpeta raiz con bloques B1_..., B2_..., etc.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "spectral_characterization.csv",
        help="CSV consolidado con puntos detectados y ajuste.",
    )
    parser.add_argument(
        "--calibration-npz",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "spectral_calibration_model.npz",
        help="Modelo de calibracion reutilizable para reconstruccion.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "spectral_characterization_plots",
        help="Carpeta de graficas de diagnostico.",
    )
    parser.add_argument("--poly-degree", type=int, default=3, help="Grado del polinomio x=f(lambda).")
    parser.add_argument("--max-files", type=int, default=None, help="Limita archivos para pruebas rapidas.")
    parser.add_argument("--show", action="store_true", help="Muestra las graficas.")


def is_block_dir(path: Path) -> bool:
    return path.is_dir() and BLOCK_DIR_PATTERN.match(path.name) is not None


def get_block_name(block_dir: Path) -> str:
    match = BLOCK_DIR_PATTERN.match(block_dir.name)
    if match is None:
        raise ValueError(f"No se pudo identificar el bloque en: {block_dir}")
    return match.group(1)


def ensure_2d_image(image: np.ndarray, source: Path | str = "<array>") -> np.ndarray:
    if image.ndim == 3:
        image = image.mean(axis=2)
    if image.ndim != 2:
        raise ValueError(f"Se esperaba una imagen 2D en {source}, pero llego con shape {image.shape}")
    return np.asarray(image, dtype=np.float64)


def discover_exposure_dirs(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta raiz: {root_dir}")

    direct_block_dirs = [path for path in root_dir.iterdir() if is_block_dir(path)]
    if direct_block_dirs:
        return [root_dir]

    return [
        path
        for path in sorted(root_dir.iterdir())
        if path.is_dir() and any(is_block_dir(child) for child in path.iterdir())
    ]


def load_block_metadata(block_dir: Path) -> pd.DataFrame | None:
    metadata_path = block_dir / "metadata.csv"
    if not metadata_path.exists():
        return None

    metadata = pd.read_csv(metadata_path)
    if metadata.empty:
        return None

    metadata.columns = [str(column).strip() for column in metadata.columns]
    return metadata


def build_metadata_lookup(metadata: pd.DataFrame | None) -> dict[str, float]:
    if metadata is None:
        return {}

    filename_column = None
    wavelength_column = None
    for column in metadata.columns:
        lowered = column.lower()
        if lowered in {"filename_npy", "filename", "file", "archivo"}:
            filename_column = column
        if lowered in {"wavelength_nm", "wavelength", "lambda_nm", "nm"}:
            wavelength_column = column

    if filename_column is None or wavelength_column is None:
        return {}

    lookup: dict[str, float] = {}
    for _, row in metadata.iterrows():
        filename = str(row[filename_column]).strip()
        wavelength = pd.to_numeric(row[wavelength_column], errors="coerce")
        if filename and not pd.isna(wavelength):
            lookup[filename] = float(wavelength)
    return lookup


def extract_wavelength_from_filename(filename: str) -> float | None:
    match = WAVELENGTH_PATTERN.search(filename)
    if match is None:
        return None
    return float(match.group(1))


def estimate_order_index(npy_path: Path, fallback_index: int) -> int:
    match = LEADING_INDEX_PATTERN.match(npy_path.stem)
    if match is not None:
        return int(match.group(1))
    return fallback_index


def wavelength_from_block_and_order(block: str, order_index: int) -> float:
    block_cfg = BLOCK_CONFIG[block]
    return float(block_cfg["start_nm"] + block_cfg["step_nm"] * order_index)


def resolve_wavelength_nm(
    npy_path: Path,
    block: str,
    fallback_index: int,
    metadata_lookup: dict[str, float],
) -> float:
    metadata_value = metadata_lookup.get(npy_path.name)
    if metadata_value is not None:
        return metadata_value

    wavelength = extract_wavelength_from_filename(npy_path.name)
    if wavelength is not None:
        return wavelength

    return wavelength_from_block_and_order(block, estimate_order_index(npy_path, fallback_index))


def collect_calibration_files(root_dir: Path, max_files: int | None = None) -> list[CalibrationFile]:
    files: list[CalibrationFile] = []
    for exposure_dir in discover_exposure_dirs(root_dir):
        exposure_folder = exposure_dir.name
        for block_dir in sorted(path for path in exposure_dir.iterdir() if is_block_dir(path)):
            block = get_block_name(block_dir)
            metadata_lookup = build_metadata_lookup(load_block_metadata(block_dir))
            npy_files = sorted(block_dir.glob("*.npy"))
            for fallback_index, npy_path in enumerate(npy_files):
                files.append(
                    CalibrationFile(
                        path=npy_path,
                        exposure_folder=exposure_folder,
                        block=block,
                        block_dir=block_dir,
                        wavelength_nm=resolve_wavelength_nm(
                            npy_path=npy_path,
                            block=block,
                            fallback_index=fallback_index,
                            metadata_lookup=metadata_lookup,
                        ),
                    )
                )

    files = sorted(files, key=lambda item: (item.exposure_folder, item.wavelength_nm, item.path.name))
    if max_files is not None:
        files = files[:max_files]
    return files


def robust_column_profile(image: np.ndarray, background: np.ndarray | None = None) -> np.ndarray:
    img = ensure_2d_image(image)
    if background is not None:
        bg = ensure_2d_image(background)
        if bg.shape != img.shape:
            raise ValueError(f"El fondo tiene shape {bg.shape}, pero la imagen tiene shape {img.shape}")
        img = np.maximum(img - bg, 0.0)

    finite = img[np.isfinite(img)]
    if finite.size == 0:
        raise ValueError("La imagen no tiene valores finitos.")

    lo, hi = np.percentile(finite, [1.0, 99.6])
    if hi <= lo:
        hi = float(np.max(finite))
    clipped = np.clip(img, lo, hi) - lo

    row_background = np.percentile(clipped, 20.0, axis=1, keepdims=True)
    corrected = np.maximum(clipped - row_background, 0.0)

    col_sum = corrected.sum(axis=0)
    col_median = np.median(corrected, axis=0)
    profile = col_sum + corrected.shape[0] * col_median
    profile -= np.percentile(profile, 5.0)
    profile = np.maximum(profile, 0.0)
    return gaussian_filter1d(profile, sigma=3.0, mode="nearest")


def build_block_backgrounds(files: list[CalibrationFile], sample_count: int = 41) -> dict[Path, np.ndarray]:
    backgrounds: dict[Path, np.ndarray] = {}
    by_block: dict[Path, list[CalibrationFile]] = {}
    for item in files:
        by_block.setdefault(item.block_dir, []).append(item)

    for block_dir, block_files in by_block.items():
        ordered = sorted(block_files, key=lambda item: item.wavelength_nm)
        if len(ordered) <= 2:
            continue
        indices = np.linspace(0, len(ordered) - 1, num=min(sample_count, len(ordered)), dtype=int)
        sample = [ensure_2d_image(np.load(ordered[idx].path), ordered[idx].path).astype(np.float32) for idx in indices]
        backgrounds[block_dir] = np.median(np.stack(sample, axis=0), axis=0).astype(np.float32)
    return backgrounds


def detect_candidate_peaks(profile: np.ndarray, max_candidates: int = 8) -> list[dict[str, float]]:
    if profile.size == 0:
        return []

    dynamic = float(np.percentile(profile, 99.0) - np.percentile(profile, 50.0))
    prominence = max(dynamic * 0.08, float(np.max(profile)) * 0.01, 1e-9)
    peaks, properties = find_peaks(profile, distance=8, prominence=prominence)

    if peaks.size == 0:
        peaks = np.array([int(np.argmax(profile))], dtype=int)
        prominences = np.array([float(np.max(profile) - np.median(profile))])
    else:
        prominences = properties.get("prominences", np.zeros_like(peaks, dtype=float))

    widths, _, left_ips, right_ips = peak_widths(profile, peaks, rel_height=0.5)
    candidates = []
    for peak, prom, width, left, right in zip(peaks, prominences, widths, left_ips, right_ips):
        width_score = min(float(width), 25.0) / 25.0
        score = float(profile[peak]) * (0.7 + 0.3 * width_score) + float(prom)
        candidates.append(
            {
                "x": float(peak),
                "score": score,
                "prominence": float(prom),
                "width_px": float(width),
                "left_x": float(left),
                "right_x": float(right),
            }
        )

    return sorted(candidates, key=lambda item: item["score"], reverse=True)[:max_candidates]


def fit_poly_with_clipping(
    wavelengths_nm: np.ndarray,
    x_pixels: np.ndarray,
    degree: int,
    max_iters: int = 8,
    sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(wavelengths_nm) & np.isfinite(x_pixels)
    mask = finite.copy()
    degree = int(min(degree, max(1, mask.sum() - 1)))

    for _ in range(max_iters):
        coeff = np.polyfit(wavelengths_nm[mask], x_pixels[mask], degree)
        residual = x_pixels - np.polyval(coeff, wavelengths_nm)
        mad = np.median(np.abs(residual[mask] - np.median(residual[mask])))
        robust_sigma = 1.4826 * mad if mad > 0 else np.std(residual[mask])
        if robust_sigma <= 0:
            break
        new_mask = finite & (np.abs(residual) <= sigma * robust_sigma)
        if new_mask.sum() <= degree + 1 or np.array_equal(new_mask, mask):
            break
        mask = new_mask

    coeff = np.polyfit(wavelengths_nm[mask], x_pixels[mask], degree)
    return coeff, mask


def add_persistence_penalty(block_rows: list[dict[str, object]]) -> None:
    counts: dict[int, int] = {}
    total = 0
    for row in block_rows:
        candidates = row["candidates"]
        assert isinstance(candidates, list)
        for cand in candidates:
            bucket = int(round(float(cand["x"]) / 3.0))
            counts[bucket] = counts.get(bucket, 0) + 1
            total += 1

    for row in block_rows:
        candidates = row["candidates"]
        assert isinstance(candidates, list)
        for cand in candidates:
            bucket = int(round(float(cand["x"]) / 3.0))
            persistence = counts.get(bucket, 0) / max(total, 1)
            cand["persistence"] = float(persistence)
            cand["adjusted_score"] = float(cand["score"]) / (1.0 + 10.0 * persistence**2)


def choose_smooth_path(block_rows: list[dict[str, object]]) -> list[dict[str, float]]:
    if not block_rows:
        return []

    add_persistence_penalty(block_rows)
    candidates_by_row: list[list[dict[str, float]]] = []
    local_costs: list[np.ndarray] = []
    for row in block_rows:
        candidates = row["candidates"]
        assert isinstance(candidates, list)
        candidates_by_row.append(candidates)
        max_score = max(float(cand["adjusted_score"]) for cand in candidates)
        local_costs.append(np.array([-np.log((float(cand["adjusted_score"]) + 1e-9) / max_score) for cand in candidates]))

    costs = [local_costs[0]]
    parents: list[np.ndarray] = []
    for idx in range(1, len(candidates_by_row)):
        prev_candidates = candidates_by_row[idx - 1]
        curr_candidates = candidates_by_row[idx]
        prev_x = np.array([float(cand["x"]) for cand in prev_candidates])
        curr_x = np.array([float(cand["x"]) for cand in curr_candidates])

        curr_cost = np.empty(len(curr_candidates), dtype=float)
        curr_parent = np.empty(len(curr_candidates), dtype=int)
        for cand_idx, x in enumerate(curr_x):
            jump = np.abs(x - prev_x)
            transition = 0.03 * jump + 0.0015 * np.maximum(jump - 20.0, 0.0) ** 2
            options = costs[-1] + transition
            parent = int(np.argmin(options))
            curr_parent[cand_idx] = parent
            curr_cost[cand_idx] = local_costs[idx][cand_idx] + options[parent]
        costs.append(curr_cost)
        parents.append(curr_parent)

    selected_idx = int(np.argmin(costs[-1]))
    path_indices = [selected_idx]
    for parent in reversed(parents):
        selected_idx = int(parent[selected_idx])
        path_indices.append(selected_idx)
    path_indices.reverse()
    return [candidates[index] for candidates, index in zip(candidates_by_row, path_indices)]


def choose_calibration_points(rows: list[dict[str, object]], degree: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    base_df = pd.DataFrame(rows).sort_values(["exposure_folder", "block", "wavelength_nm"], kind="stable")
    if base_df.empty:
        return base_df, {}

    selected_rows = []
    block_coeffs: dict[str, np.ndarray] = {}

    sorted_rows = sorted(rows, key=lambda item: (str(item["exposure_folder"]), str(item["block"]), float(item["wavelength_nm"])))
    row_df = pd.DataFrame(sorted_rows)
    for (exposure_folder, block), group in row_df.groupby(["exposure_folder", "block"], sort=False):
        indices = list(group.index)
        block_rows = [sorted_rows[index] for index in indices]
        selected_path = choose_smooth_path(block_rows)

        temp_rows = []
        for row, best in zip(block_rows, selected_path):
            out = {key: value for key, value in row.items() if key != "candidates"}
            out.update(
                {
                    "selected_x": float(best["x"]),
                    "peak_score": float(best["score"]),
                    "adjusted_peak_score": float(best.get("adjusted_score", best["score"])),
                    "peak_persistence": float(best.get("persistence", 0.0)),
                    "peak_prominence": float(best["prominence"]),
                    "peak_width_px": float(best["width_px"]),
                    "peak_left_x": float(best["left_x"]),
                    "peak_right_x": float(best["right_x"]),
                }
            )
            temp_rows.append(out)

        temp_df = pd.DataFrame(temp_rows)
        wl = temp_df["wavelength_nm"].to_numpy(dtype=float)
        x = temp_df["selected_x"].to_numpy(dtype=float)
        local_degree = int(min(degree, max(1, len(temp_df) - 1)))
        coeff, inlier_mask = fit_poly_with_clipping(wl, x, degree=local_degree)
        block_key = f"{exposure_folder}/{block}"
        block_coeffs[block_key] = coeff
        temp_df["fit_x"] = np.polyval(coeff, wl)
        temp_df["residual_px"] = temp_df["selected_x"] - temp_df["fit_x"]
        temp_df["fit_inlier"] = inlier_mask
        selected_rows.extend(temp_df.to_dict(orient="records"))

    selected_df = pd.DataFrame(selected_rows).sort_values(
        ["exposure_folder", "wavelength_nm"], kind="stable"
    ).reset_index(drop=True)
    return selected_df, block_coeffs


def build_results_dataframe(root_dir: Path, degree: int, max_files: int | None = None) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    files = collect_calibration_files(root_dir, max_files=max_files)
    backgrounds = build_block_backgrounds(files)
    for item in files:
        image = ensure_2d_image(np.load(item.path), item.path)
        background = backgrounds.get(item.block_dir) if item.block == "B7" else None
        profile = robust_column_profile(image, background=background)
        candidates = detect_candidate_peaks(profile)
        best = candidates[0]
        rows.append(
            {
                "exposure_folder": item.exposure_folder,
                "block": item.block,
                "filename": item.path.name,
                "path": str(item.path),
                "wavelength_nm": item.wavelength_nm,
                "raw_x": float(best["x"]),
                "raw_peak_score": float(best["score"]),
                "profile_max": float(np.max(profile)),
                "candidates": candidates,
            }
        )

    return choose_calibration_points(rows, degree=degree)


def wavelength_axis_from_calibration(
    calibration_npz: Path,
    n_pixels: int,
) -> np.ndarray:
    model = np.load(calibration_npz)
    coeff_lambda_from_x = model["coeff_lambda_from_x"]
    return np.polyval(coeff_lambda_from_x, np.arange(n_pixels, dtype=float))


def pixel_positions_for_wavelengths(calibration_npz: Path, wavelengths_nm: np.ndarray) -> np.ndarray:
    model = np.load(calibration_npz)
    measured_wavelengths = model["wavelengths_nm"].astype(float)
    measured_x = model["x_pixels"].astype(float)
    order = np.argsort(measured_wavelengths)
    return np.interp(wavelengths_nm, measured_wavelengths[order], measured_x[order])


def reconstruct_spectral_cube(
    array: np.ndarray,
    calibration_npz: Path,
    target_wavelengths_nm: np.ndarray | None = None,
    spectral_axis: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(array, dtype=np.float32)
    moved = np.moveaxis(data, spectral_axis, -1)
    n_pixels = moved.shape[-1]
    source_pixels = np.arange(n_pixels, dtype=float)

    if target_wavelengths_nm is None:
        model = np.load(calibration_npz)
        start = float(np.ceil(np.nanmin(model["wavelengths_nm"])))
        end = float(np.floor(np.nanmax(model["wavelengths_nm"])))
        target_wavelengths_nm = np.arange(start, end + 0.5, 1.0)

    target_pixels = pixel_positions_for_wavelengths(calibration_npz, target_wavelengths_nm)
    flat = moved.reshape(-1, n_pixels)
    reconstructed = np.empty((flat.shape[0], target_wavelengths_nm.size), dtype=np.float32)
    for idx, spectrum in enumerate(flat):
        reconstructed[idx] = np.interp(
            target_pixels,
            source_pixels,
            spectrum,
            left=np.nan,
            right=np.nan,
        )

    reconstructed = reconstructed.reshape(*moved.shape[:-1], target_wavelengths_nm.size)
    reconstructed = np.moveaxis(reconstructed, -1, spectral_axis)
    return reconstructed, target_wavelengths_nm


def save_calibration_model(df: pd.DataFrame, output_npz: Path) -> None:
    inliers = df["fit_inlier"].to_numpy(dtype=bool)
    wavelengths = df.loc[inliers, "wavelength_nm"].to_numpy(dtype=float)
    x_pixels = df.loc[inliers, "selected_x"].to_numpy(dtype=float)
    sort_idx = np.argsort(x_pixels)
    coeff_lambda_from_x = np.polyfit(x_pixels[sort_idx], wavelengths[sort_idx], deg=min(3, len(sort_idx) - 1))
    coeff_x_from_lambda = np.polyfit(wavelengths, x_pixels, deg=min(5, len(wavelengths) - 1))

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        coeff_x_from_lambda=coeff_x_from_lambda,
        coeff_lambda_from_x=coeff_lambda_from_x,
        wavelengths_nm=wavelengths,
        x_pixels=x_pixels,
        residual_px=df.loc[inliers, "residual_px"].to_numpy(dtype=float),
    )


def save_plots(df: pd.DataFrame, plots_dir: Path, show_plot: bool) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    ordered = df.sort_values("wavelength_nm", kind="stable")

    fig, ax = plt.subplots(figsize=(11, 6))
    for block, block_df in ordered.groupby("block", sort=False):
        block_df = block_df.sort_values("wavelength_nm", kind="stable")
        ax.plot(
            block_df["wavelength_nm"],
            block_df["selected_x"],
            marker="o",
            markersize=3,
            linewidth=1.1,
            label=block,
        )
    ax.scatter(
        ordered.loc[~ordered["fit_inlier"], "wavelength_nm"],
        ordered.loc[~ordered["fit_inlier"], "selected_x"],
        s=30,
        marker="x",
        label="Excluido del ajuste",
    )
    ax.set_title("Curva de calibracion espectral")
    ax.set_xlabel("Longitud de onda (nm)")
    ax.set_ylabel("Pixel de dispersion x")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output = plots_dir / "calibration_curve.png"
    fig.savefig(output, dpi=200)
    paths.append(output)
    if show_plot:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axhline(0, color="black", linewidth=1)
    ax.scatter(ordered["wavelength_nm"], ordered["residual_px"], s=16)
    ax.set_title("Residuales del ajuste")
    ax.set_xlabel("Longitud de onda (nm)")
    ax.set_ylabel("Residual (px)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output = plots_dir / "calibration_residuals.png"
    fig.savefig(output, dpi=200)
    paths.append(output)
    if show_plot:
        plt.show()
    plt.close(fig)

    return paths


def run_calibration(args: argparse.Namespace) -> None:
    df, _block_coeffs = build_results_dataframe(args.root, degree=args.poly_degree, max_files=args.max_files)
    if df.empty:
        print(f"No se encontraron archivos .npy procesables dentro de {args.root}")
        return

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    save_calibration_model(df, args.calibration_npz)
    plot_paths = save_plots(df, args.plots_dir, args.show)

    inliers = int(df["fit_inlier"].sum())
    rms = float(np.sqrt(np.mean(np.square(df.loc[df["fit_inlier"], "residual_px"]))))
    print(f"CSV guardado en: {args.output_csv}")
    print(f"Modelo guardado en: {args.calibration_npz}")
    print(f"Archivos procesados: {len(df)} | puntos usados en ajuste: {inliers}")
    print(f"RMS del ajuste: {rms:.3f} px")
    print(f"Graficas generadas: {len(plot_paths)}")


def run_reconstruction(args: argparse.Namespace) -> None:
    if not args.calibration_npz.exists():
        raise FileNotFoundError(f"No existe el modelo de calibracion: {args.calibration_npz}")

    data = np.load(args.input_npy)
    if args.start_nm is None or args.end_nm is None:
        target_wavelengths = None
    else:
        target_wavelengths = np.arange(args.start_nm, args.end_nm + args.step_nm * 0.5, args.step_nm)

    reconstructed, wavelengths = reconstruct_spectral_cube(
        data,
        calibration_npz=args.calibration_npz,
        target_wavelengths_nm=target_wavelengths,
        spectral_axis=args.spectral_axis,
    )

    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, reconstructed)
    axis_path = args.output_npy.with_name(args.output_npy.stem + "_wavelengths_nm.npy")
    np.save(axis_path, wavelengths)

    print(f"Cubo reconstruido guardado en: {args.output_npy}")
    print(f"Eje espectral guardado en: {axis_path}")
    print(f"Shape entrada: {data.shape} -> salida: {reconstructed.shape}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command in {None, "calibrate"}:
        run_calibration(args)
    elif args.command == "reconstruct":
        run_reconstruction(args)
    else:  # pragma: no cover
        parser.error(f"Comando no soportado: {args.command}")


if __name__ == "__main__":
    main()
