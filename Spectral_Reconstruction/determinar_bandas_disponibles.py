r"""
Determinar bandas espectrales distinguibles en recortes de firmas.

Idea:
  - La firma tiene indices de pixel espectral, no longitudes de onda.
  - La curva de dispersion indica donde cae cada longitud de onda en pixeles.
  - Para cada recorte, se coloca la curva de dispersion relativa al primer
    indice del recorte:

        x_firma(lambda) = recorte_start + (selected_x(lambda) - min(selected_x))

  - Con peak_width_px de la caracterizacion se estima FWHM_nm:

        FWHM_nm(lambda) = FWHM_px(lambda) / |d(selected_x)/d(lambda)|

  - Se cuentan bandas distinguibles usando una seleccion greedy: la siguiente
    banda debe estar separada segun un criterio fisico basado en FWHM.

Este script NO suaviza ni modifica las firmas recortadas; solo usa calibracion
y caracterizacion para estimar resolucion y bandas disponibles.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


@dataclass
class BandSummary:
    cube_id: str
    crop_start: int
    crop_stop: int
    crop_width_px: int
    wavelength_min_nm: float
    wavelength_max_nm: float
    wavelengths_inside_count: int
    distinguishable_bands_count: int
    separation_factor: float
    median_selected_spacing_nm: float
    median_selected_spacing_px: float
    median_fwhm_nm: float
    median_fwhm_px: float
    median_dispersion_px_per_nm: float
    status: str
    reason: str


def safe_savgol(y: np.ndarray, window: int = 31, polyorder: int = 3) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.size < 5:
        return y.copy()
    window = min(window, y.size if y.size % 2 == 1 else y.size - 1)
    window = max(window, polyorder + 2)
    if window % 2 == 0:
        window += 1
    if window > y.size:
        window = y.size if y.size % 2 == 1 else y.size - 1
    if window <= polyorder:
        return y.copy()
    return savgol_filter(y, window_length=window, polyorder=polyorder, mode="interp")


def load_dispersion(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"wavelength_nm", "selected_x"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en calibracion: {missing}")
    df = df[list(required)].dropna().sort_values("wavelength_nm")
    df = df.groupby("wavelength_nm", as_index=False)["selected_x"].median()
    df["selected_x_smooth"] = safe_savgol(df["selected_x"].to_numpy(), window=41, polyorder=3)
    wavelength = df["wavelength_nm"].to_numpy(dtype=np.float64)
    selected_x = df["selected_x_smooth"].to_numpy(dtype=np.float64)
    df["dispersion_px_per_nm"] = np.gradient(selected_x, wavelength)
    return df


def load_fwhm(characterization_path: Path, wavelengths: np.ndarray) -> np.ndarray:
    df = pd.read_csv(characterization_path)
    required = {"wavelength_nm", "peak_width_px"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en caracterizacion: {missing}")
    if "fit_inlier" in df.columns:
        df = df[df["fit_inlier"].astype(bool)]
    df = df[list(required)].dropna().sort_values("wavelength_nm")
    df = df.groupby("wavelength_nm", as_index=False)["peak_width_px"].median()
    return np.interp(
        wavelengths,
        df["wavelength_nm"].to_numpy(dtype=np.float64),
        df["peak_width_px"].to_numpy(dtype=np.float64),
    )


def greedy_distinguishable_bands(
    wavelength_nm: np.ndarray,
    fwhm_nm: np.ndarray,
    separation_factor: float,
) -> np.ndarray:
    selected: list[int] = []
    if wavelength_nm.size == 0:
        return np.array([], dtype=int)
    last_lambda: float | None = None
    last_fwhm: float | None = None
    for i, wavelength in enumerate(wavelength_nm):
        if not np.isfinite(wavelength) or not np.isfinite(fwhm_nm[i]):
            continue
        if last_lambda is None:
            selected.append(i)
            last_lambda = float(wavelength)
            last_fwhm = float(max(fwhm_nm[i], 1e-9))
            continue
        # separation_factor=0.5 equivale a no solapar los semianchos FWHM:
        # Δlambda >= 0.5*(FWHM_i + FWHM_j)
        required_spacing = separation_factor * (float(last_fwhm or 0.0) + float(fwhm_nm[i]))
        if float(wavelength) - last_lambda >= required_spacing:
            selected.append(i)
            last_lambda = float(wavelength)
            last_fwhm = float(max(fwhm_nm[i], 1e-9))
    return np.asarray(selected, dtype=int)


def selected_spacing_stats(df: pd.DataFrame) -> tuple[float, float]:
    selected = df[df["distinguishable_center"]].sort_values("firma_index_float")
    if len(selected) < 2:
        return float("nan"), float("nan")
    spacing_nm = np.diff(selected["wavelength_nm"].to_numpy(dtype=np.float64))
    spacing_px = np.diff(selected["firma_index_float"].to_numpy(dtype=np.float64))
    return float(np.nanmedian(np.abs(spacing_nm))), float(np.nanmedian(np.abs(spacing_px)))


def process_crop(
    crop_row: pd.Series,
    dispersion: pd.DataFrame,
    characterization_path: Path,
    output_dir: Path,
    separation_factor: float,
) -> BandSummary:
    cube_id = str(crop_row["cube_id"])
    crop_start = int(crop_row["start"])
    crop_stop = int(crop_row["stop"])
    crop_width = crop_stop - crop_start

    df = dispersion.copy()
    x0 = float(df["selected_x_smooth"].min())
    df["x_relative"] = df["selected_x_smooth"] - x0
    df["firma_index_float"] = crop_start + df["x_relative"]
    inside = df[(df["firma_index_float"] >= crop_start) & (df["firma_index_float"] < crop_stop)].copy()

    if inside.empty:
        return BandSummary(
            cube_id=cube_id,
            crop_start=crop_start,
            crop_stop=crop_stop,
            crop_width_px=crop_width,
            wavelength_min_nm=float("nan"),
            wavelength_max_nm=float("nan"),
            wavelengths_inside_count=0,
            distinguishable_bands_count=0,
            separation_factor=separation_factor,
            median_selected_spacing_nm=float("nan"),
            median_selected_spacing_px=float("nan"),
            median_fwhm_nm=float("nan"),
            median_fwhm_px=float("nan"),
            median_dispersion_px_per_nm=float("nan"),
            status="review",
            reason="sin_longitudes_dentro_del_recorte",
        )

    wavelengths = inside["wavelength_nm"].to_numpy(dtype=np.float64)
    fwhm_px = load_fwhm(characterization_path, wavelengths)
    dispersion_abs = np.abs(inside["dispersion_px_per_nm"].to_numpy(dtype=np.float64))
    dispersion_abs[dispersion_abs < 1e-9] = np.nan
    fwhm_nm = fwhm_px / dispersion_abs
    inside["fwhm_px"] = fwhm_px
    inside["fwhm_nm"] = fwhm_nm
    inside["dispersion_abs_px_per_nm"] = dispersion_abs

    finite_resolution = np.isfinite(fwhm_nm)
    selected_idx = greedy_distinguishable_bands(
        wavelengths[finite_resolution],
        fwhm_nm[finite_resolution],
        separation_factor,
    )
    selected_mask = np.zeros(len(inside), dtype=bool)
    finite_positions = np.flatnonzero(finite_resolution)
    selected_mask[finite_positions[selected_idx]] = True
    inside["distinguishable_center"] = selected_mask

    csv_dir = output_dir / "CSV"
    plot_dir = output_dir / "PLOTS"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    inside.to_csv(csv_dir / f"{cube_id}_bandas_disponibles.csv", index=False, encoding="utf-8-sig")

    save_plot(plot_dir / f"{cube_id}_bandas_disponibles.png", cube_id, inside, crop_start, crop_stop)

    median_fwhm_nm = float(np.nanmedian(fwhm_nm)) if np.any(np.isfinite(fwhm_nm)) else float("nan")
    median_fwhm_px = float(np.nanmedian(fwhm_px)) if fwhm_px.size else float("nan")
    median_disp = float(np.nanmedian(dispersion_abs)) if np.any(np.isfinite(dispersion_abs)) else float("nan")
    median_selected_spacing_nm, median_selected_spacing_px = selected_spacing_stats(inside)
    status = "ok"
    reason = ""
    if np.sum(selected_mask) < 2:
        status = "review"
        reason = "muy_pocas_bandas_distinguibles"

    return BandSummary(
        cube_id=cube_id,
        crop_start=crop_start,
        crop_stop=crop_stop,
        crop_width_px=crop_width,
        wavelength_min_nm=float(np.nanmin(wavelengths)),
        wavelength_max_nm=float(np.nanmax(wavelengths)),
        wavelengths_inside_count=int(len(inside)),
        distinguishable_bands_count=int(np.sum(selected_mask)),
        separation_factor=separation_factor,
        median_selected_spacing_nm=median_selected_spacing_nm,
        median_selected_spacing_px=median_selected_spacing_px,
        median_fwhm_nm=median_fwhm_nm,
        median_fwhm_px=median_fwhm_px,
        median_dispersion_px_per_nm=median_disp,
        status=status,
        reason=reason,
    )


def save_plot(path: Path, cube_id: str, df: pd.DataFrame, crop_start: int, crop_stop: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(df["wavelength_nm"], df["firma_index_float"], color="#2563eb", linewidth=1.3)
    selected = df[df["distinguishable_center"]]
    axes[0].scatter(
        selected["wavelength_nm"],
        selected["firma_index_float"],
        color="#dc2626",
        s=18,
        label="Bandas distinguibles",
        zorder=3,
    )
    axes[0].axhline(crop_start, color="gray", linestyle="--", linewidth=0.8)
    axes[0].axhline(crop_stop, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Indice dentro de firma")
    axes[0].set_title(f"{cube_id} | {int(selected.shape[0])} bandas distinguibles por FWHM")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(df["wavelength_nm"], df["fwhm_nm"], color="#7c3aed", linewidth=1.2)
    axes[1].scatter(selected["wavelength_nm"], selected["fwhm_nm"], color="#dc2626", s=18, zorder=3)
    axes[1].set_xlabel("Longitud de onda (nm)")
    axes[1].set_ylabel("FWHM estimado (nm)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(path: Path, summaries: list[BandSummary]) -> None:
    if not summaries:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        for item in summaries:
            writer.writerow(asdict(item))


def main() -> int:
    parser = argparse.ArgumentParser(description="Determina bandas distinguibles usando dispersion y FWHM.")
    parser.add_argument(
        "--recortes",
        type=Path,
        default=Path("Firmas_automaticas/recortes_firmas/resumen_recortes.csv"),
    )
    parser.add_argument(
        "--dispersion",
        type=Path,
        default=Path("spectral_characterization_plots/curva_calibracion_integrada_puntos_generados_naturales_1450_1700.csv"),
    )
    parser.add_argument(
        "--characterization",
        type=Path,
        default=Path("spectral_characterization_plots/spectral_characterization.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Firmas_automaticas/bandas_disponibles"),
    )
    parser.add_argument(
        "--separation-factor",
        type=float,
        default=0.5,
        help=(
            "Factor de separacion entre centros usando FWHM. "
            "0.5 exige Delta_lambda >= 0.5*(FWHM_i + FWHM_j), "
            "equivalente a que no se solapen los semianchos FWHM."
        ),
    )
    args = parser.parse_args()

    recortes = pd.read_csv(args.recortes)
    dispersion = load_dispersion(args.dispersion)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[BandSummary] = []
    for _, row in recortes.iterrows():
        summary = process_crop(row, dispersion, args.characterization, args.output_dir, args.separation_factor)
        summaries.append(summary)
        print(
            f"{summary.cube_id}: {summary.distinguishable_bands_count} bandas distinguibles "
            f"(criterio={summary.separation_factor:.2f}) "
            f"({summary.wavelength_min_nm:.1f}-{summary.wavelength_max_nm:.1f} nm) {summary.status}"
        )

    write_summary(args.output_dir / "resumen_bandas_disponibles.csv", summaries)
    print(f"Salida: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
