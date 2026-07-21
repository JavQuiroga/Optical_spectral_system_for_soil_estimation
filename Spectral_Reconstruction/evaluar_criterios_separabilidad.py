r"""
Evaluar varios criterios de separabilidad espectral basados en FWHM.

Este script corre determinar_bandas_disponibles para varios factores de
separacion y organiza las salidas en carpetas independientes:

    Firmas_automaticas/bandas_separabilidad/
        criterio_050/
        criterio_035/
        criterio_025/
        resumen_comparativo.csv

El factor se interpreta asi:

    Delta_lambda >= factor * (FWHM_i + FWHM_j)

No se modifica el FWHM medido ni se crean bandas nuevas. Se seleccionan
subconjuntos de las longitudes de onda calibradas bajo distintos niveles de
solapamiento permitido.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from determinar_bandas_disponibles import (
    BandSummary,
    load_dispersion,
    process_crop,
    write_summary,
)
from graficar_gaussianas_bandas import plot_cube


def parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0:
            raise ValueError("Los factores de separacion deben ser positivos.")
        values.append(value)
    return sorted(set(values), reverse=True)


def criterion_name(factor: float) -> str:
    return f"criterio_{int(round(factor * 100)):03d}"


def criterion_label(factor: float) -> str:
    if factor >= 0.5:
        return "estricto"
    if factor >= 0.35:
        return "moderado"
    return "permisivo"


def run_criterion(
    recortes: pd.DataFrame,
    dispersion: pd.DataFrame,
    characterization: Path,
    output_dir: Path,
    factor: float,
) -> list[BandSummary]:
    criterion_dir = output_dir / criterion_name(factor)
    criterion_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[BandSummary] = []
    for _, row in recortes.iterrows():
        summary = process_crop(
            crop_row=row,
            dispersion=dispersion,
            characterization_path=characterization,
            output_dir=criterion_dir,
            separation_factor=factor,
        )
        summaries.append(summary)
        print(
            f"{criterion_name(factor)} | {summary.cube_id}: "
            f"{summary.distinguishable_bands_count} bandas"
        )

    write_summary(criterion_dir / "resumen_bandas_disponibles.csv", summaries)
    generate_gaussian_plots(criterion_dir, factor)
    return summaries


def generate_gaussian_plots(criterion_dir: Path, factor: float) -> None:
    csv_dir = criterion_dir / "CSV"
    output_dir = criterion_dir / "GAUSSIANAS"
    files = sorted(csv_dir.glob("*_bandas_disponibles.csv"))
    for csv_path in files:
        plot_cube(
            csv_path=csv_path,
            output_dir=output_dir,
            center_column=None,
            suffix=f"criterio_{int(round(factor * 100)):03d}",
            title_label=f"{criterion_label(factor)} | factor {factor:.2f}",
        )


def write_comparative_summary(path: Path, all_summaries: dict[float, list[BandSummary]]) -> None:
    rows: list[dict[str, object]] = []
    for factor, summaries in all_summaries.items():
        counts = np.asarray([item.distinguishable_bands_count for item in summaries], dtype=float)
        spacing_nm = np.asarray([item.median_selected_spacing_nm for item in summaries], dtype=float)
        fwhm_nm = np.asarray([item.median_fwhm_nm for item in summaries], dtype=float)
        sampled = np.asarray([item.wavelengths_inside_count for item in summaries], dtype=float)
        rows.append(
            {
                "criterio": criterion_name(factor),
                "descripcion": criterion_label(factor),
                "separation_factor": factor,
                "num_cubos": len(summaries),
                "bandas_separables_mediana": float(np.nanmedian(counts)),
                "bandas_separables_min": int(np.nanmin(counts)) if counts.size else 0,
                "bandas_separables_max": int(np.nanmax(counts)) if counts.size else 0,
                "bandas_muestreadas_mediana": float(np.nanmedian(sampled)),
                "espaciado_seleccionado_nm_mediana": float(np.nanmedian(spacing_nm)),
                "fwhm_nm_mediana": float(np.nanmedian(fwhm_nm)),
                "carpeta": str(path.parent / criterion_name(factor)),
            }
        )

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_long_summary(path: Path, all_summaries: dict[float, list[BandSummary]]) -> None:
    rows: list[dict[str, object]] = []
    for factor, summaries in all_summaries.items():
        for summary in summaries:
            row = asdict(summary)
            row["criterio"] = criterion_name(factor)
            row["descripcion"] = criterion_label(factor)
            rows.append(row)

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Genera carpetas de bandas separables para varios criterios FWHM."
    )
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
        default=Path("Firmas_automaticas/bandas_separabilidad"),
    )
    parser.add_argument(
        "--criteria",
        default="0.50,0.35,0.25",
        help="Factores de separacion a evaluar, separados por coma.",
    )
    args = parser.parse_args()

    recortes = pd.read_csv(args.recortes)
    dispersion = load_dispersion(args.dispersion)
    factors = parse_float_list(args.criteria)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: dict[float, list[BandSummary]] = {}
    for factor in factors:
        all_summaries[factor] = run_criterion(
            recortes=recortes,
            dispersion=dispersion,
            characterization=args.characterization,
            output_dir=args.output_dir,
            factor=factor,
        )

    write_comparative_summary(args.output_dir / "resumen_comparativo.csv", all_summaries)
    write_long_summary(args.output_dir / "resumen_comparativo_por_cubo.csv", all_summaries)

    print(f"Salida principal: {args.output_dir}")
    print(f"Resumen comparativo: {args.output_dir / 'resumen_comparativo.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
