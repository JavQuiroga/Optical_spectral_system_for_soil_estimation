r"""
Graficar bandas como gaussianas usando centros y FWHM.

Entrada:
    Firmas_automaticas/bandas_disponibles/CSV/*_bandas_disponibles.csv
    Firmas_automaticas/bandas_disponibles/operativas_30/CSV/*_bandas_operativas_30.csv
    Firmas_automaticas/bandas_disponibles/operativas_40/CSV/*_bandas_operativas_40.csv

Salida:
    Firmas_automaticas/bandas_disponibles/GAUSSIANAS/

La conversion usada es:
    sigma = FWHM / (2*sqrt(2*ln(2))) ~= FWHM / 2.35482
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def gaussian(x: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    sigma = fwhm / FWHM_TO_SIGMA
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def select_centers(df: pd.DataFrame, center_column: str | None, all_bands: bool = False) -> pd.DataFrame:
    if all_bands:
        return df.copy()
    if center_column and center_column in df.columns:
        return df[df[center_column].astype(bool)].copy()
    if "distinguishable_center" in df.columns and center_column is None:
        return df[df["distinguishable_center"].astype(bool)].copy()
    return df.copy()


def plot_cube(
    csv_path: Path,
    output_dir: Path,
    center_column: str | None,
    suffix: str,
    title_label: str,
    all_bands: bool = False,
) -> None:
    df = pd.read_csv(csv_path)
    centers = select_centers(df, center_column, all_bands=all_bands)
    centers = centers[
        np.isfinite(centers["wavelength_nm"].to_numpy(dtype=float))
        & np.isfinite(centers["fwhm_nm"].to_numpy(dtype=float))
    ].copy()
    if centers.empty:
        return

    cube_id = (
        csv_path.name
        .replace("_bandas_disponibles.csv", "")
        .replace("_bandas_operativas_30.csv", "")
        .replace("_bandas_operativas_40.csv", "")
    )
    x = np.linspace(
        max(0, centers["wavelength_nm"].min() - 250),
        centers["wavelength_nm"].max() + 250,
        3000,
    )

    curves = []
    for _, row in centers.iterrows():
        curves.append(gaussian(x, float(row["wavelength_nm"]), float(row["fwhm_nm"])))
    curves = np.asarray(curves)
    envelope = curves.sum(axis=0)
    if envelope.max() > 0:
        envelope = envelope / envelope.max()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for i, (_, row) in enumerate(centers.iterrows()):
        y = curves[i]
        label = f"{row['wavelength_nm']:.0f} nm | FWHM {row['fwhm_nm']:.0f} nm"
        axes[0].plot(x, y, linewidth=0.45 if len(centers) > 100 else 0.9, alpha=0.18 if len(centers) > 100 else 0.75, label=label)
        axes[0].axvline(float(row["wavelength_nm"]), color="black", alpha=0.15, linewidth=0.8)

    axes[0].set_title(f"{cube_id} | Gaussianas individuales | {title_label} ({len(centers)})")
    axes[0].set_ylabel("Respuesta normalizada")
    axes[0].grid(True, alpha=0.3)
    if len(centers) <= 15:
        axes[0].legend(loc="upper right", fontsize=7, ncol=2)

    axes[1].plot(x, envelope, color="#7c3aed", linewidth=1.6, label="Suma normalizada")
    axes[1].scatter(
        centers["wavelength_nm"],
        np.ones(len(centers)),
        color="#dc2626",
        s=25,
        zorder=3,
        label="Centros seleccionados",
    )
    if len(centers) <= 80:
        for _, row in centers.iterrows():
            axes[1].text(
                float(row["wavelength_nm"]),
                1.04,
                f"{row['wavelength_nm']:.0f}",
                ha="center",
                va="bottom",
                fontsize=6 if len(centers) > 20 else 8,
                rotation=45,
            )
    axes[1].set_xlabel("Longitud de onda (nm)")
    axes[1].set_ylabel("Suma normalizada")
    axes[1].set_ylim(0, 1.18)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{cube_id}_gaussianas_{suffix}.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Grafica gaussianas a partir de centros y FWHM.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Firmas_automaticas/bandas_disponibles/CSV"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Firmas_automaticas/bandas_disponibles/GAUSSIANAS"),
    )
    parser.add_argument(
        "--pattern",
        default="*_bandas_disponibles.csv",
        help="Patron de archivos CSV a graficar.",
    )
    parser.add_argument(
        "--center-column",
        default=None,
        help="Columna booleana que marca centros. Si no existe, se grafican todas las filas.",
    )
    parser.add_argument(
        "--suffix",
        default="bandas",
        help="Sufijo para los nombres de las imagenes generadas.",
    )
    parser.add_argument(
        "--title-label",
        default="bandas",
        help="Etiqueta descriptiva para el titulo de la grafica.",
    )
    parser.add_argument(
        "--all-bands",
        action="store_true",
        help="Grafica todas las filas del CSV, ignorando columnas de seleccion.",
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        print(f"No encontre CSV en {args.input_dir}")
        return 2

    for csv_path in files:
        plot_cube(csv_path, args.output_dir, args.center_column, args.suffix, args.title_label, args.all_bands)
        print(f"OK {csv_path.name}")
    print(f"Salida: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
