r"""
Graficar en una sola imagen las gaussianas seleccionadas por varios criterios
de separabilidad.

Entrada esperada:
    Firmas_automaticas/bandas_separabilidad/criterio_050/CSV/
    Firmas_automaticas/bandas_separabilidad/criterio_035/CSV/
    Firmas_automaticas/bandas_separabilidad/criterio_025/CSV/
    Firmas_automaticas/bandas_separabilidad/criterio_010/CSV/

Salida:
    Firmas_automaticas/bandas_separabilidad/comparacion_criterios_gaussianas.png
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


def criterion_name(factor: float) -> str:
    return f"criterio_{int(round(factor * 100)):03d}"


def find_cube_file(base_dir: Path, criterion: str, cube_id: str | None) -> Path:
    csv_dir = base_dir / criterion / "CSV"
    if cube_id:
        matches = sorted(csv_dir.glob(f"{cube_id}*_bandas_disponibles.csv"))
    else:
        matches = sorted(csv_dir.glob("*_bandas_disponibles.csv"))
    if not matches:
        raise FileNotFoundError(f"No encontre CSV para {criterion} en {csv_dir}")
    return matches[0]


def load_selected(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "distinguishable_center" not in df.columns:
        raise ValueError(f"{csv_path} no tiene columna distinguishable_center")
    selected = df[df["distinguishable_center"].astype(bool)].copy()
    selected = selected[
        np.isfinite(selected["wavelength_nm"].to_numpy(dtype=float))
        & np.isfinite(selected["fwhm_nm"].to_numpy(dtype=float))
    ].copy()
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Compara gaussianas de varios criterios en una sola imagen.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("Firmas_automaticas/bandas_separabilidad"),
    )
    parser.add_argument(
        "--criteria",
        default="0.50,0.35,0.25,0.10",
        help="Factores de separacion separados por coma.",
    )
    parser.add_argument(
        "--cube-id",
        default=None,
        help="Prefijo/id del cubo a graficar. Si se omite, usa el primero disponible.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Firmas_automaticas/bandas_separabilidad/comparacion_criterios_gaussianas.png"),
    )
    args = parser.parse_args()

    factors = [float(item.strip()) for item in args.criteria.split(",") if item.strip()]
    data: list[tuple[float, str, Path, pd.DataFrame]] = []
    for factor in factors:
        criterion = criterion_name(factor)
        csv_path = find_cube_file(args.base_dir, criterion, args.cube_id)
        selected = load_selected(csv_path)
        data.append((factor, criterion, csv_path, selected))

    all_wavelengths = np.concatenate([df["wavelength_nm"].to_numpy(dtype=float) for _, _, _, df in data])
    x = np.linspace(max(0, np.nanmin(all_wavelengths) - 250), np.nanmax(all_wavelengths) + 250, 3000)

    fig, axes = plt.subplots(len(data), 1, figsize=(13, 3.2 * len(data)), sharex=True)
    if len(data) == 1:
        axes = [axes]

    for ax, (factor, criterion, csv_path, selected) in zip(axes, data):
        curves = []
        for _, row in selected.iterrows():
            y = gaussian(x, float(row["wavelength_nm"]), float(row["fwhm_nm"]))
            curves.append(y)
            ax.plot(x, y, linewidth=0.85, alpha=0.45)

        if curves:
            envelope = np.sum(np.asarray(curves), axis=0)
            if np.nanmax(envelope) > 0:
                envelope = envelope / np.nanmax(envelope)
            ax.plot(x, envelope, color="#7c3aed", linewidth=2.0, label="Suma normalizada")

        ax.scatter(
            selected["wavelength_nm"],
            np.ones(len(selected)),
            color="#dc2626",
            s=18,
            zorder=3,
            label="Centros",
        )
        ax.set_ylabel("Respuesta")
        ax.set_title(f"{criterion} | factor={factor:.2f} | {len(selected)} bandas seleccionadas")
        ax.set_ylim(0, 1.18)
        ax.grid(True, alpha=0.28)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Longitud de onda (nm)")
    cube_label = data[0][2].name.replace("_bandas_disponibles.csv", "")
    fig.suptitle(f"Comparacion de criterios de separabilidad - {cube_label}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)

    print(f"Salida: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
