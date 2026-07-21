r"""
Exportar firmas de reflectancia reducidas a las bandas seleccionadas.

Toma:
  - recortes de reflectancia originales:
      Firmas_automaticas/recortes_firmas/CSV/*_recorte.csv
  - bandas seleccionadas por un criterio:
      Firmas_automaticas/bandas_separabilidad/criterio_010/CSV/*_bandas_disponibles.csv

Genera:
  - una firma por cubo con 27 reflectancias, si se usa criterio_010;
  - formato ancho para entrenamiento;
  - formato largo para revisar/graficar;
  - metadata de bandas.

Nota:
  La reflectancia se toma de la firma original recortada. Si el centro de banda
  cae en indice fraccional, se interpola linealmente entre los indices vecinos.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cube_id_from_recorte(path: Path) -> str:
    return path.name.replace("_recorte.csv", "")


def cube_id_from_bandas(path: Path) -> str:
    return path.name.replace("_bandas_disponibles.csv", "")


def load_selected_bands(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "distinguishable_center" not in df.columns:
        raise ValueError(f"{path} no tiene columna distinguishable_center")
    selected = df[df["distinguishable_center"].astype(bool)].copy()
    selected = selected.sort_values("wavelength_nm").reset_index(drop=True)
    selected.insert(0, "band_number", np.arange(1, len(selected) + 1))
    return selected


def reflectance_at_bands(recorte: pd.DataFrame, bands: pd.DataFrame) -> pd.DataFrame:
    x = recorte["band_index"].to_numpy(dtype=float)
    y = recorte["soil_reflectance"].to_numpy(dtype=float)
    band_x = bands["firma_index_float"].to_numpy(dtype=float)
    reflectance = np.interp(band_x, x, y, left=np.nan, right=np.nan)

    out = bands[
        [
            "band_number",
            "wavelength_nm",
            "firma_index_float",
            "fwhm_px",
            "fwhm_nm",
            "dispersion_abs_px_per_nm",
        ]
    ].copy()
    out["reflectance"] = reflectance
    return out


def save_plot(path: Path, cube_id: str, signature: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        signature["wavelength_nm"],
        signature["reflectance"],
        marker="o",
        linewidth=1.4,
        markersize=4,
        color="#2563eb",
    )
    ax.set_title(f"{cube_id} | firma de reflectancia en bandas seleccionadas")
    ax.set_xlabel("Longitud de onda central (nm)")
    ax.set_ylabel("Reflectancia")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Exporta firmas de reflectancia por bandas seleccionadas.")
    parser.add_argument(
        "--recortes-dir",
        type=Path,
        default=Path("Firmas_automaticas/recortes_firmas/CSV"),
    )
    parser.add_argument(
        "--bandas-dir",
        type=Path,
        default=Path("Firmas_automaticas/bandas_separabilidad/criterio_010/CSV"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Firmas_automaticas/dataset_reflectancia_criterio_010"),
    )
    args = parser.parse_args()

    recorte_files = {cube_id_from_recorte(path): path for path in args.recortes_dir.glob("*_recorte.csv")}
    banda_files = {cube_id_from_bandas(path): path for path in args.bandas_dir.glob("*_bandas_disponibles.csv")}
    common_ids = sorted(set(recorte_files) & set(banda_files))

    if not common_ids:
        print("No encontre cubos comunes entre recortes y bandas.")
        return 2

    firmas_dir = args.output_dir / "firmas"
    plots_dir = args.output_dir / "PLOTS"
    firmas_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    long_rows: list[pd.DataFrame] = []
    wide_rows: list[dict[str, object]] = []
    metadata_rows: list[pd.DataFrame] = []

    for cube_id in common_ids:
        recorte = pd.read_csv(recorte_files[cube_id])
        bands = load_selected_bands(banda_files[cube_id])
        signature = reflectance_at_bands(recorte, bands)
        signature.insert(0, "cube_id", cube_id)

        signature.to_csv(firmas_dir / f"{cube_id}_firma_reflectancia_bandas.csv", index=False, encoding="utf-8-sig")
        save_plot(plots_dir / f"{cube_id}_firma_reflectancia_bandas.png", cube_id, signature)

        long_rows.append(signature)

        wide: dict[str, object] = {"cube_id": cube_id}
        for _, row in signature.iterrows():
            band_number = int(row["band_number"])
            wavelength = int(round(float(row["wavelength_nm"])))
            wide[f"B{band_number:02d}_{wavelength}nm"] = float(row["reflectance"])
        wide_rows.append(wide)

        metadata_rows.append(
            signature[
                [
                    "cube_id",
                    "band_number",
                    "wavelength_nm",
                    "firma_index_float",
                    "fwhm_px",
                    "fwhm_nm",
                    "dispersion_abs_px_per_nm",
                ]
            ]
        )

        print(f"{cube_id}: {len(signature)} bandas exportadas")

    long_df = pd.concat(long_rows, ignore_index=True)
    metadata_df = pd.concat(metadata_rows, ignore_index=True)
    wide_df = pd.DataFrame(wide_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(args.output_dir / "firmas_reflectancia_formato_largo.csv", index=False, encoding="utf-8-sig")
    wide_df.to_csv(args.output_dir / "firmas_reflectancia_formato_ancho.csv", index=False, encoding="utf-8-sig")
    metadata_df.to_csv(args.output_dir / "metadata_bandas_por_cubo.csv", index=False, encoding="utf-8-sig")

    # Si las bandas son iguales para todos los cubos, deja tambien una metadata unica.
    first_id = common_ids[0]
    first_metadata = metadata_df[metadata_df["cube_id"] == first_id].drop(columns=["cube_id"])
    first_metadata.to_csv(args.output_dir / "metadata_bandas_referencia.csv", index=False, encoding="utf-8-sig")

    print(f"Salida: {args.output_dir}")
    print(f"Formato ancho: {args.output_dir / 'firmas_reflectancia_formato_ancho.csv'}")
    print(f"Formato largo: {args.output_dir / 'firmas_reflectancia_formato_largo.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
