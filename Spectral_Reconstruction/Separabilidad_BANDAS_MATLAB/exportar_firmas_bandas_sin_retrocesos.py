"""Exporta las firmas CSV separadas por bandas a formatos NPY y MAT.

Lee los CSV generados por ``generar_firmas_todas_las_muestras.m`` sin
modificarlos. Para cada firma crea:
  - NPY: reflectancia interpolada, vector columna float32 (N x 1).
  - MAT: reflectancia, longitudes de onda y trazabilidad de la seleccion.

Uso desde la raiz del proyecto:
    python Spectral_Reconstruction/Separabilidad_BANDAS_MATLAB/
        exportar_firmas_bandas_sin_retrocesos.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat


TOLERANCE_DIR_RE = re.compile(r"^(\d+)px_(\d+)_bandas$")
SOIL_ID_RE = re.compile(r"^(Soil_\d+)")
REQUIRED_COLUMNS = {
    "source_row",
    "wavelength_nm",
    "posicion_equivalente_soil",
    "soil_reflectance_interpolada",
}


def matlab_cellstr(value: str) -> np.ndarray:
    """Create a MATLAB-compatible one-element cell string."""
    return np.array([value], dtype=object)


def export_csv(csv_path: Path, pixel_step: int, expected_bands: int) -> dict[str, object]:
    data = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas {sorted(missing)}")
    if len(data) != expected_bands:
        raise ValueError(
            f"Tiene {len(data)} filas; se esperaban {expected_bands} bandas"
        )

    source_row = data["source_row"].to_numpy(dtype=np.float64).reshape(-1, 1)
    wavelength = data["wavelength_nm"].to_numpy(dtype=np.float64).reshape(-1, 1)
    equivalent_position = data["posicion_equivalente_soil"].to_numpy(
        dtype=np.float64
    ).reshape(-1, 1)
    reflectance = data["soil_reflectance_interpolada"].to_numpy(
        dtype=np.float32
    ).reshape(-1, 1)

    if not (
        np.isfinite(source_row).all()
        and np.isfinite(wavelength).all()
        and np.isfinite(equivalent_position).all()
        and np.isfinite(reflectance).all()
    ):
        raise ValueError("El CSV contiene valores no finitos")

    match = SOIL_ID_RE.match(csv_path.stem)
    soil_id = match.group(1) if match else csv_path.stem
    criterio = f"sin_retrocesos_salto_minimo_{pixel_step}px"

    npy_dir = csv_path.parent / "NPY"
    mat_dir = csv_path.parent / "MAT"
    npy_dir.mkdir(exist_ok=True)
    mat_dir.mkdir(exist_ok=True)

    npy_path = npy_dir / f"{csv_path.stem}.npy"
    mat_path = mat_dir / f"{csv_path.stem}.mat"

    # Formato equivalente a las firmas NPY finales del proyecto: N x 1.
    np.save(npy_path, reflectance)

    mat_payload: dict[str, np.ndarray] = {
        "reflectanciaSoil": reflectance.astype(np.float64),
        "ejeBandas": wavelength,
        "reflectanciaSoilOriginal": reflectance.astype(np.float64),
        "bandasOriginal": wavelength,
        "sourceRow": source_row,
        "posicionEquivalenteSoil": equivalent_position,
        "metodoSeleccion": matlab_cellstr(criterio),
        "targetLength": np.array([[expected_bands]], dtype=np.int32),
        "pixelStep": np.array([[pixel_step]], dtype=np.int32),
        "cubeId": matlab_cellstr(soil_id),
        "archivoOrigen": matlab_cellstr(csv_path.name),
    }
    if expected_bands == 27:
        mat_payload["reflectanciaSoil27"] = reflectance.astype(np.float64)
        mat_payload["ejeBandas27"] = wavelength
    savemat(mat_path, mat_payload)

    return {
        "csv_file": str(csv_path.relative_to(csv_path.parents[2])),
        "pixel_step": pixel_step,
        "n_bands": expected_bands,
        "npy_file": str(npy_path.relative_to(csv_path.parents[2])),
        "mat_file": str(mat_path.relative_to(csv_path.parents[2])),
        "status": "ok",
    }


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    root_dir = (
        script_dir.parent / "Firmas_automaticas" / "firmas_bandas_sin_retrocesos"
    )
    if not root_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de firmas: {root_dir}")

    summary_rows: list[dict[str, object]] = []
    tolerance_dirs = sorted(
        path for path in root_dir.iterdir() if path.is_dir() and TOLERANCE_DIR_RE.match(path.name)
    )
    if not tolerance_dirs:
        raise FileNotFoundError("No se encontraron carpetas de tolerancia (ej. 2px_84_bandas).")

    for tolerance_dir in tolerance_dirs:
        match = TOLERANCE_DIR_RE.match(tolerance_dir.name)
        assert match is not None
        pixel_step, expected_bands = map(int, match.groups())
        csv_files = sorted(tolerance_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No hay CSV de firmas en {tolerance_dir}")

        print(f"Convirtiendo {len(csv_files)} firmas: {tolerance_dir.name}")
        for csv_path in csv_files:
            try:
                summary_rows.append(export_csv(csv_path, pixel_step, expected_bands))
            except Exception as exc:  # Continue so one damaged CSV does not stop the batch.
                summary_rows.append(
                    {
                        "csv_file": str(csv_path.relative_to(root_dir)),
                        "pixel_step": pixel_step,
                        "n_bands": expected_bands,
                        "npy_file": "",
                        "mat_file": "",
                        "status": f"error: {exc}",
                    }
                )

    summary_path = root_dir / "resumen_conversion_mat_npy.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["csv_file", "pixel_step", "n_bands", "npy_file", "mat_file", "status"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    n_ok = sum(row["status"] == "ok" for row in summary_rows)
    print(f"Conversiones correctas: {n_ok} de {len(summary_rows)}")
    print(f"Resumen: {summary_path}")
    return 0 if n_ok == len(summary_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
