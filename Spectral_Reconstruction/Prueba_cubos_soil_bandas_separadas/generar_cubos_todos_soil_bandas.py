"""Genera cubos Soil con bandas separadas para todas las capturas disponibles.

No modifica cubos crudos ni firmas. Para cada captura:
  1. Lee cubos crudos desde --raw-root y guarda resultados en --output-root.
  2. Busca la seleccion espacial mas reciente entre ``cubos`` y
     ``cubos_manuales``.
  3. Lee el intervalo espectral final de 300 bandas.
  4. Genera cubos de 84, 49, 35 y 27 bandas y un recorte de 180 x 180 px.

Ejemplos:
  python generar_cubos_todos_soil_bandas.py --raw-root D:\\JavierSTSIVA --output-root D:\\JavierSTSIVA\\resultados_cubos_soil_bandas
  python generar_cubos_todos_soil_bandas.py --only Soil_1__cube_20260617_171738
  python generar_cubos_todos_soil_bandas.py --limit 5
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import traceback
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKFLOW_PATH = SCRIPT_DIR / "generar_prueba_soil_1_cubos_bandas.py"


def load_workflow():
    spec = importlib.util.spec_from_file_location("workflow_cubos_soil", WORKFLOW_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar: {WORKFLOW_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate_ids(spectral_root: Path, raw_input_root: Path) -> list[str]:
    """Return only IDs whose raw cube is physically available in Capturas_soil.

    ``cubos`` puede contener firmas de capturas que no estan presentes en este
    equipo. Partir del crudo evita intentar reconstruir archivos inexistentes.
    """
    nested_raw_root = raw_input_root / "Capturas_soil"
    raw_root = nested_raw_root if nested_raw_root.exists() else raw_input_root
    ids: list[str] = []
    for raw_path in raw_root.glob("Soil_*/*.npy"):
        soil_id = f"{raw_path.parent.name}__{raw_path.stem}"
        has_selection = any(
            (spectral_root / "Firmas_automaticas" / source / soil_id / "resultado.npz").exists()
            for source in ("cubos", "cubos_manuales")
        )
        if has_selection:
            ids.append(soil_id)
        else:
            print(f"Se omite {soil_id}: no tiene resultado.npz en cubos ni cubos_manuales.")
    return sorted(ids)


def already_completed(output_root: Path, soil_id: str) -> bool:
    """Return True only when all four final cubes and their validation are valid."""
    sample_name = soil_id.split("__", maxsplit=1)[0]
    cube_dir = output_root / soil_id
    expected = (
        (2, 84),
        (4, 49),
        (6, 35),
        (8, 27),
    )
    for pixel_step, n_bands in expected:
        cube_file = (
            cube_dir
            / f"{pixel_step}px_{n_bands}_bandas"
            / f"{sample_name}__{n_bands}_Bandas_cubo_recorte_cuadrado_reflectancia.npy"
        )
        if not cube_file.is_file() or cube_file.stat().st_size == 0:
            return False

    validation_path = cube_dir / f"resumen_validacion_{soil_id}.csv"
    if not validation_path.is_file():
        return False
    try:
        with validation_path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        return len(rows) == 4 and all(row.get("status") == "ok" for row in rows)
    except (OSError, csv.Error):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Procesa todos los cubos Soil con bandas separadas.")
    parser.add_argument("--only", help="Procesa solo este cube_id exacto.")
    parser.add_argument("--limit", type=int, help="Limita el numero de cubos (util para prueba).")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=SCRIPT_DIR.parent,
        help="Carpeta que contiene Firmas_automaticas y los Soil_* crudos.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        help="Carpeta de entrada con Capturas_soil o directamente con Soil_*.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Carpeta donde se escribiran los resultados.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocesa incluso los cubos ya completos.",
    )
    args = parser.parse_args()

    spectral_root = args.data_root.resolve()
    raw_root = (args.raw_root or spectral_root).resolve()
    output_root = (args.output_root or (SCRIPT_DIR / "resultados")).resolve()
    ids = [args.only] if args.only else candidate_ids(spectral_root, raw_root)
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit debe ser mayor que cero.")
        ids = ids[: args.limit]
    if not ids:
        raise RuntimeError("No se encontraron resultado.npz en cubos ni cubos_manuales.")

    workflow = load_workflow()
    summary: list[dict[str, str]] = []
    for position, soil_id in enumerate(ids, start=1):
        print(f"[{position}/{len(ids)}] {soil_id}")
        try:
            source = workflow.select_most_recent_result(spectral_root, soil_id)
            if not args.force and already_completed(output_root, soil_id):
                print("  Ya estaba completo: se omite.")
                summary.append(
                    {"cube_id": soil_id, "status": "skipped_completed", "selection_result_npz": str(source), "error": ""}
                )
                continue

            # El modulo principal conserva toda la logica validada con Soil_1.
            workflow.SOIL_ID = soil_id
            workflow.DATA_ROOT = spectral_root
            workflow.RAW_ROOT = raw_root
            workflow.OUTPUT_ROOT = output_root
            workflow.main()
            summary.append({"cube_id": soil_id, "status": "ok", "selection_result_npz": str(source), "error": ""})
        except Exception as error:  # Continua con los demas cubos y deja evidencia del fallo.
            summary.append({"cube_id": soil_id, "status": "error", "selection_result_npz": "", "error": str(error)})
            print(f"  ERROR: {error}")
            traceback.print_exc()

    summary_path = output_root / "resumen_procesamiento_todos_los_cubos.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cube_id", "status", "selection_result_npz", "error"])
        writer.writeheader()
        writer.writerows(summary)

    processed = sum(row["status"] == "ok" for row in summary)
    skipped = sum(row["status"] == "skipped_completed" for row in summary)
    print(f"Procesados ahora: {processed}; ya completos: {skipped}; total: {len(summary)}")
    print(f"Resumen: {summary_path}")
    return 0 if all(row["status"] != "error" for row in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
