r"""
Exporta datos ligeros desde resultados ya procesados en Firmas_automaticas.

No copia cubos .npy ni mascaras completas. Extrae lo necesario para analizar
segmentaciones en otro PC:
    - metadata completa por cubo
    - caracteristicas numericas resumidas
    - firmas 1D SOIL/WHITE/DARK/reflectancia

Uso tipico:
    python Spectral_Reconstruction/exportar_datos_ligeros_cubos.py ^
        --output-dir "E:\ruta\Firmas_automaticas" ^
        --export-dir "E:\export_ligero_firmas"
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def robust_roughness(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 5:
        return float("nan")
    y = values[finite].astype(float)
    span = np.nanpercentile(y, 95) - np.nanpercentile(y, 5)
    span = max(float(span), 1e-9)
    return float(np.nanmedian(np.abs(np.diff(y))) / span)


def add_component(row: dict[str, object], metadata: dict[str, object], role: str) -> None:
    components = metadata.get("selected_components", {})
    if not isinstance(components, dict):
        return
    component = components.get(role, {})
    if not isinstance(component, dict):
        return

    prefix = f"{role}_"
    row[prefix + "centroid_x"] = safe_float(component.get("centroid_x"))
    row[prefix + "centroid_y"] = safe_float(component.get("centroid_y"))
    row[prefix + "area_fraction"] = safe_float(component.get("area_fraction"))
    row[prefix + "brightness"] = safe_float(component.get("brightness"))
    row[prefix + "border_fraction"] = safe_float(component.get("border_fraction"))
    row[prefix + "fill_fraction"] = safe_float(component.get("fill_fraction"))


def add_expected_distances(row: dict[str, object], metadata: dict[str, object]) -> None:
    config = metadata.get("config", {})
    if not isinstance(config, dict):
        return
    for role in ("soil", "white"):
        expected = config.get(f"expected_{role}")
        if not isinstance(expected, list) or len(expected) != 2:
            continue
        x = safe_float(row.get(f"{role}_centroid_x"))
        y = safe_float(row.get(f"{role}_centroid_y"))
        if np.isfinite(x) and np.isfinite(y):
            row[f"{role}_dist_expected"] = float(
                math.hypot(x - float(expected[0]), y - float(expected[1]))
            )


def add_attempt_summary(row: dict[str, object], metadata: dict[str, object]) -> None:
    attempts = metadata.get("preview_recipe_attempts", [])
    if not isinstance(attempts, list):
        return
    ok = [item for item in attempts if isinstance(item, dict) and item.get("status") != "error"]
    row["num_preview_recipes_ok"] = len(ok)
    row["num_preview_recipes_error"] = len(attempts) - len(ok)
    scores = [safe_float(item.get("recipe_score")) for item in ok]
    scores = [score for score in scores if np.isfinite(score)]
    if scores:
        scores_sorted = sorted(scores, reverse=True)
        row["preview_recipe_score_max"] = scores_sorted[0]
        row["preview_recipe_score_mean"] = float(np.mean(scores_sorted))
        row["preview_recipe_score_std"] = float(np.std(scores_sorted))
        row["preview_recipe_score_margin"] = (
            scores_sorted[0] - scores_sorted[1] if len(scores_sorted) > 1 else float("nan")
        )

    selected = metadata.get("selected_preview_recipe_index")
    if selected is None:
        return
    for item in ok:
        if int(item.get("recipe_index", -1)) != int(selected):
            continue
        ranges = item.get("ranges")
        if isinstance(ranges, list):
            row["selected_recipe_ranges"] = ";".join(
                f"{pair[0]}:{pair[1]}"
                for pair in ranges
                if isinstance(pair, list) and len(pair) == 2
            )
        row["selected_recipe_confidence"] = safe_float(item.get("confidence"))
        row["selected_recipe_invalid_fraction"] = safe_float(
            item.get("invalid_reflectance_fraction")
        )
        row["selected_recipe_outside_fraction"] = safe_float(
            item.get("reflectance_outside_fraction")
        )
        break


def add_signature_summary(row: dict[str, object], result_path: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    if not result_path.exists():
        row["resultado_existe"] = False
        return arrays

    row["resultado_existe"] = True
    with np.load(result_path, allow_pickle=False) as data:
        for name in ("soil_signature", "white_signature", "dark_signature", "soil_reflectance"):
            if name in data.files:
                arrays[name] = data[name].astype(np.float32)

    reflectance = arrays.get("soil_reflectance")
    soil = arrays.get("soil_signature")
    white = arrays.get("white_signature")
    dark = arrays.get("dark_signature")
    if reflectance is not None:
        usable = reflectance[100:900].astype(float)
        row["reflectance_nan_fraction_100_900"] = float(np.mean(~np.isfinite(usable)))
        row["reflectance_outside_0_15_fraction_100_900"] = float(
            np.mean((usable < 0) | (usable > 1.5) | ~np.isfinite(usable))
        )
        row["reflectance_median_100_900"] = float(np.nanmedian(usable))
        row["reflectance_std_100_900"] = float(np.nanstd(usable))
        row["reflectance_p05_100_900"] = float(np.nanpercentile(usable, 5))
        row["reflectance_p95_100_900"] = float(np.nanpercentile(usable, 95))
        row["reflectance_roughness_100_900"] = robust_roughness(usable)

    if soil is not None and white is not None and dark is not None:
        denom = (white - dark)[100:900].astype(float)
        row["denominator_median_100_900"] = float(np.nanmedian(denom))
        row["denominator_p05_100_900"] = float(np.nanpercentile(denom, 5))
        row["denominator_nonpositive_fraction_100_900"] = float(np.mean(denom <= 0))
        row["soil_raw_median_100_900"] = float(np.nanmedian(soil[100:900]))
        row["white_raw_median_100_900"] = float(np.nanmedian(white[100:900]))
        row["dark_raw_median_100_900"] = float(np.nanmedian(dark[100:900]))
        row["white_minus_soil_raw_median_100_900"] = float(
            np.nanmedian(white[100:900] - soil[100:900])
        )
    return arrays


def save_signature_csv(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if not arrays:
        return
    max_len = max(len(value) for value in arrays.values())
    data: dict[str, object] = {"idx": np.arange(max_len)}
    for name, values in arrays.items():
        padded = np.full(max_len, np.nan, dtype=np.float32)
        padded[: len(values)] = values
        data[name] = padded
    pd.DataFrame(data).to_csv(path, index=False)


def export_cube(cube_dir: Path, export_dir: Path) -> dict[str, object] | None:
    metadata_path = cube_dir / "metadata.json"
    result_path = cube_dir / "resultado.npz"
    if not metadata_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cube_id = str(metadata.get("cube_id", cube_dir.name))
    segmentation = metadata.get("segmentation", {})
    if not isinstance(segmentation, dict):
        segmentation = {}
    role_scores = metadata.get("role_scores", {})
    if not isinstance(role_scores, dict):
        role_scores = {}

    row: dict[str, object] = {
        "cube_id": cube_id,
        "metadata_existe": True,
        "resultado_existe": result_path.exists(),
        "status_auto": metadata.get("status"),
        "reason_auto": metadata.get("reason"),
        "confidence": safe_float(metadata.get("confidence")),
        "invalid_reflectance_fraction": safe_float(metadata.get("invalid_reflectance_fraction")),
        "reflectance_outside_fraction": safe_float(metadata.get("reflectance_outside_fraction")),
        "selected_k": segmentation.get("selected_k"),
        "selected_quality": safe_float(segmentation.get("selected_quality")),
        "selected_preview_recipe_index": metadata.get("selected_preview_recipe_index"),
        "selected_preview_recipe_score": safe_float(
            segmentation.get("selected_preview_recipe_score")
        ),
        "role_score_soil": safe_float(role_scores.get("soil")),
        "role_score_white": safe_float(role_scores.get("white")),
        "role_score_dark": safe_float(role_scores.get("dark")),
        "metadata_original": str(metadata_path),
    }
    add_component(row, metadata, "soil")
    add_component(row, metadata, "white")
    add_expected_distances(row, metadata)
    add_attempt_summary(row, metadata)
    arrays = add_signature_summary(row, result_path)

    (export_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (export_dir / "firmas").mkdir(parents=True, exist_ok=True)
    shutil.copy2(metadata_path, export_dir / "metadata" / f"{cube_id}.json")
    save_signature_csv(export_dir / "firmas" / f"{cube_id}_firmas.csv", arrays)

    diagnostic = cube_dir / "diagnostico.png"
    if diagnostic.exists():
        (export_dir / "diagnosticos").mkdir(parents=True, exist_ok=True)
        shutil.copy2(diagnostic, export_dir / "diagnosticos" / f"{cube_id}_diagnostico.png")
        row["diagnostico_exportado"] = str(export_dir / "diagnosticos" / f"{cube_id}_diagnostico.png")
    return row


def zip_directory(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_dir.rglob("*"):
            if path == zip_path or path.is_dir():
                continue
            archive.write(path, path.relative_to(source_dir.parent))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exporta metadata y firmas ligeras desde cubos ya procesados."
    )
    parser.add_argument("--output-dir", required=True, help="Carpeta Firmas_automaticas origen.")
    parser.add_argument("--export-dir", required=True, help="Carpeta donde guardar el export ligero.")
    parser.add_argument("--zip", action="store_true", help="Crea tambien un .zip del export.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    export_dir = Path(args.export_dir)
    cubes_dir = output_dir / "cubos"
    if not cubes_dir.exists():
        raise SystemExit(f"No existe la carpeta de cubos: {cubes_dir}")

    export_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cube_dir in sorted(cubes_dir.iterdir()):
        if not cube_dir.is_dir():
            continue
        row = export_cube(cube_dir, export_dir)
        if row is not None:
            rows.append(row)

    pd.DataFrame(rows).to_csv(export_dir / "caracteristicas_cubos.csv", index=False)
    summary = {
        "output_dir_origen": str(output_dir),
        "cubos_exportados": len(rows),
        "incluye_npy_originales": False,
        "incluye_mascaras_completas": False,
    }
    (export_dir / "resumen_export.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.zip:
        zip_path = export_dir.with_suffix(".zip")
        zip_directory(export_dir, zip_path)
        print(f"ZIP: {zip_path}")

    print(f"Cubos exportados: {len(rows)}")
    print(f"Export ligero: {export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
