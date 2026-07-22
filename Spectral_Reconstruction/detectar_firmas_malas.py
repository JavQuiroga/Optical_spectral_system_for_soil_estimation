r"""
Detectar automaticamente firmas/segmentaciones probablemente malas.

Este script NO corrige firmas. Hace un triage para que no tengas que mirar
diagnostico por diagnostico.

Entrada tipica:
    Firmas_automaticas/

Lee:
    cubos/*/metadata.json
    cubos/*/resultado.npz
    recortes_firmas/resumen_recortes.csv              (si existe)

Genera:
    Firmas_automaticas/control_calidad_firmas/
        resumen_control_calidad.csv
        revisar.txt
        diagnosticos_por_estado/
            buenas/
            revisar/
            malas/

Estados:
    ok       -> probablemente usable
    revisar  -> conviene mirar diagnostico
    mala     -> muy probablemente requiere correccion manual
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class QualityResult:
    cube_id: str
    status_qc: str
    score_malo: float
    reasons: str
    source_status: str
    source_reason: str
    confidence: float
    selected_k: int | None
    invalid_reflectance_fraction: float
    reflectance_outside_fraction: float
    reflectance_nan_fraction: float
    reflectance_outside_0_15_fraction: float
    reflectance_roughness_100_900: float
    reflectance_median_100_900: float
    reflectance_std_100_900: float
    soil_centroid_x: float
    soil_centroid_y: float
    soil_expected_distance: float
    soil_area_fraction: float
    soil_border_fraction: float
    soil_fill_fraction: float
    soil_brightness: float
    white_brightness: float
    white_soil_brightness_gap: float
    crop_status: str
    crop_reason: str
    crop_finite_fraction: float
    crop_roughness: float
    diagnostic_path: str
    metadata_path: str


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def robust_roughness(y: np.ndarray) -> float:
    finite = np.isfinite(y)
    if np.count_nonzero(finite) < 5:
        return float("nan")
    values = y[finite].astype(float)
    scale = np.nanpercentile(values, 95) - np.nanpercentile(values, 5)
    scale = max(float(scale), 1e-9)
    return float(np.nanmedian(np.abs(np.diff(values))) / scale)


def load_crop_summary(output_dir: Path) -> dict[str, dict[str, object]]:
    path = output_dir / "recortes_firmas" / "resumen_recortes.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {str(row["cube_id"]): row.to_dict() for _, row in df.iterrows()}


def classify(score: float) -> str:
    if score >= 65:
        return "mala"
    if score >= 30:
        return "revisar"
    return "ok"


def add_reason(reasons: list[str], condition: bool, label: str, points: float) -> float:
    if condition:
        reasons.append(label)
        return points
    return 0.0


def evaluate_cube(
    cube_dir: Path,
    crop_by_id: dict[str, dict[str, object]],
    expected_soil: tuple[float, float],
    require_recorte: bool,
) -> QualityResult:
    cube_id = cube_dir.name
    metadata_path = cube_dir / "metadata.json"
    result_path = cube_dir / "resultado.npz"
    diagnostic_path = cube_dir / "diagnostico.png"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    selected = metadata.get("selected_components", {})
    soil = selected.get("soil", {})
    white = selected.get("white", {})
    segmentation = metadata.get("segmentation", {})

    confidence = safe_float(metadata.get("confidence"))
    source_status = str(metadata.get("status", ""))
    source_reason = str(metadata.get("reason", ""))
    invalid_fraction = safe_float(metadata.get("invalid_reflectance_fraction"), 1.0)
    outside_fraction = safe_float(metadata.get("reflectance_outside_fraction"), 1.0)

    soil_x = safe_float(soil.get("centroid_x"))
    soil_y = safe_float(soil.get("centroid_y"))
    soil_distance = (
        math.hypot(soil_x - expected_soil[0], soil_y - expected_soil[1])
        if np.isfinite(soil_x) and np.isfinite(soil_y)
        else float("inf")
    )
    soil_area = safe_float(soil.get("area_fraction"))
    soil_border = safe_float(soil.get("border_fraction"))
    soil_fill = safe_float(soil.get("fill_fraction"))
    soil_brightness = safe_float(soil.get("brightness"))
    white_brightness = safe_float(white.get("brightness"))
    brightness_gap = white_brightness - soil_brightness

    reflectance_nan_fraction = 1.0
    reflectance_outside_0_15_fraction = 1.0
    reflectance_roughness = float("nan")
    reflectance_median = float("nan")
    reflectance_std = float("nan")
    if result_path.exists():
        with np.load(result_path) as data:
            reflectance = data["soil_reflectance"].astype(float).reshape(-1)
        region = reflectance[100:min(900, reflectance.size)]
        finite = np.isfinite(region)
        reflectance_nan_fraction = float(1.0 - np.mean(finite)) if region.size else 1.0
        finite_values = region[finite]
        if finite_values.size:
            reflectance_outside_0_15_fraction = float(np.mean((finite_values < 0) | (finite_values > 1.5)))
            reflectance_roughness = robust_roughness(region)
            reflectance_median = float(np.nanmedian(finite_values))
            reflectance_std = float(np.nanstd(finite_values))

    crop = crop_by_id.get(cube_id, {})
    crop_status = str(crop.get("status", "sin_recorte"))
    crop_reason = str(crop.get("reason", ""))
    crop_finite = safe_float(crop.get("finite_fraction_crop"))
    crop_roughness = safe_float(crop.get("roughness_crop"))

    reasons: list[str] = []
    score = 0.0
    score += add_reason(reasons, source_status == "review", f"metadata_review:{source_reason}", 25)
    score += add_reason(reasons, confidence < 0.70, "confianza_baja", 25)
    score += add_reason(reasons, confidence < 0.76, "confianza_media_baja", 12)
    score += add_reason(reasons, confidence < 0.55, "confianza_muy_baja", 20)
    score += add_reason(reasons, invalid_fraction > 0.20, "muchas_reflectancias_invalidas_metadata", 35)
    score += add_reason(reasons, outside_fraction > 0.10, "reflectancia_fuera_rango_metadata", 25)
    score += add_reason(reasons, reflectance_nan_fraction > 0.15, "muchos_nan_100_900", 35)
    score += add_reason(reasons, reflectance_outside_0_15_fraction > 0.10, "reflectancia_fuera_0_1p5", 25)
    score += add_reason(reasons, reflectance_outside_0_15_fraction > 0.02, "reflectancia_fuera_0_1p5_suave", 10)
    score += add_reason(reasons, soil_distance > 0.23, "soil_lejos_zona_esperada", 30)
    score += add_reason(reasons, soil_area < 0.09 and brightness_gap < 0.38, "soil_area_pequena_con_poco_contraste", 25)
    score += add_reason(reasons, soil_area < 0.015, "soil_area_muy_pequena", 30)
    score += add_reason(reasons, soil_area > 0.405, "soil_area_muy_grande", 25)
    score += add_reason(reasons, soil_area > 0.30 and soil_fill < 0.50, "soil_grande_y_poco_compacto", 15)
    score += add_reason(reasons, soil_border > 0.025 and soil_area > 0.30, "soil_grande_toca_borde", 15)
    score += add_reason(reasons, soil_border > 0.08, "soil_toca_borde", 20)
    score += add_reason(reasons, soil_fill < 0.26, "soil_poco_compacto", 25)
    score += add_reason(reasons, brightness_gap < 0.08, "white_no_muy_mayor_que_soil", 25)
    score += add_reason(reasons, brightness_gap < 0.36 and soil_brightness > 0.60, "patron_brillo_soil_sospechoso", 15)
    score += add_reason(reasons, np.isfinite(reflectance_roughness) and reflectance_roughness > 0.08, "firma_muy_rugosa", 15)
    score += add_reason(reasons, np.isfinite(reflectance_std) and reflectance_std > 0.20, "reflectancia_std_alta", 15)
    score += add_reason(reasons, np.isfinite(reflectance_std) and reflectance_std > 0.40, "reflectancia_std_muy_alta", 20)
    score += add_reason(reasons, np.isfinite(reflectance_std) and reflectance_std > 0.70, "reflectancia_std_extrema", 20)
    if crop_status != "sin_recorte":
        score += add_reason(reasons, crop_status == "review", f"recorte_review:{crop_reason}", 20)
        score += add_reason(reasons, np.isfinite(crop_finite) and crop_finite < 0.90, "recorte_con_nan", 25)
    elif require_recorte:
        score += add_reason(reasons, True, "sin_recorte", 15)

    return QualityResult(
        cube_id=cube_id,
        status_qc=classify(score),
        score_malo=float(score),
        reasons=";".join(reasons),
        source_status=source_status,
        source_reason=source_reason,
        confidence=confidence,
        selected_k=segmentation.get("selected_k"),
        invalid_reflectance_fraction=invalid_fraction,
        reflectance_outside_fraction=outside_fraction,
        reflectance_nan_fraction=reflectance_nan_fraction,
        reflectance_outside_0_15_fraction=reflectance_outside_0_15_fraction,
        reflectance_roughness_100_900=reflectance_roughness,
        reflectance_median_100_900=reflectance_median,
        reflectance_std_100_900=reflectance_std,
        soil_centroid_x=soil_x,
        soil_centroid_y=soil_y,
        soil_expected_distance=float(soil_distance),
        soil_area_fraction=soil_area,
        soil_border_fraction=soil_border,
        soil_fill_fraction=soil_fill,
        soil_brightness=soil_brightness,
        white_brightness=white_brightness,
        white_soil_brightness_gap=float(brightness_gap),
        crop_status=crop_status,
        crop_reason=crop_reason,
        crop_finite_fraction=crop_finite,
        crop_roughness=crop_roughness,
        diagnostic_path=str(diagnostic_path),
        metadata_path=str(metadata_path),
    )


def write_results(path: Path, results: list[QualityResult]) -> None:
    if not results:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def copy_diagnostics(output_dir: Path, results: list[QualityResult]) -> None:
    base_dir = output_dir / "diagnosticos_por_estado"
    state_dirs = {
        "ok": base_dir / "buenas",
        "revisar": base_dir / "revisar",
        "mala": base_dir / "malas",
    }
    for directory in state_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    for result in results:
        dst_dir = state_dirs.get(result.status_qc, base_dir / result.status_qc)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{result.cube_id}__score_{int(result.score_malo):03d}.png"
        try:
            save_qc_panel(dst, result)
        except Exception:
            src = Path(result.diagnostic_path)
            if src.exists():
                shutil.copy2(src, dst)


def load_selected_labels(cube_dir: Path, selected_k: int | None) -> np.ndarray | None:
    if selected_k is None or not np.isfinite(selected_k):
        return None
    path = cube_dir / "por_k" / f"k_{int(selected_k):02d}" / f"resultado_k_{int(selected_k):02d}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        if "labels" not in data:
            return None
        return data["labels"].copy()


def save_qc_panel(path: Path, result: QualityResult) -> None:
    metadata_path = Path(result.metadata_path)
    cube_dir = metadata_path.parent
    result_path = cube_dir / "resultado.npz"
    if not result_path.exists():
        raise FileNotFoundError(result_path)

    with np.load(result_path) as data:
        preview = data["preview"].astype(float)
        reflectance = data["soil_reflectance"].astype(float).reshape(-1)
        masks = {
            "soil": data["soil_mask"].astype(bool),
            "white": data["white_mask"].astype(bool),
            "dark": data["dark_mask"].astype(bool),
        }
    labels = load_selected_labels(cube_dir, result.selected_k)

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.82])
    ax_seg = fig.add_subplot(gs[0, 0])
    ax_roi = fig.add_subplot(gs[0, 1])
    ax_sig = fig.add_subplot(gs[1, :])

    if labels is not None:
        ax_seg.imshow(labels, cmap="tab20")
    else:
        ax_seg.imshow(preview, cmap="gray", vmin=0, vmax=1)
    ax_seg.set_title(f"Segmentacion K-means | K={result.selected_k}")
    ax_seg.axis("off")

    ax_roi.imshow(preview, cmap="gray", vmin=0, vmax=1)
    colors = {"soil": "#f0c419", "white": "#ffffff", "dark": "#00b7ff"}
    labels_text = {"soil": "SOIL", "white": "WHITE", "dark": "DARK"}
    for role, mask in masks.items():
        ax_roi.contour(mask.astype(float), levels=[0.5], colors=[colors[role]], linewidths=1.7)
        yy, xx = np.nonzero(mask)
        if yy.size:
            ax_roi.text(
                float(np.mean(xx)),
                float(np.mean(yy)),
                labels_text[role],
                color=colors[role],
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2),
            )
    ax_roi.set_title(f"ROIs interiores | {result.status_qc} | score={result.score_malo:.0f}")
    ax_roi.axis("off")

    x = np.arange(reflectance.size)
    ax_sig.plot(x, reflectance, color="#8b5a2b", linewidth=1.15, label="Reflectancia SOIL")
    ax_sig.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_sig.axhline(1.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    finite = reflectance[np.isfinite(reflectance)]
    if finite.size:
        y_low, y_high = np.nanpercentile(finite, [1, 99])
        margin = max(0.05, 0.12 * (y_high - y_low))
        ax_sig.set_ylim(y_low - margin, y_high + margin)
    ax_sig.set_title("Firma de reflectancia extraida")
    ax_sig.set_xlabel("Indice espectral")
    ax_sig.set_ylabel("Reflectancia")
    ax_sig.grid(True, alpha=0.3)
    ax_sig.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"{result.cube_id} | {result.status_qc.upper()} | score={result.score_malo:.0f}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Detecta firmas o segmentaciones probablemente malas.")
    parser.add_argument("--input-dir", type=Path, default=Path("Firmas_automaticas"))
    parser.add_argument("--cubos-subdir", default="cubos")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-soil", default="0.43,0.55", help="Centro esperado de soil como x,y.")
    parser.add_argument(
        "--require-recorte",
        action="store_true",
        help="Penaliza cubos sin recorte. Por defecto no penaliza para poder correr control justo despues de segmentar.",
    )
    parser.add_argument(
        "--ignore-recortes",
        action="store_true",
        help="Ignora completamente resumen_recortes.csv aunque exista.",
    )
    args = parser.parse_args()

    expected_parts = [float(x.strip()) for x in args.expected_soil.split(",")]
    if len(expected_parts) != 2:
        raise ValueError("--expected-soil debe tener formato x,y")
    expected_soil = (expected_parts[0], expected_parts[1])

    input_dir = args.input_dir
    output_dir = args.output_dir or (input_dir / "control_calidad_firmas")
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_by_id = {} if args.ignore_recortes else load_crop_summary(input_dir)

    metadata_files = sorted((input_dir / args.cubos_subdir).glob("*/metadata.json"))
    if not metadata_files:
        print(f"No encontre metadata.json en {input_dir / args.cubos_subdir}")
        return 2

    results = [
        evaluate_cube(path.parent, crop_by_id, expected_soil, args.require_recorte)
        for path in metadata_files
    ]
    results = sorted(results, key=lambda item: item.score_malo, reverse=True)
    write_results(output_dir / "resumen_control_calidad.csv", results)
    copy_diagnostics(output_dir, results)

    review_or_bad = [item for item in results if item.status_qc != "ok"]
    with (output_dir / "revisar.txt").open("w", encoding="utf-8") as handle:
        for item in review_or_bad:
            handle.write(f"{item.status_qc}\tscore={item.score_malo:.1f}\t{item.cube_id}\t{item.reasons}\n")

    counts = {status: sum(item.status_qc == status for item in results) for status in ("ok", "revisar", "mala")}
    print(f"Analizados: {len(results)} | ok={counts['ok']} revisar={counts['revisar']} mala={counts['mala']}")
    print(f"Salida: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
