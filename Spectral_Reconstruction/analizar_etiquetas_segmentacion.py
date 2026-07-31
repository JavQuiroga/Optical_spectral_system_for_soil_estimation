r"""
Analiza segmentaciones etiquetadas manualmente como buenas/malas.

Entrada esperada:
    Firmas_automaticas/control_calidad_firmas/diagnosticos_por_estado/
        buenas/*.png
        malas/*.png

Lee, para cada cubo etiquetado:
    Firmas_automaticas/cubos/<cube_id>/metadata.json
    Firmas_automaticas/cubos/<cube_id>/resultado.npz

Genera:
    Firmas_automaticas/control_calidad_firmas/analisis_etiquetas/
        caracteristicas_etiquetadas.csv
        diferencias_buenas_malas.csv
        mejores_umbrales.csv
        resumen_por_receta.csv
        recomendaciones.txt
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CUBE_ID_RE = re.compile(r"(Soil_\d+__cube_\d+_\d+)")


@dataclass
class LabelItem:
    cube_id: str
    etiqueta_manual: str
    diagnostico_path: Path
    score_nombre: float


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


def parse_score_from_name(path: Path) -> float:
    match = re.search(r"score_(\d+)", path.stem)
    if not match:
        return float("nan")
    return float(match.group(1))


def find_label_items(label_dir: Path) -> list[LabelItem]:
    items: list[LabelItem] = []
    state_dir = label_dir / "diagnosticos_por_estado"
    for folder_name, label in (("buenas", "buena"), ("malas", "mala")):
        folder = state_dir / folder_name
        if not folder.exists():
            continue
        for image_path in sorted(folder.glob("*.png")):
            match = CUBE_ID_RE.search(image_path.name)
            if not match:
                continue
            items.append(
                LabelItem(
                    cube_id=match.group(1),
                    etiqueta_manual=label,
                    diagnostico_path=image_path,
                    score_nombre=parse_score_from_name(image_path),
                )
            )
    return items


def component_feature(
    row: dict[str, object],
    metadata: dict[str, object],
    role: str,
    expected: tuple[float, float] | None,
) -> None:
    components = metadata.get("selected_components", {})
    if not isinstance(components, dict):
        return
    component = components.get(role, {})
    if not isinstance(component, dict):
        return

    prefix = f"{role}_"
    x = safe_float(component.get("centroid_x"))
    y = safe_float(component.get("centroid_y"))
    row[prefix + "centroid_x"] = x
    row[prefix + "centroid_y"] = y
    row[prefix + "area_fraction"] = safe_float(component.get("area_fraction"))
    row[prefix + "brightness"] = safe_float(component.get("brightness"))
    row[prefix + "border_fraction"] = safe_float(component.get("border_fraction"))
    row[prefix + "fill_fraction"] = safe_float(component.get("fill_fraction"))

    if expected is not None and np.isfinite(x) and np.isfinite(y):
        row[prefix + "dist_expected"] = float(math.hypot(x - expected[0], y - expected[1]))


def summarize_attempts(row: dict[str, object], metadata: dict[str, object]) -> None:
    attempts = metadata.get("preview_recipe_attempts", [])
    if not isinstance(attempts, list) or not attempts:
        return

    ok_attempts = [
        item
        for item in attempts
        if isinstance(item, dict) and item.get("status") != "error"
    ]
    scores = [safe_float(item.get("recipe_score")) for item in ok_attempts]
    scores = [score for score in scores if np.isfinite(score)]
    row["num_preview_recipes_ok"] = len(ok_attempts)
    row["num_preview_recipes_error"] = len(attempts) - len(ok_attempts)
    if scores:
        sorted_scores = sorted(scores, reverse=True)
        row["preview_recipe_score_max"] = sorted_scores[0]
        row["preview_recipe_score_mean"] = float(np.mean(sorted_scores))
        row["preview_recipe_score_std"] = float(np.std(sorted_scores))
        row["preview_recipe_score_margin"] = (
            sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else float("nan")
        )

    selected_index = metadata.get("selected_preview_recipe_index")
    if selected_index is None:
        return
    for item in ok_attempts:
        if not isinstance(item, dict):
            continue
        if int(item.get("recipe_index", -1)) != int(selected_index):
            continue
        row["selected_recipe_confidence"] = safe_float(item.get("confidence"))
        row["selected_recipe_invalid_fraction"] = safe_float(
            item.get("invalid_reflectance_fraction")
        )
        row["selected_recipe_outside_fraction"] = safe_float(
            item.get("reflectance_outside_fraction")
        )
        ranges = item.get("ranges")
        if isinstance(ranges, list):
            row["selected_recipe_ranges"] = ";".join(
                f"{pair[0]}:{pair[1]}"
                for pair in ranges
                if isinstance(pair, list) and len(pair) == 2
            )


def find_cube_dir(item: LabelItem, output_dirs: list[Path]) -> Path:
    for output_dir in output_dirs:
        cube_dir = output_dir / "cubos" / item.cube_id
        if (cube_dir / "metadata.json").exists() or (cube_dir / "resultado.npz").exists():
            return cube_dir
    return output_dirs[0] / "cubos" / item.cube_id


def load_features(item: LabelItem, output_dirs: list[Path]) -> dict[str, object]:
    cube_dir = find_cube_dir(item, output_dirs)
    metadata_path = cube_dir / "metadata.json"
    result_path = cube_dir / "resultado.npz"

    row: dict[str, object] = {
        "cube_id": item.cube_id,
        "etiqueta_manual": item.etiqueta_manual,
        "es_mala": 1 if item.etiqueta_manual == "mala" else 0,
        "score_nombre": item.score_nombre,
        "diagnostico_path": str(item.diagnostico_path),
        "metadata_path": str(metadata_path),
        "resultado_path": str(result_path),
        "metadata_existe": metadata_path.exists(),
        "resultado_existe": result_path.exists(),
    }

    if not metadata_path.exists():
        return row

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = metadata.get("config", {})
    if not isinstance(config, dict):
        config = {}

    expected_soil = config.get("expected_soil")
    expected_white = config.get("expected_white")
    if isinstance(expected_soil, list) and len(expected_soil) == 2:
        expected_soil_tuple = (float(expected_soil[0]), float(expected_soil[1]))
    else:
        expected_soil_tuple = None
    if isinstance(expected_white, list) and len(expected_white) == 2:
        expected_white_tuple = (float(expected_white[0]), float(expected_white[1]))
    else:
        expected_white_tuple = None

    segmentation = metadata.get("segmentation", {})
    if not isinstance(segmentation, dict):
        segmentation = {}
    role_scores = metadata.get("role_scores", {})
    if not isinstance(role_scores, dict):
        role_scores = {}

    row.update(
        {
            "status_auto": metadata.get("status"),
            "reason_auto": metadata.get("reason"),
            "confidence": safe_float(metadata.get("confidence")),
            "invalid_reflectance_fraction": safe_float(
                metadata.get("invalid_reflectance_fraction")
            ),
            "reflectance_outside_fraction": safe_float(
                metadata.get("reflectance_outside_fraction")
            ),
            "selected_k": segmentation.get("selected_k"),
            "selected_quality": safe_float(segmentation.get("selected_quality")),
            "selected_preview_recipe_index": metadata.get("selected_preview_recipe_index"),
            "selected_preview_recipe_score": safe_float(
                segmentation.get("selected_preview_recipe_score")
            ),
            "role_score_soil": safe_float(role_scores.get("soil")),
            "role_score_white": safe_float(role_scores.get("white")),
            "role_score_dark": safe_float(role_scores.get("dark")),
        }
    )
    component_feature(row, metadata, "soil", expected_soil_tuple)
    component_feature(row, metadata, "white", expected_white_tuple)
    summarize_attempts(row, metadata)

    if not result_path.exists():
        return row

    with np.load(result_path, allow_pickle=False) as data:
        reflectance = data["soil_reflectance"].astype(float)
        soil = data["soil_signature"].astype(float)
        white = data["white_signature"].astype(float)
        dark = data["dark_signature"].astype(float)
        preview = data["preview"].astype(float)
        soil_mask = data["soil_mask"].astype(bool)
        white_mask = data["white_mask"].astype(bool)
        dark_mask = data["dark_mask"].astype(bool)

    usable = reflectance[100:900]
    denom = (white - dark)[100:900]
    soil_raw = soil[100:900]
    white_raw = white[100:900]
    dark_raw = dark[100:900]

    row.update(
        {
            "reflectance_nan_fraction_100_900": float(np.mean(~np.isfinite(usable))),
            "reflectance_outside_0_15_fraction_100_900": float(
                np.mean((usable < 0) | (usable > 1.5) | ~np.isfinite(usable))
            ),
            "reflectance_median_100_900": float(np.nanmedian(usable)),
            "reflectance_std_100_900": float(np.nanstd(usable)),
            "reflectance_p05_100_900": float(np.nanpercentile(usable, 5)),
            "reflectance_p95_100_900": float(np.nanpercentile(usable, 95)),
            "reflectance_roughness_100_900": robust_roughness(usable),
            "denominator_median_100_900": float(np.nanmedian(denom)),
            "denominator_p05_100_900": float(np.nanpercentile(denom, 5)),
            "denominator_nonpositive_fraction_100_900": float(np.mean(denom <= 0)),
            "soil_raw_median_100_900": float(np.nanmedian(soil_raw)),
            "white_raw_median_100_900": float(np.nanmedian(white_raw)),
            "dark_raw_median_100_900": float(np.nanmedian(dark_raw)),
            "white_minus_soil_raw_median_100_900": float(
                np.nanmedian(white_raw - soil_raw)
            ),
            "soil_pixels_final": int(np.count_nonzero(soil_mask)),
            "white_pixels_final": int(np.count_nonzero(white_mask)),
            "dark_pixels_final": int(np.count_nonzero(dark_mask)),
            "preview_soil_median": float(np.nanmedian(preview[soil_mask])),
            "preview_white_median": float(np.nanmedian(preview[white_mask])),
            "preview_dark_median": float(np.nanmedian(preview[dark_mask])),
            "preview_white_soil_gap": float(
                np.nanmedian(preview[white_mask]) - np.nanmedian(preview[soil_mask])
            ),
        }
    )
    return row


def numeric_columns(df: pd.DataFrame) -> list[str]:
    ignored = {"es_mala"}
    cols: list[str] = []
    for col in df.columns:
        if col in ignored:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            valid = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) >= 10 and valid.nunique() > 1:
                cols.append(col)
    return cols


def compare_groups(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in numeric_columns(df):
        good = df.loc[df["es_mala"] == 0, col].replace([np.inf, -np.inf], np.nan).dropna()
        bad = df.loc[df["es_mala"] == 1, col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(good) < 5 or len(bad) < 5:
            continue
        pooled = math.sqrt(float((good.var(ddof=1) + bad.var(ddof=1)) / 2))
        effect = (float(bad.mean()) - float(good.mean())) / max(pooled, 1e-12)
        rows.append(
            {
                "feature": col,
                "media_buenas": float(good.mean()),
                "media_malas": float(bad.mean()),
                "mediana_buenas": float(good.median()),
                "mediana_malas": float(bad.median()),
                "effect_mala_menos_buena": effect,
                "abs_effect": abs(effect),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_effect", ascending=False)


def best_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    y = df["es_mala"].to_numpy(dtype=int)
    rows = []
    for col in numeric_columns(df):
        x = df[col].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        finite = np.isfinite(x)
        if np.count_nonzero(finite) < 20:
            continue
        values = np.unique(np.nanpercentile(x[finite], np.linspace(5, 95, 91)))
        for direction in ("mayor", "menor"):
            best = None
            for threshold in values:
                pred = x >= threshold if direction == "mayor" else x <= threshold
                pred = pred.astype(int)
                pred[~finite] = 0
                tp = int(np.sum((pred == 1) & (y == 1)))
                tn = int(np.sum((pred == 0) & (y == 0)))
                fp = int(np.sum((pred == 1) & (y == 0)))
                fn = int(np.sum((pred == 0) & (y == 1)))
                tpr = tp / max(tp + fn, 1)
                tnr = tn / max(tn + fp, 1)
                precision = tp / max(tp + fp, 1)
                balanced_accuracy = 0.5 * (tpr + tnr)
                f1 = 2 * precision * tpr / max(precision + tpr, 1e-12)
                score = 0.55 * balanced_accuracy + 0.45 * f1
                item = (score, threshold, balanced_accuracy, f1, tpr, tnr, tp, fp, fn, tn)
                if best is None or item[0] > best[0]:
                    best = item
            if best is not None:
                rows.append(
                    {
                        "feature": col,
                        "direccion_mala_si": direction,
                        "umbral": float(best[1]),
                        "balanced_accuracy": float(best[2]),
                        "f1_malas": float(best[3]),
                        "recall_malas": float(best[4]),
                        "recall_buenas": float(best[5]),
                        "tp_malas": int(best[6]),
                        "fp_buenas_marcadas_malas": int(best[7]),
                        "fn_malas_no_detectadas": int(best[8]),
                        "tn_buenas": int(best[9]),
                        "score": float(best[0]),
                    }
                )
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def write_plots(df: pd.DataFrame, differences: pd.DataFrame, out_dir: Path) -> None:
    top = differences.head(8)["feature"].tolist()
    if not top:
        return
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.ravel()
    for ax, col in zip(axes, top):
        good = df.loc[df["es_mala"] == 0, col].replace([np.inf, -np.inf], np.nan).dropna()
        bad = df.loc[df["es_mala"] == 1, col].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(good, bins=30, alpha=0.55, label="buenas", color="#2ca02c")
        ax.hist(bad, bins=30, alpha=0.55, label="malas", color="#d62728")
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in axes[len(top) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout(rect=(0, 0, 0.98, 0.95))
    fig.savefig(out_dir / "histogramas_caracteristicas_top.png", dpi=160)
    plt.close(fig)


def write_recommendations(
    df: pd.DataFrame,
    differences: pd.DataFrame,
    thresholds: pd.DataFrame,
    out_dir: Path,
) -> None:
    total = len(df)
    n_bad = int(df["es_mala"].sum())
    n_good = total - n_bad
    lines = [
        "Analisis de etiquetas manuales buenas/malas",
        "",
        f"Cubos analizados: {total}",
        f"Buenas: {n_good}",
        f"Malas: {n_bad}",
        "",
        "Variables que mas cambian entre buenas y malas:",
    ]
    for _, row in differences.head(10).iterrows():
        lines.append(
            "- {feature}: mediana buenas={good:.4g}, mediana malas={bad:.4g}, efecto={effect:.2f}".format(
                feature=row["feature"],
                good=row["mediana_buenas"],
                bad=row["mediana_malas"],
                effect=row["effect_mala_menos_buena"],
            )
        )

    lines.extend(["", "Umbrales individuales mas utiles para detectar malas:"])
    for _, row in thresholds.head(10).iterrows():
        symbol = ">=" if row["direccion_mala_si"] == "mayor" else "<="
        lines.append(
            "- mala si {feature} {symbol} {threshold:.4g}: recall malas={rm:.2f}, recall buenas={rb:.2f}, F1={f1:.2f}".format(
                feature=row["feature"],
                symbol=symbol,
                threshold=row["umbral"],
                rm=row["recall_malas"],
                rb=row["recall_buenas"],
                f1=row["f1_malas"],
            )
        )

    lines.extend(
        [
            "",
            "Lectura practica para mejorar la automatica:",
            "- Usar estas variables para reordenar la eleccion de receta/K, no solo para castigar al final.",
            "- Penalizar recetas donde WHITE queda lejos de su zona esperada o con poco contraste frente a SOIL.",
            "- Favorecer recetas con margen claro entre la mejor y segunda mejor receta.",
            "- Mantener SOIL anclado cerca de su zona esperada y dejar que varie solo dentro de la ventana pequena.",
            "- Si las malas se explican por WHITE, conviene reforzar el score de WHITE mas que mover SOIL.",
        ]
    )
    (out_dir / "recomendaciones.txt").write_text("\n".join(lines), encoding="utf-8")


def truthy_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "si", "yes"})


def merge_external_features(df: pd.DataFrame, paths: list[str]) -> pd.DataFrame:
    merged = df.copy()
    protected = {
        "etiqueta_manual",
        "es_mala",
        "score_nombre",
        "diagnostico_path",
    }
    for path_text in paths:
        path = Path(path_text)
        if not path.exists():
            print(f"Advertencia: no existe feature-csv: {path}")
            continue
        extra = pd.read_csv(path)
        if "cube_id" not in extra.columns:
            print(f"Advertencia: feature-csv sin cube_id: {path}")
            continue
        extra = extra.drop_duplicates(subset=["cube_id"], keep="last")
        temp = merged.merge(extra, on="cube_id", how="left", suffixes=("", "__extra"))
        for col in extra.columns:
            if col == "cube_id" or col in protected:
                continue
            extra_col = f"{col}__extra" if col in merged.columns else col
            if extra_col not in temp.columns:
                continue
            if col not in temp.columns or col == extra_col:
                temp[col] = temp[extra_col]
            else:
                if col in {"metadata_existe", "resultado_existe"}:
                    temp[col] = truthy_series(temp[col]) | truthy_series(temp[extra_col])
                    temp = temp.drop(columns=[extra_col])
                    continue
                current_missing = temp[col].isna()
                temp.loc[current_missing, col] = temp.loc[current_missing, extra_col]
                temp = temp.drop(columns=[extra_col])
        extra_cols = [col for col in temp.columns if col.endswith("__extra")]
        if extra_cols:
            temp = temp.drop(columns=extra_cols)
        merged = temp
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extrae caracteristicas desde buenas/malas para mejorar segmentacion."
    )
    parser.add_argument(
        "--output-dir",
        default="Spectral_Reconstruction/Firmas_automaticas",
        help="Carpeta que contiene cubos/ y control_calidad_firmas/.",
    )
    parser.add_argument(
        "--extra-output-dir",
        action="append",
        default=[],
        help="Otra carpeta tipo Firmas_automaticas donde tambien buscar cubos/. Puede repetirse.",
    )
    parser.add_argument(
        "--feature-csv",
        action="append",
        default=[],
        help="CSV exportado por exportar_datos_ligeros_cubos.py. Puede repetirse.",
    )
    parser.add_argument(
        "--label-dir",
        default=None,
        help="Por defecto usa output-dir/control_calidad_firmas.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dirs = [output_dir] + [Path(path) for path in args.extra_output_dir]
    label_dir = Path(args.label_dir) if args.label_dir else output_dir / "control_calidad_firmas"
    out_dir = label_dir / "analisis_etiquetas"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = find_label_items(label_dir)
    if not items:
        raise SystemExit(f"No encontre PNG etiquetados en {label_dir}")

    rows = [load_features(item, output_dirs) for item in items]
    df = pd.DataFrame(rows)
    if args.feature_csv:
        df = merge_external_features(df, args.feature_csv)
    df.to_csv(out_dir / "caracteristicas_etiquetadas.csv", index=False)

    valid_df = df[truthy_series(df["metadata_existe"]) & truthy_series(df["resultado_existe"])].copy()
    if valid_df.empty:
        raise SystemExit("No encontre metadata/resultado para las etiquetas.")

    differences = compare_groups(valid_df)
    thresholds = best_thresholds(valid_df)
    differences.to_csv(out_dir / "diferencias_buenas_malas.csv", index=False)
    thresholds.to_csv(out_dir / "mejores_umbrales.csv", index=False)

    if "selected_recipe_ranges" in valid_df.columns:
        recipe_summary = (
            valid_df.groupby(["selected_recipe_ranges", "etiqueta_manual"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        recipe_summary["total"] = recipe_summary.select_dtypes(include=[np.number]).sum(axis=1)
        if "mala" in recipe_summary.columns:
            recipe_summary["fraccion_malas"] = recipe_summary["mala"] / recipe_summary["total"]
        recipe_summary.to_csv(out_dir / "resumen_por_receta.csv", index=False)

    write_plots(valid_df, differences, out_dir)
    write_recommendations(valid_df, differences, thresholds, out_dir)

    print(f"Etiquetas leidas: {len(df)}")
    print(f"Con metadata y resultado: {len(valid_df)}")
    print(f"Salidas en: {out_dir}")
    print("Top variables:")
    for _, row in differences.head(8).iterrows():
        print(
            f"  {row['feature']}: medianas buena/mala "
            f"{row['mediana_buenas']:.4g}/{row['mediana_malas']:.4g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
