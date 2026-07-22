r"""
Comparar distintas configuraciones de rangos de preview para segmentacion.

La idea es correr el pipeline automatico varias veces con diferentes
--preview-ranges, aplicar control de calidad y comparar cuantos cubos quedan
como buenas/revisar/malas.

Ejemplo:

python .\comparar_rangos_segmentacion.py ^
  --input-dir "E:\JavierSTSIVA" ^
  --base-output ".\Firmas_automaticas_pruebas_rangos" ^
  --layout y_lambda_x ^
  --recipes "180:320,250:430,390:550,700:820;180:320,250:430;250:430,390:550;700:820" ^
  --limit 50

Cada recipe separada por ; crea una carpeta propia.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd


def recipe_name(recipe: str) -> str:
    cleaned = (
        recipe.replace(":", "_")
        .replace(",", "__")
        .replace(" ", "")
        .replace("-", "_")
    )
    return f"rangos_{cleaned}"


def run_command(command: list[str], cwd: Path) -> None:
    print(" ".join(command))
    completed = subprocess.run(command, cwd=cwd)
    if completed.returncode != 0:
        raise RuntimeError(f"Comando fallo con codigo {completed.returncode}")


def summarize_qc(qc_csv: Path, recipe: str, output_dir: Path) -> dict[str, object]:
    df = pd.read_csv(qc_csv)
    counts = df["status_qc"].value_counts().to_dict()
    return {
        "recipe": recipe,
        "output_dir": str(output_dir),
        "total": int(len(df)),
        "buenas": int(counts.get("ok", 0)),
        "revisar": int(counts.get("revisar", 0)),
        "malas": int(counts.get("mala", 0)),
        "score_mediano": float(df["score_malo"].median()),
        "score_promedio": float(df["score_malo"].mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compara rangos de preview para segmentacion automatica.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--base-output", type=Path, default=Path("Firmas_automaticas_pruebas_rangos"))
    parser.add_argument("--layout", default="y_lambda_x", choices=["y_lambda_x", "y_x_lambda"])
    parser.add_argument(
        "--recipes",
        required=True,
        help="Configuraciones separadas por ;. Ej: 180:320,250:430;250:430,390:550",
    )
    parser.add_argument("--n-subranges", type=int, default=3)
    parser.add_argument("--k-values", default="4,5")
    parser.add_argument("--soil-radius-pixels", type=float, default=90.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    args.base_output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    recipes = [item.strip() for item in args.recipes.split(";") if item.strip()]
    for recipe in recipes:
        out_dir = args.base_output / recipe_name(recipe)
        command = [
            sys.executable,
            str(script_dir / "automatizar_firmas_cubos.py"),
            "--input-dir",
            str(args.input_dir),
            "--output-dir",
            str(out_dir),
            "--layout",
            args.layout,
            "--preview-ranges",
            recipe,
            "--n-subranges",
            str(args.n_subranges),
            "--k-values",
            args.k_values,
            "--soil-radius-pixels",
            str(args.soil_radius_pixels),
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        run_command(command, cwd=root)

        qc_command = [
            sys.executable,
            str(script_dir / "detectar_firmas_malas.py"),
            "--input-dir",
            str(out_dir),
            "--cubos-subdir",
            "cubos",
            "--ignore-recortes",
        ]
        run_command(qc_command, cwd=root)
        rows.append(summarize_qc(out_dir / "control_calidad_firmas" / "resumen_control_calidad.csv", recipe, out_dir))

    summary_path = args.base_output / "resumen_comparacion_rangos.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Resumen: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
