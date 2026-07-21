r"""
Seleccion manual de ROIs SOIL / WHITE / DARK para cubos .npy.

Este script es el equivalente manual del flujo automatico. Permite corregir
cubos donde la segmentacion falla y guarda una salida compatible con:

    seleccionar_tramos_uso_firmas.py

Salida por cubo:

    Firmas_automaticas/cubos_manuales/<cube_id>/resultado.npz
    Firmas_automaticas/cubos_manuales/<cube_id>/metadata.json
    Firmas_automaticas/cubos_manuales/<cube_id>/diagnostico.png
    Firmas_automaticas/cubos_manuales/<cube_id>/firmas.csv

Uso recomendado:

    python .\Spectral_Reconstruction\seleccionar_firmas_manual.py ^
      --cube-path ".\Spectral_Reconstruction\Capturas_soil\Soil_1\cube_20260617_171738.npy" ^
      --output-dir ".\Spectral_Reconstruction\Firmas_automaticas" ^
      --layout y_lambda_x ^
      --preview-start 390 ^
      --preview-stop 550 ^
      --overwrite

Controles:
  - Para cada ROI, haz clic alrededor de la region.
  - Presiona ENTER cuando termines el poligono.
  - Cierra la ventana solo cuando el script lo indique.

Convencion interna:
  El cubo se maneja como (fila, banda, columna).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath


ROLES = ("soil", "white", "dark")
ROLE_COLORS = {
    "soil": "#f0c419",
    "white": "#ffffff",
    "dark": "#00b7ff",
}


@dataclass
class ManualResult:
    cube_id: str
    input_path: str
    status: str
    reason: str
    shape_y: int
    shape_x: int
    bands: int
    soil_pixels: int
    white_pixels: int
    dark_pixels: int
    invalid_reflectance_fraction: float
    reflectance_outside_fraction: float
    preview_start: int
    preview_stop: int


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.zeros_like(image, dtype=np.float32)
    p1, p99 = np.nanpercentile(image[finite], [1, 99])
    if p99 <= p1:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - p1) / (p99 - p1), 0, 1).astype(np.float32)


def load_cube(path: Path, layout: str) -> np.ndarray:
    cube = np.load(path)
    if cube.ndim != 3:
        raise ValueError(f"El cubo debe ser 3D. Forma encontrada: {cube.shape}")
    if layout == "y_lambda_x":
        return cube.astype(np.float32, copy=False)
    if layout == "y_x_lambda":
        return np.moveaxis(cube, 2, 1).astype(np.float32, copy=False)
    raise ValueError(f"Layout no soportado: {layout}")


def cube_id_from_path(path: Path) -> str:
    return f"{path.parent.name}__{path.stem}"


def make_preview(cube: np.ndarray, start: int, stop: int) -> tuple[np.ndarray, tuple[int, int]]:
    bands = cube.shape[1]
    start = max(0, min(start, bands - 1))
    stop = max(start + 1, min(stop, bands))
    preview = np.nanmean(cube[:, start:stop, :], axis=1)
    return normalize_image(preview), (start, stop)


def polygon_to_mask(points: list[tuple[float, float]], shape: tuple[int, int]) -> np.ndarray:
    if len(points) < 3:
        raise ValueError("El poligono debe tener al menos 3 puntos.")
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    coords = np.column_stack((xx.ravel(), yy.ravel()))
    polygon = MplPath(points)
    return polygon.contains_points(coords).reshape(shape)


def select_polygon(preview: np.ndarray, role: str) -> tuple[np.ndarray, list[tuple[float, float]]]:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(preview, cmap="turbo")
    ax.set_title(
        f"Seleccione ROI {role.upper()} | clics alrededor de la region, ENTER para terminar",
        fontsize=11,
    )
    ax.set_xlabel("Columna espacial")
    ax.set_ylabel("Fila espacial")
    ax.grid(False)
    print(f"\nSeleccione ROI {role.upper()}: haga clics y presione ENTER al terminar.")
    points = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close(fig)
    mask = polygon_to_mask(points, preview.shape)
    if int(mask.sum()) == 0:
        raise ValueError(f"La mascara {role} quedo vacia.")
    return mask, [(float(x), float(y)) for x, y in points]


def signature_from_mask(cube: np.ndarray, mask: np.ndarray, reduction: str) -> np.ndarray:
    # cube esta como (fila, banda, columna). Para aplicar una mascara espacial
    # (fila, columna), movemos bandas al final: (fila, columna, banda).
    pixels = np.moveaxis(cube, 1, 2)[mask]
    if pixels.size == 0:
        raise ValueError("Mascara vacia.")
    if reduction == "mean":
        return np.nanmean(pixels, axis=0).astype(np.float32)
    return np.nanmedian(pixels, axis=0).astype(np.float32)


def compute_reflectance(soil: np.ndarray, white: np.ndarray, dark: np.ndarray) -> tuple[np.ndarray, float, float]:
    denom = white - dark
    valid_denom = np.abs(denom) > np.finfo(np.float32).eps
    reflectance = np.full_like(soil, np.nan, dtype=np.float32)
    reflectance[valid_denom] = (soil[valid_denom] - dark[valid_denom]) / denom[valid_denom]
    invalid_fraction = float(np.mean(~np.isfinite(reflectance)))
    finite = reflectance[np.isfinite(reflectance)]
    outside_fraction = float(np.mean((finite < 0) | (finite > 1.5))) if finite.size else 1.0
    return reflectance, invalid_fraction, outside_fraction


def save_csv(path: Path, soil: np.ndarray, white: np.ndarray, dark: np.ndarray, reflectance: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["band_index", "soil_signature", "white_signature", "dark_signature", "soil_reflectance"])
        for i, values in enumerate(zip(soil, white, dark, reflectance)):
            writer.writerow([i, *[float(v) if np.isfinite(v) else "" for v in values]])


def save_diagnostic(
    path: Path,
    preview: np.ndarray,
    masks: dict[str, np.ndarray],
    reflectance: np.ndarray,
    preview_range: tuple[int, int],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].imshow(preview, cmap="turbo")
    for role, mask in masks.items():
        axes[0].contour(mask.astype(float), levels=[0.5], colors=[ROLE_COLORS[role]], linewidths=1.5)
    axes[0].set_title(f"ROIs manuales | preview {preview_range[0]}:{preview_range[1]}")
    axes[0].set_xlabel("Columna espacial")
    axes[0].set_ylabel("Fila espacial")

    axes[1].plot(np.arange(reflectance.size), reflectance, color="#8b5a2b", linewidth=1.2)
    axes[1].set_title("Reflectancia SOIL manual")
    axes[1].set_xlabel("Indice espectral")
    axes[1].set_ylabel("Reflectancia")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def process_cube(
    cube_path: Path,
    output_dir: Path,
    results_subdir: str,
    layout: str,
    preview_start: int,
    preview_stop: int,
    reduction: str,
    overwrite: bool,
) -> ManualResult:
    cube = load_cube(cube_path, layout)
    preview, preview_range = make_preview(cube, preview_start, preview_stop)
    cube_id = cube_id_from_path(cube_path)
    cube_out = output_dir / results_subdir / cube_id
    result_path = cube_out / "resultado.npz"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"Ya existe {result_path}. Use --overwrite para reemplazar.")
    cube_out.mkdir(parents=True, exist_ok=True)

    masks: dict[str, np.ndarray] = {}
    polygons: dict[str, list[tuple[float, float]]] = {}
    for role in ROLES:
        mask, points = select_polygon(preview, role)
        masks[role] = mask
        polygons[role] = points
        print(f"ROI {role}: {int(mask.sum())} pixeles")

    soil = signature_from_mask(cube, masks["soil"], reduction)
    white = signature_from_mask(cube, masks["white"], reduction)
    dark = signature_from_mask(cube, masks["dark"], reduction)
    reflectance, invalid_fraction, outside_fraction = compute_reflectance(soil, white, dark)

    np.savez_compressed(
        result_path,
        soil_signature=soil,
        white_signature=white,
        dark_signature=dark,
        soil_reflectance=reflectance,
        soil_mask=masks["soil"],
        white_mask=masks["white"],
        dark_mask=masks["dark"],
        preview=preview.astype(np.float32),
        preview_range=np.array(preview_range, dtype=np.int32),
        preview_subranges=np.array([preview_range], dtype=np.int32),
    )
    save_csv(cube_out / "firmas.csv", soil, white, dark, reflectance)
    save_diagnostic(cube_out / "diagnostico.png", preview, masks, reflectance, preview_range)

    status = "ok"
    reason = ""
    if invalid_fraction > 0.2:
        status = "review"
        reason = "muchas_bandas_invalidas"

    result = ManualResult(
        cube_id=cube_id,
        input_path=str(cube_path),
        status=status,
        reason=reason,
        shape_y=int(cube.shape[0]),
        shape_x=int(cube.shape[2]),
        bands=int(cube.shape[1]),
        soil_pixels=int(masks["soil"].sum()),
        white_pixels=int(masks["white"].sum()),
        dark_pixels=int(masks["dark"].sum()),
        invalid_reflectance_fraction=invalid_fraction,
        reflectance_outside_fraction=outside_fraction,
        preview_start=int(preview_range[0]),
        preview_stop=int(preview_range[1]),
    )

    metadata = asdict(result)
    metadata["selection_mode"] = "manual"
    metadata["reduction"] = reduction
    metadata["polygons"] = polygons
    with (cube_out / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(f"Guardado: {result_path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Seleccion manual de ROIs para extraer firmas de reflectancia.")
    parser.add_argument("--cube-path", type=Path, required=True, help="Ruta del cubo .npy a corregir manualmente.")
    parser.add_argument("--output-dir", type=Path, default=Path("Firmas_automaticas"))
    parser.add_argument(
        "--results-subdir",
        default="cubos_manuales",
        help="Subcarpeta dentro de output-dir donde se guardan los resultados manuales.",
    )
    parser.add_argument("--layout", choices=["y_lambda_x", "y_x_lambda"], default="y_lambda_x")
    parser.add_argument("--preview-start", type=int, default=390)
    parser.add_argument("--preview-stop", type=int, default=550)
    parser.add_argument("--reduction", choices=["median", "mean"], default="median")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    process_cube(
        cube_path=args.cube_path,
        output_dir=args.output_dir,
        results_subdir=args.results_subdir,
        layout=args.layout,
        preview_start=args.preview_start,
        preview_stop=args.preview_stop,
        reduction=args.reduction,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
