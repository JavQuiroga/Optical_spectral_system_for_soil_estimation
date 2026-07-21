r"""
Seleccion automatica y visualizacion de tramos usables en firmas espectrales.

Entrada esperada:
    Firmas_automaticas/cubos/<id>/resultado.npz

Salida:
    Firmas_automaticas/recortes_firmas/
        resumen_recortes.csv
        CSV/
        NPY/
        PLOTS/

La seleccion busca el tramo mas estable de cada firma de reflectancia y lo
expande un poco antes y despues. Es un primer paso diagnostico: las graficas
permiten revisar si el tramo elegido es correcto antes de usarlo para contar
bandas utiles o cruzarlo con la curva de caracterizacion espectral.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


@dataclass
class CropResult:
    cube_id: str
    source_npz: str
    total_bands: int
    start: int
    stop: int
    usable_bands: int
    margin_bands: int
    search_start: int
    search_stop: int
    target_bands: int
    core_start: int
    core_stop: int
    finite_fraction_crop: float
    roughness_crop: float
    status: str
    reason: str


def robust_mad(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def fill_nan_linear(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    finite = np.isfinite(y)
    if finite.all():
        return y.copy()
    if not finite.any():
        return np.zeros_like(y)
    x = np.arange(y.size)
    filled = y.copy()
    filled[~finite] = np.interp(x[~finite], x[finite], y[finite])
    return filled


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="same")


def largest_true_component(mask: np.ndarray, min_len: int) -> tuple[int, int] | None:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return None

    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts = changes[0::2]
    stops = changes[1::2]
    lengths = stops - starts

    valid = lengths >= min_len
    if not np.any(valid):
        index = int(np.argmax(lengths))
    else:
        valid_indices = np.flatnonzero(valid)
        index = int(valid_indices[np.argmax(lengths[valid])])
    return int(starts[index]), int(stops[index])


def component_containing_index(mask: np.ndarray, index: int) -> tuple[int, int] | None:
    mask = np.asarray(mask, dtype=bool)
    if index < 0 or index >= mask.size or not mask[index]:
        return None
    start = index
    stop = index + 1
    while start > 0 and mask[start - 1]:
        start -= 1
    while stop < mask.size and mask[stop]:
        stop += 1
    return int(start), int(stop)


def select_usable_range(
    y: np.ndarray,
    margin_bands: int = 30,
    min_core_bands: int = 80,
    search_start: int = 100,
    search_stop: int = 900,
    target_bands: int = 300,
    smooth_sigma: float = 7.0,
    roughness_window: int = 31,
    derivative_multiplier: float = 8.0,
    roughness_multiplier: float = 8.0,
    valley_fraction: float = 0.55,
) -> tuple[int, int, int, int, dict[str, np.ndarray | float]]:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    n = y.size
    search_start = max(0, min(int(search_start), n - 1))
    search_stop = max(search_start + 1, min(int(search_stop), n))
    target_bands = max(1, min(int(target_bands), search_stop - search_start))
    finite = np.isfinite(y)
    y_filled = fill_nan_linear(y)
    smooth = ndimage.gaussian_filter1d(y_filled, sigma=smooth_sigma, mode="nearest")

    residual = np.abs(y_filled - smooth)
    roughness = rolling_mean(residual, roughness_window)
    derivative = np.r_[0, np.abs(np.diff(smooth))]

    finite_rate = rolling_mean(finite.astype(np.float32), roughness_window)
    rough_med = float(np.nanmedian(roughness))
    rough_mad = robust_mad(roughness)
    deriv_med = float(np.nanmedian(derivative))
    deriv_mad = robust_mad(derivative)

    rough_thr = rough_med + roughness_multiplier * max(rough_mad, 1e-6)
    deriv_thr = deriv_med + derivative_multiplier * max(deriv_mad, 1e-6)

    stable = (
        finite_rate > 0.95
        ) & (
        roughness <= rough_thr
        ) & (
        derivative <= deriv_thr
    )

    # Evitar que un borde artificial muy limpio gane solo por ser plano.
    edge_guard = max(5, min(40, n // 40))
    stable[:edge_guard] = False
    stable[-edge_guard:] = False
    stable[:search_start] = False
    stable[search_stop:] = False

    interior = slice(search_start, search_stop)
    valley_index = int(search_start + np.nanargmin(smooth[interior]))
    interior_values = smooth[interior]
    y_min = float(np.nanmin(interior_values))
    y_high = float(np.nanpercentile(interior_values, 85))
    valley_level = y_min + valley_fraction * max(y_high - y_min, 1e-6)

    valley_mask = (smooth <= valley_level) & (finite_rate > 0.80)
    valley_mask[:edge_guard] = False
    valley_mask[-edge_guard:] = False
    valley_mask[:search_start] = False
    valley_mask[search_stop:] = False

    component = component_containing_index(valley_mask, valley_index)
    if component is None:
        # Si el minimo cae justo en una discontinuidad, tomar el componente
        # de valle mas cercano.
        labeled, count = ndimage.label(valley_mask)
        if count > 0:
            centers = []
            for label in range(1, count + 1):
                idx = np.flatnonzero(labeled == label)
                if idx.size:
                    centers.append((abs(float(np.mean(idx)) - valley_index), int(idx[0]), int(idx[-1]) + 1))
            _, core_start, core_stop = min(centers, key=lambda item: item[0])
        else:
            component = largest_true_component(stable, min_core_bands)
            if component is None:
                finite_component = largest_true_component(finite, min_core_bands)
                core_start, core_stop = finite_component if finite_component else (0, n)
            else:
                core_start, core_stop = component
    else:
        core_start, core_stop = component

    # Afinar bordes: no dejar que el nucleo quede demasiado corto.
    if core_stop - core_start < min_core_bands:
        half = min_core_bands // 2
        core_start = max(search_start, valley_index - half)
        core_stop = min(search_stop, valley_index + half)

    if target_bands > 0:
        center = int(round((core_start + core_stop) / 2))
        start = center - target_bands // 2
        start = max(search_start, min(start, search_stop - target_bands))
        stop = start + target_bands
    else:
        start = max(search_start, core_start - margin_bands)
        stop = min(search_stop, core_stop + margin_bands)
    if stop <= start:
        start, stop = search_start, search_stop

    diagnostics = {
        "smooth": smooth,
        "roughness": roughness,
        "derivative": derivative,
        "good": valley_mask.astype(np.float32),
        "stable": stable.astype(np.float32),
        "rough_thr": float(rough_thr),
        "deriv_thr": float(deriv_thr),
        "valley_index": float(valley_index),
        "valley_level": float(valley_level),
        "search_start": float(search_start),
        "search_stop": float(search_stop),
        "target_bands": float(target_bands),
    }
    return start, stop, core_start, core_stop, diagnostics


def save_plot(
    path: Path,
    cube_id: str,
    y: np.ndarray,
    start: int,
    stop: int,
    core_start: int,
    core_stop: int,
    diagnostics: dict[str, np.ndarray | float],
) -> None:
    x = np.arange(y.size)
    smooth = np.asarray(diagnostics["smooth"])
    roughness = np.asarray(diagnostics["roughness"])
    good = np.asarray(diagnostics["good"])
    search_start = int(diagnostics["search_start"])
    search_stop = int(diagnostics["search_stop"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(x, y, color="#8b5a2b", linewidth=1.0, label="Firma completa")
    axes[0].plot(x, smooth, color="black", linewidth=1.0, alpha=0.8, label="Suavizada")
    axes[0].axvspan(start, stop - 1, color="#3b82f6", alpha=0.18, label="Recorte con margen")
    axes[0].axvspan(core_start, core_stop - 1, color="#22c55e", alpha=0.16, label="Nucleo estable")
    axes[0].axvline(search_start, color="gray", linestyle="--", linewidth=0.8)
    axes[0].axvline(search_stop, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Reflectancia")
    axes[0].set_title(f"{cube_id} | bandas {start}:{stop} ({stop - start} bandas)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(x, roughness, color="#7c3aed", linewidth=1.0)
    axes[1].axvspan(start, stop - 1, color="#3b82f6", alpha=0.18)
    axes[1].axvline(search_start, color="gray", linestyle="--", linewidth=0.8)
    axes[1].axvline(search_stop, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Rugosidad")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, good, color="#059669", linewidth=1.0)
    axes[2].axvspan(start, stop - 1, color="#3b82f6", alpha=0.18)
    axes[2].axvline(search_start, color="gray", linestyle="--", linewidth=0.8)
    axes[2].axvline(search_stop, color="gray", linestyle="--", linewidth=0.8)
    axes[2].set_xlabel("Indice de banda")
    axes[2].set_ylabel("Mascara estable")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def process_signature(
    npz_path: Path,
    output_dir: Path,
    margin_bands: int,
    min_core_bands: int,
    search_start: int,
    search_stop: int,
    target_bands: int,
    signature_key: str,
) -> CropResult:
    cube_id = npz_path.parent.name
    with np.load(npz_path) as data:
        y = data[signature_key].astype(np.float32).reshape(-1)

    start, stop, core_start, core_stop, diagnostics = select_usable_range(
        y,
        margin_bands=margin_bands,
        min_core_bands=min_core_bands,
        search_start=search_start,
        search_stop=search_stop,
        target_bands=target_bands,
    )
    crop = y[start:stop]

    finite_fraction = float(np.mean(np.isfinite(crop))) if crop.size else 0.0
    roughness_crop = float(np.nanmedian(np.asarray(diagnostics["roughness"])[start:stop]))
    status = "ok" if crop.size >= min_core_bands and finite_fraction > 0.90 else "review"
    reason = "" if status == "ok" else "recorte_corto_o_con_nan"

    npy_dir = output_dir / "NPY"
    csv_dir = output_dir / "CSV"
    plot_dir = output_dir / "PLOTS"
    for directory in (npy_dir, csv_dir, plot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    np.save(npy_dir / f"{cube_id}_recorte.npy", crop)
    np.save(npy_dir / f"{cube_id}_indices.npy", np.array([start, stop], dtype=np.int32))

    csv_path = csv_dir / f"{cube_id}_recorte.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["band_index", signature_key])
        for index, value in zip(range(start, stop), crop):
            writer.writerow([index, float(value) if np.isfinite(value) else "nan"])

    save_plot(
        plot_dir / f"{cube_id}_seleccion_tramo.png",
        cube_id,
        y,
        start,
        stop,
        core_start,
        core_stop,
        diagnostics,
    )

    return CropResult(
        cube_id=cube_id,
        source_npz=str(npz_path),
        total_bands=int(y.size),
        start=int(start),
        stop=int(stop),
        usable_bands=int(stop - start),
        margin_bands=int(margin_bands),
        search_start=int(search_start),
        search_stop=int(search_stop),
        target_bands=int(target_bands),
        core_start=int(core_start),
        core_stop=int(core_stop),
        finite_fraction_crop=finite_fraction,
        roughness_crop=roughness_crop,
        status=status,
        reason=reason,
    )


def write_summary(path: Path, results: list[CropResult]) -> None:
    if not results:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Selecciona tramos usables en firmas de reflectancia."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Firmas_automaticas"),
        help="Carpeta con cubos/*/resultado.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Carpeta de salida. Por defecto: <input-dir>/recortes_firmas.",
    )
    parser.add_argument(
        "--cubos-subdir",
        default="cubos",
        help="Subcarpeta dentro de input-dir que contiene */resultado.npz. Ej: cubos o cubos_manuales.",
    )
    parser.add_argument("--signature-key", default="soil_reflectance")
    parser.add_argument("--margin-bands", type=int, default=35)
    parser.add_argument("--min-core-bands", type=int, default=120)
    parser.add_argument("--search-start", type=int, default=100)
    parser.add_argument("--search-stop", type=int, default=900)
    parser.add_argument("--target-bands", type=int, default=300)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_dir / "recortes_firmas"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cubos_dir = input_dir / args.cubos_subdir
    npz_files = sorted(cubos_dir.glob("*/resultado.npz"))
    if not npz_files:
        print(f"No encontre resultado.npz en {cubos_dir}")
        return 2

    results: list[CropResult] = []
    for npz_path in npz_files:
        try:
            result = process_signature(
                npz_path,
                output_dir,
                args.margin_bands,
                args.min_core_bands,
                args.search_start,
                args.search_stop,
                args.target_bands,
                args.signature_key,
            )
            results.append(result)
            print(
                f"{result.cube_id}: {result.start}:{result.stop} "
                f"({result.usable_bands} bandas) {result.status}"
            )
        except Exception as exc:
            print(f"ERROR {npz_path}: {exc}")

    write_summary(output_dir / "resumen_recortes.csv", results)
    print(f"Salida: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
