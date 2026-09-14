"""Prueba: cubo Soil_1 con recorte de 300 bandas y bandas separables.

Flujo:
1. Cubo crudo -> recorte espectral individual de 300 bandas.
2. Normalizacion por firmas dark/white para obtener reflectancia por pixel.
3. Mapeo de esas 300 bandas a 84, 49, 35 y 27 bandas separables.
4. Mascara Soil -> recorte espacial fijo de 180 x 180 pixeles.

Los archivos fuente se leen solamente; los resultados se guardan junto a
este script dentro de ``resultados/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


# El ejecutor por lote asigna este identificador antes de llamar a main().
SOIL_ID = "Soil_1__cube_20260617_171738"
DATA_ROOT: Path | None = None
RAW_ROOT: Path | None = None
OUTPUT_ROOT: Path | None = None
# Soil_1 ocupa aproximadamente 180 px en cada eje. Este tamaño disminuye
# notablemente el fondo exterior manteniendo un parche cuadrado.
SPATIAL_SIZE = 180


def fixed_square_bounds(mask: np.ndarray, size: int) -> tuple[int, int, int, int]:
    """Return fixed-size bounds centered on the Soil mask bounding box."""
    y, x = np.where(mask)
    if y.size == 0:
        raise ValueError("La mascara Soil no contiene pixeles.")

    center_y = int(round((int(y.min()) + int(y.max())) / 2))
    center_x = int(round((int(x.min()) + int(x.max())) / 2))
    half = size // 2
    return center_y - half, center_y + half, center_x - half, center_x + half


def select_most_recent_result(spectral_root: Path, soil_id: str) -> Path:
    """Choose the most recent valid selection between automatic and manual sources."""
    candidates = [
        spectral_root / "Firmas_automaticas" / "cubos" / soil_id / "resultado.npz",
        spectral_root / "Firmas_automaticas" / "cubos_manuales" / soil_id / "resultado.npz",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise FileNotFoundError(f"No hay resultado.npz para {soil_id} en cubos ni cubos_manuales.")
    return max(existing, key=lambda path: path.stat().st_mtime)


def raw_cube_path(data_root: Path, sample_folder: str, raw_stem: str) -> Path:
    """Support both project layouts: Capturas_soil/Soil_X and root/Soil_X."""
    filename = f"{raw_stem}.npy"
    nested = data_root / "Capturas_soil" / sample_folder / filename
    direct = data_root / sample_folder / filename
    if nested.exists() or not direct.exists():
        return nested
    return direct


def crop_with_padding(
    cube: np.ndarray, mask: np.ndarray, bounds: tuple[int, int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a fixed square; pad outside-image locations with NaN / False."""
    y0, y1, x0, x1 = bounds
    height, width = mask.shape
    size_y, size_x = y1 - y0, x1 - x0

    out_cube = np.full((size_y, size_x, cube.shape[2]), np.nan, dtype=np.float32)
    out_mask = np.zeros((size_y, size_x), dtype=bool)

    src_y0, src_y1 = max(y0, 0), min(y1, height)
    src_x0, src_x1 = max(x0, 0), min(x1, width)
    dst_y0, dst_y1 = src_y0 - y0, src_y1 - y0
    dst_x0, dst_x1 = src_x0 - x0, src_x1 - x0

    out_cube[dst_y0:dst_y1, dst_x0:dst_x1, :] = cube[src_y0:src_y1, src_x0:src_x1, :]
    out_mask[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return out_cube, out_mask


def interpolate_spectral_axis(cube: np.ndarray, positions_1based: np.ndarray) -> np.ndarray:
    """Linearly interpolate a (Y, X, bands) cube at 1-based positions."""
    positions = np.asarray(positions_1based, dtype=np.float64).reshape(-1) - 1.0
    if positions.min() < 0 or positions.max() > cube.shape[2] - 1:
        raise ValueError("Las posiciones espectrales quedan fuera del cubo de 300 bandas.")

    low = np.floor(positions).astype(int)
    high = np.ceil(positions).astype(int)
    alpha = (positions - low).astype(np.float32)

    low_values = cube[:, :, low]
    high_values = cube[:, :, high]
    return (low_values * (1.0 - alpha) + high_values * alpha).astype(np.float32)


def reflectance_cube(cube_raw: np.ndarray, white: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """Apply the same reflectance formula used for the 1D Soil signatures."""
    denominator = white - dark
    finite_abs = np.abs(denominator[np.isfinite(denominator)])
    scale = float(np.percentile(finite_abs, 5)) if finite_abs.size else 0.0
    epsilon = max(scale * 0.05, np.finfo(np.float32).eps)
    valid_bands = np.isfinite(denominator) & (np.abs(denominator) > epsilon)

    output = np.full(cube_raw.shape, np.nan, dtype=np.float32)
    output[:, :, valid_bands] = (
        (cube_raw[:, :, valid_bands] - dark[valid_bands])
        / denominator[valid_bands]
    )
    return output


def save_diagnostic_image(cube: np.ndarray, mask: np.ndarray, wavelength: float, path: Path) -> None:
    band_index = cube.shape[2] // 2
    image = cube[:, :, band_index].copy()
    image[~mask] = np.nan

    fig, axis = plt.subplots(figsize=(6, 6))
    shown = axis.imshow(image, cmap="viridis")
    axis.set_title(f"{SOIL_ID} | banda {band_index + 1} | {wavelength:.1f} nm")
    axis.axis("off")
    fig.colorbar(shown, ax=axis, label="Reflectancia")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def reconstruct_full_band(
    raw: np.ndarray,
    spectral_start: int,
    position_1based: float,
    white: np.ndarray,
    dark: np.ndarray,
) -> np.ndarray:
    """Reconstruct one full spatial band using the same spectral interpolation."""
    position_zero_based = float(position_1based) - 1.0
    low = int(np.floor(position_zero_based))
    high = int(np.ceil(position_zero_based))
    alpha = np.float32(position_zero_based - low)

    denominator = white - dark
    finite_abs = np.abs(denominator[np.isfinite(denominator)])
    scale = float(np.percentile(finite_abs, 5)) if finite_abs.size else 0.0
    epsilon = max(scale * 0.05, np.finfo(np.float32).eps)

    def normalized_plane(index: int) -> np.ndarray:
        if not np.isfinite(denominator[index]) or abs(denominator[index]) <= epsilon:
            return np.full((raw.shape[0], raw.shape[2]), np.nan, dtype=np.float32)
        plane = raw[:, spectral_start + index, :].astype(np.float32)
        return (plane - dark[index]) / denominator[index]

    low_plane = normalized_plane(low)
    high_plane = normalized_plane(high)
    return (low_plane * (1.0 - alpha) + high_plane * alpha).astype(np.float32)


def save_full_and_cropped_band(
    full_band: np.ndarray,
    cropped_cube: np.ndarray,
    crop_bounds: tuple[int, int, int, int],
    band_index: int,
    wavelength: float,
    path: Path,
) -> None:
    """Save a gray-scale full-band/crop comparison with dashed crop outline."""
    y0, y1, x0, x1 = crop_bounds
    # El producto visual principal es el recorte cuadrado. La máscara Soil se
    # conserva en un archivo independiente para trazabilidad y análisis futuro.
    cropped_band = cropped_cube[:, :, band_index]

    finite = full_band[np.isfinite(full_band)]
    vmin, vmax = np.percentile(finite, [1, 99])
    if vmax <= vmin:
        vmax = vmin + 1e-6

    cmap = plt.colormaps["gray"].copy()
    cmap.set_bad("white")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    full_image = axes[0].imshow(full_band, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="red",
            linewidth=2.0,
            linestyle="--",
        )
    )
    axes[0].set_title(
        "Banda reconstruida completa\ncuadro punteado: recorte espacial",
        fontsize=11,
    )
    axes[0].set_xlabel("Eje espacial de escaneo (px)")
    axes[0].set_ylabel("Eje espacial vertical (px)")

    axes[1].imshow(cropped_band, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(
        "Banda recortada espacialmente\nparche cuadrado de Soil",
        fontsize=11,
    )
    axes[1].set_xlabel("Eje espacial de escaneo (px)")
    axes[1].set_ylabel("Eje espacial vertical (px)")

    fig.colorbar(full_image, ax=axes[1], shrink=0.80, pad=0.03, label="Reflectancia")
    fig.suptitle(
        f"{SOIL_ID} | banda {band_index + 1} | {wavelength:.1f} nm",
        fontsize=14,
    )
    fig.subplots_adjust(top=0.82, wspace=0.24)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    spectral_root = DATA_ROOT if DATA_ROOT is not None else script_dir.parent
    output_base = OUTPUT_ROOT if OUTPUT_ROOT is not None else script_dir / "resultados"
    results_dir = output_base / SOIL_ID
    results_dir.mkdir(parents=True, exist_ok=True)

    sample_folder, raw_stem = SOIL_ID.split("__", maxsplit=1)
    raw_base = RAW_ROOT if RAW_ROOT is not None else spectral_root
    raw_path = raw_cube_path(raw_base, sample_folder, raw_stem)
    result_npz_path = select_most_recent_result(spectral_root, SOIL_ID)
    indices_path = spectral_root / "Firmas_automaticas" / "recortes_firmas_final" / "NPY" / f"{SOIL_ID}_indices.npy"
    signatures_root = spectral_root / "Firmas_automaticas" / "Firmas_bandas_separadas"

    for path in (raw_path, result_npz_path, indices_path, signatures_root):
        if not path.exists():
            raise FileNotFoundError(f"No existe: {path}")

    # El recorte individual final se guarda como [inicio, fin), por ejemplo [255, 555].
    spectral_limits = np.load(indices_path).astype(int).reshape(-1)
    if spectral_limits.size != 2 or spectral_limits[1] <= spectral_limits[0]:
        raise ValueError(f"Indices de recorte invalidos: {spectral_limits}")
    spectral_start, spectral_stop = map(int, spectral_limits)
    n_crop_bands = spectral_stop - spectral_start
    if n_crop_bands != 300:
        raise ValueError(f"Se esperaban 300 bandas recortadas; llegaron {n_crop_bands}.")

    with np.load(result_npz_path, allow_pickle=False) as signatures:
        soil_mask = signatures["soil_mask"].astype(bool)
        white_full = signatures["white_signature"].astype(np.float32)
        dark_full = signatures["dark_signature"].astype(np.float32)

    if soil_mask.ndim != 2:
        raise ValueError(f"La mascara debe ser 2D; llego {soil_mask.shape}.")

    bounds = fixed_square_bounds(soil_mask, SPATIAL_SIZE)
    y0, y1, x0, x1 = bounds
    raw = np.load(raw_path, mmap_mode="r")
    if raw.shape[0] != soil_mask.shape[0] or raw.shape[2] != soil_mask.shape[1]:
        raise ValueError(
            f"Mascara {soil_mask.shape} no coincide con ejes espaciales del crudo {raw.shape}."
        )
    if spectral_stop > raw.shape[1]:
        raise ValueError("El recorte espectral queda fuera del cubo crudo.")

    # La extraccion espacial se hace antes de materializar el arreglo grande,
    # pero es equivalente a recortar el cubo de bandas ya reconstruido.
    src_y0, src_y1 = max(y0, 0), min(y1, raw.shape[0])
    src_x0, src_x1 = max(x0, 0), min(x1, raw.shape[2])
    cube_crop_raw = raw[src_y0:src_y1, spectral_start:spectral_stop, src_x0:src_x1]
    cube_crop_raw = np.transpose(cube_crop_raw, (0, 2, 1)).astype(np.float32)

    white = white_full[spectral_start:spectral_stop]
    dark = dark_full[spectral_start:spectral_stop]
    cube_300_valid = reflectance_cube(cube_crop_raw, white, dark)

    # Inserta padding, si hiciera falta, para asegurar una salida espacial fija.
    cube_300 = np.full((SPATIAL_SIZE, SPATIAL_SIZE, n_crop_bands), np.nan, dtype=np.float32)
    mask_square = np.zeros((SPATIAL_SIZE, SPATIAL_SIZE), dtype=bool)
    dst_y0, dst_y1 = src_y0 - y0, src_y1 - y0
    dst_x0, dst_x1 = src_x0 - x0, src_x1 - x0
    cube_300[dst_y0:dst_y1, dst_x0:dst_x1, :] = cube_300_valid
    mask_square[dst_y0:dst_y1, dst_x0:dst_x1] = soil_mask[src_y0:src_y1, src_x0:src_x1]

    criteria = [
        (2, 84),
        (4, 49),
        (6, 35),
        (8, 27),
    ]
    validation_rows: list[dict[str, float | int | str]] = []

    for pixel_step, expected_bands in criteria:
        prefix = f"{sample_folder}__{expected_bands}_Bandas"
        signature_path = (
            signatures_root
            / f"{pixel_step}px_{expected_bands}_bandas"
            / "CSV"
            / f"{prefix}.csv"
        )
        signature_csv = pd.read_csv(signature_path)
        required_signature_columns = {
            "source_row",
            "wavelength_nm",
            "posicion_equivalente_soil",
            "soil_reflectance_interpolada",
        }
        missing_columns = required_signature_columns.difference(signature_csv.columns)
        if missing_columns:
            raise ValueError(f"Faltan columnas en {signature_path}: {sorted(missing_columns)}")
        if len(signature_csv) != expected_bands:
            raise ValueError(f"{signature_path.name} no contiene {expected_bands} bandas.")

        # Esta columna es el mapeo exacto usado previamente para generar la firma.
        positions = signature_csv["posicion_equivalente_soil"].to_numpy(dtype=np.float64)
        wavelengths = signature_csv["wavelength_nm"].to_numpy(dtype=np.float64)
        # Este es el cubo final principal: un parche cuadrado de tamaño fijo.
        # No se enmascaran las esquinas, porque el usuario necesita conservar
        # una imagen cuadrada. La máscara original queda guardada por separado.
        cube_final = interpolate_spectral_axis(cube_300, positions)

        output_dir = results_dir / f"{pixel_step}px_{expected_bands}_bandas"
        output_dir.mkdir(exist_ok=True)
        np.save(output_dir / f"{prefix}_cubo_recorte_cuadrado_reflectancia.npy", cube_final)
        np.save(output_dir / f"{prefix}_mascara_soil.npy", mask_square)
        np.save(output_dir / f"{prefix}_wavelengths_nm.npy", wavelengths.reshape(-1, 1))

        # La firma original se obtuvo como mediana espacial y luego interpolacion.
        # El cubo hace interpolacion por pixel y luego mediana espacial; ambas
        # operaciones no son identicas al usar mediana, por lo que se documenta
        # la diferencia en vez de forzar el cubo a coincidir artificialmente.
        signature_from_cube = np.nanmedian(cube_final[mask_square, :], axis=0)
        signature_reference = signature_csv["soil_reflectance_interpolada"].to_numpy(dtype=np.float32)
        difference = signature_from_cube - signature_reference
        max_abs_difference = float(np.nanmax(np.abs(difference)))
        mean_abs_difference = float(np.nanmean(np.abs(difference)))
        rmse_difference = float(np.sqrt(np.nanmean(difference**2)))

        comparison = pd.DataFrame(
            {
                "wavelength_nm": wavelengths,
                "posicion_equivalente_soil": positions,
                "firma_reflectancia_referencia": signature_reference,
                "mediana_reflectancia_cubo": signature_from_cube,
                "diferencia_cubo_menos_referencia": difference,
            }
        )
        comparison.to_csv(
            output_dir / f"{prefix}_comparacion_firma_vs_cubo.csv",
            index=False,
            encoding="utf-8-sig",
        )

        diagnostic_band = cube_final.shape[2] // 2
        save_diagnostic_image(
            cube_final,
            mask_square,
            wavelengths[diagnostic_band],
            output_dir / f"{prefix}_banda_media.png",
        )

        # Para el criterio de 27 bandas se guarda la comparacion solicitada:
        # banda completa con zona punteada y esa misma banda ya recortada.
        if expected_bands == 27:
            full_band = reconstruct_full_band(
                raw,
                spectral_start,
                positions[diagnostic_band],
                white,
                dark,
            )
            save_full_and_cropped_band(
                full_band,
                cube_final,
                bounds,
                diagnostic_band,
                wavelengths[diagnostic_band],
                output_dir / f"{prefix}_banda_completa_y_recortada.png",
            )

        metadata = {
            "soil_id": SOIL_ID,
            "input_raw_cube": str(raw_path),
            "selection_result_npz": str(result_npz_path),
            "spectral_crop_indices_start_stop": [spectral_start, spectral_stop],
            "spectral_crop_bands": n_crop_bands,
            "pixel_step": pixel_step,
            "separable_bands": expected_bands,
            "spatial_size": SPATIAL_SIZE,
            "spatial_bounds_y0_y1_x0_x1": [y0, y1, x0, x1],
            "soil_mask_pixels": int(mask_square.sum()),
            "signature_reference_csv": str(signature_path),
            "max_abs_difference_vs_signature": max_abs_difference,
            "mean_abs_difference_vs_signature": mean_abs_difference,
            "rmse_vs_signature": rmse_difference,
            "comparison_note": (
                "La firma referencia aplica mediana y luego interpolacion; "
                "el cubo aplica interpolacion por pixel y luego mediana."
            ),
            "cube_dtype": "float32",
            "output_spatial_content": (
                "Parche cuadrado sin enmascarar; la máscara Soil original se "
                "guarda separada para identificar los píxeles usados en la firma."
            ),
        }
        (output_dir / f"{prefix}_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        validation_rows.append(
            {
                "pixel_step": pixel_step,
                "n_bands": expected_bands,
                "cube_shape": "x".join(map(str, cube_final.shape)),
                "soil_pixels": int(mask_square.sum()),
                "max_abs_difference_vs_signature": max_abs_difference,
                "mean_abs_difference_vs_signature": mean_abs_difference,
                "rmse_vs_signature": rmse_difference,
                "status": "ok",
            }
        )
        print(
            f"{expected_bands} bandas: {cube_final.shape}; "
            f"diferencia maxima vs firma={max_abs_difference:.8g}"
        )

    pd.DataFrame(validation_rows).to_csv(
        results_dir / f"resumen_validacion_{SOIL_ID}.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Resultados: {results_dir}")


if __name__ == "__main__":
    main()
