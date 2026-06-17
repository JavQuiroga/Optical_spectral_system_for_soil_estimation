from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_bool_array(values: np.ndarray) -> np.ndarray:
    if values.dtype == bool:
        return values
    text = np.char.lower(values.astype(str))
    return np.isin(text, ["true", "1", "yes", "y", "si", "sí"])


def load_calibration(
    calibration_csv: Path | None = None,
    calibration_npz: Path | None = None,
) -> dict[str, np.ndarray]:
    """Carga puntos wavelength_nm <-> x_pixel desde CSV o NPZ ya calculado."""
    if calibration_csv is None and calibration_npz is None:
        raise ValueError("Debes proporcionar --calibration-csv o --calibration-npz.")

    if calibration_npz is not None:
        if not calibration_npz.exists():
            raise FileNotFoundError(f"No existe calibracion NPZ: {calibration_npz}")
        data = np.load(calibration_npz, allow_pickle=True)
        names = set(data.files)

        if {"wavelengths_nm", "x_pixels"}.issubset(names):
            wavelengths = data["wavelengths_nm"]
            x_pixels = data["x_pixels"]
        elif {"wavelength_nm", "selected_x"}.issubset(names):
            wavelengths = data["wavelength_nm"]
            x_pixels = data["selected_x"]
        else:
            raise ValueError(
                "El NPZ debe contener wavelengths_nm/x_pixels "
                "o wavelength_nm/selected_x."
            )

        result = {
            "wavelength_nm": np.asarray(wavelengths, dtype=np.float64).reshape(-1),
            "selected_x": np.asarray(x_pixels, dtype=np.float64).reshape(-1),
        }
        for optional in [
            "fit_inlier",
            "residual_px",
            "block",
            "peak_width_px",
            "peak_persistence",
            "adjusted_peak_score",
        ]:
            if optional in names:
                result[optional] = np.asarray(data[optional]).reshape(-1)
        print(f"Calibracion cargada desde NPZ: {calibration_npz}")
        return result

    assert calibration_csv is not None
    if not calibration_csv.exists():
        raise FileNotFoundError(f"No existe calibracion CSV: {calibration_csv}")

    table = np.genfromtxt(
        calibration_csv,
        delimiter=",",
        names=True,
        dtype=None,
        encoding=None,
        autostrip=True,
    )
    if table.ndim == 0:
        table = np.array([table], dtype=table.dtype)

    names = set(table.dtype.names or [])
    required = {"wavelength_nm", "selected_x"}
    missing = required - names
    if missing:
        raise ValueError(f"Al CSV le faltan columnas requeridas: {sorted(missing)}")

    result = {
        "wavelength_nm": table["wavelength_nm"].astype(np.float64).reshape(-1),
        "selected_x": table["selected_x"].astype(np.float64).reshape(-1),
    }
    for optional in [
        "fit_inlier",
        "residual_px",
        "block",
        "peak_width_px",
        "peak_persistence",
        "adjusted_peak_score",
    ]:
        if optional in names:
            result[optional] = np.asarray(table[optional]).reshape(-1)

    print(f"Calibracion cargada desde CSV: {calibration_csv}")
    return result


def _aggregate_duplicate_wavelengths(
    wavelengths_nm: np.ndarray,
    x_pixels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unique_wavelengths = np.unique(wavelengths_nm)
    if unique_wavelengths.size == wavelengths_nm.size:
        return wavelengths_nm, x_pixels

    print(
        "Advertencia: hay longitudes de onda duplicadas; "
        "se consolidan usando la mediana de selected_x."
    )
    agg_x = np.empty(unique_wavelengths.size, dtype=np.float64)
    for i, wavelength in enumerate(unique_wavelengths):
        agg_x[i] = np.median(x_pixels[wavelengths_nm == wavelength])
    return unique_wavelengths, agg_x


def _aggregate_duplicate_x_pixels(
    wavelengths_nm: np.ndarray,
    x_pixels: np.ndarray,
    max_wavelength_span_per_pixel_nm: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Consolida pixeles repetidos y elimina colapsos espectrales grandes."""
    rounded_x = np.rint(x_pixels).astype(np.int64)
    unique_x = np.unique(rounded_x)

    if unique_x.size == rounded_x.size:
        return wavelengths_nm, x_pixels

    keep_wavelengths = []
    keep_x = []
    removed_groups = 0
    large_spans = []

    for x in unique_x:
        mask = rounded_x == x
        group_wavelengths = wavelengths_nm[mask]
        span = float(np.max(group_wavelengths) - np.min(group_wavelengths))
        if (
            max_wavelength_span_per_pixel_nm is not None
            and span > max_wavelength_span_per_pixel_nm
        ):
            removed_groups += 1
            large_spans.append((int(x), float(np.min(group_wavelengths)), float(np.max(group_wavelengths))))
            continue
        keep_wavelengths.append(float(np.median(group_wavelengths)))
        keep_x.append(float(np.median(x_pixels[mask])))

    if removed_groups:
        print(
            "Advertencia: se eliminaron "
            f"{removed_groups} pixeles con colapso espectral mayor a "
            f"{max_wavelength_span_per_pixel_nm:g} nm."
        )
        for x, w_min, w_max in large_spans[:8]:
            print(f"  x={x}: {w_min:.1f}-{w_max:.1f} nm")
        if len(large_spans) > 8:
            print(f"  ... y {len(large_spans) - 8} grupos mas.")

    if len(keep_wavelengths) < 2:
        raise ValueError("Quedaron menos de 2 puntos despues de consolidar selected_x duplicados.")

    return np.asarray(keep_wavelengths), np.asarray(keep_x)


def clean_calibration_points(
    calibration: dict[str, np.ndarray],
    max_abs_residual_px: float | None = None,
    min_points: int = 8,
    max_x_jump_px: float = 30.0,
    max_wavelength_span_per_pixel_nm: float | None = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Filtra, ordena y consolida puntos confiables de calibracion."""
    wavelengths_nm = np.asarray(calibration["wavelength_nm"], dtype=np.float64).reshape(-1)
    x_pixels = np.asarray(calibration["selected_x"], dtype=np.float64).reshape(-1)

    if wavelengths_nm.size != x_pixels.size:
        raise ValueError("wavelength_nm y selected_x no tienen la misma longitud.")

    valid = np.isfinite(wavelengths_nm) & np.isfinite(x_pixels)

    if "fit_inlier" in calibration:
        inliers = parse_bool_array(np.asarray(calibration["fit_inlier"]).reshape(-1))
        if inliers.size == valid.size:
            valid &= inliers
            print(f"Puntos fit_inlier=True: {int(inliers.sum())}/{inliers.size}")
        else:
            print("Advertencia: fit_inlier existe pero su tamano no coincide; se ignora.")

    if "residual_px" in calibration:
        residual = np.asarray(calibration["residual_px"], dtype=np.float64).reshape(-1)
        if residual.size == valid.size:
            finite_residual = np.isfinite(residual)
            if np.any(finite_residual):
                p95 = float(np.percentile(np.abs(residual[finite_residual]), 95))
                print(f"Residual abs p95: {p95:.3f} px")
            if max_abs_residual_px is not None:
                before = int(valid.sum())
                valid &= finite_residual & (np.abs(residual) <= max_abs_residual_px)
                removed = before - int(valid.sum())
                if removed:
                    print(
                        "Puntos descartados por residual alto "
                        f"(>{max_abs_residual_px:g} px): {removed}"
                    )
        else:
            print("Advertencia: residual_px existe pero su tamano no coincide; se ignora.")

    wavelengths_nm = wavelengths_nm[valid]
    x_pixels = x_pixels[valid]

    if wavelengths_nm.size < min_points:
        print(
            "Advertencia: pocos puntos validos de calibracion: "
            f"{wavelengths_nm.size}. Minimo sugerido: {min_points}."
        )
    if wavelengths_nm.size < 2:
        raise ValueError("No hay suficientes puntos validos de calibracion.")

    order = np.argsort(wavelengths_nm)
    wavelengths_nm = wavelengths_nm[order]
    x_pixels = x_pixels[order]

    wavelengths_nm, x_pixels = _aggregate_duplicate_wavelengths(wavelengths_nm, x_pixels)
    wavelengths_nm, x_pixels = _aggregate_duplicate_x_pixels(
        wavelengths_nm,
        x_pixels,
        max_wavelength_span_per_pixel_nm=max_wavelength_span_per_pixel_nm,
    )

    order = np.argsort(wavelengths_nm)
    wavelengths_nm = wavelengths_nm[order]
    x_pixels = x_pixels[order]

    dx = np.diff(x_pixels)
    if np.any(np.abs(dx) > max_x_jump_px):
        print(
            "Advertencia: se detectaron saltos grandes en selected_x "
            f"(umbral {max_x_jump_px:g} px)."
        )
        jump_idx = np.where(np.abs(dx) > max_x_jump_px)[0]
        for i in jump_idx[:10]:
            print(
                f"  {wavelengths_nm[i]:.1f}->{wavelengths_nm[i + 1]:.1f} nm: "
                f"x {x_pixels[i]:.2f}->{x_pixels[i + 1]:.2f}"
            )
        if jump_idx.size > 10:
            print(f"  ... y {jump_idx.size - 10} saltos mas.")

    if np.any(np.diff(x_pixels) < 0):
        print(
            "Advertencia: selected_x no es monotono creciente con wavelength_nm. "
            "np.interp puede suavizar zonas ambiguas, pero revisa la calibracion."
        )

    print(
        "Calibracion limpia: "
        f"{wavelengths_nm.size} puntos, "
        f"{wavelengths_nm.min():.1f}-{wavelengths_nm.max():.1f} nm, "
        f"x {x_pixels.min():.2f}-{x_pixels.max():.2f} px"
    )
    return wavelengths_nm, x_pixels


def make_output_wavelength_axis(start_nm: float, end_nm: float, step_nm: float) -> np.ndarray:
    if step_nm <= 0:
        raise ValueError("--step-nm debe ser mayor que cero.")
    if end_nm < start_nm:
        raise ValueError("--end-nm debe ser mayor o igual a --start-nm.")
    n = int(np.floor((end_nm - start_nm) / step_nm)) + 1
    return (start_nm + np.arange(n, dtype=np.float64) * step_nm).astype(np.float32)


def reconstruct_spectral_cube(
    data: np.ndarray,
    calibration_wavelengths_nm: np.ndarray,
    calibration_x_pixels: np.ndarray,
    output_wavelengths_nm: np.ndarray,
    spectral_axis: int = -1,
    out_of_range: str = "nan",
    dtype: str = "float32",
) -> np.ndarray:
    """Interpola el eje espacial de dispersion hacia un eje wavelength_nm."""
    if data.ndim not in (2, 3):
        raise ValueError(f"Solo se soportan arreglos 2D o 3D; llego shape={data.shape}")

    spectral_axis = int(spectral_axis)
    if spectral_axis < 0:
        spectral_axis += data.ndim
    if spectral_axis < 0 or spectral_axis >= data.ndim:
        raise ValueError(f"--spectral-axis invalido para shape {data.shape}")

    moved = np.moveaxis(data, spectral_axis, -1)
    spectral_width = moved.shape[-1]

    if np.any((calibration_x_pixels < 0) | (calibration_x_pixels > spectral_width - 1)):
        bad = np.where((calibration_x_pixels < 0) | (calibration_x_pixels > spectral_width - 1))[0]
        print(
            "Advertencia: hay posiciones x de calibracion fuera del ancho espectral "
            f"del dato ({spectral_width}). Se eliminaran {bad.size} puntos."
        )
        inside = (calibration_x_pixels >= 0) & (calibration_x_pixels <= spectral_width - 1)
        calibration_wavelengths_nm = calibration_wavelengths_nm[inside]
        calibration_x_pixels = calibration_x_pixels[inside]
        if calibration_wavelengths_nm.size < 2:
            raise ValueError("No quedan suficientes puntos dentro del ancho del cubo.")

    cal_min = float(np.min(calibration_wavelengths_nm))
    cal_max = float(np.max(calibration_wavelengths_nm))
    req_min = float(np.min(output_wavelengths_nm))
    req_max = float(np.max(output_wavelengths_nm))
    outside = (output_wavelengths_nm < cal_min) | (output_wavelengths_nm > cal_max)

    if np.any(outside):
        print(
            "Advertencia: el rango solicitado sale del rango calibrado "
            f"({cal_min:.1f}-{cal_max:.1f} nm)."
        )
        if out_of_range == "clip":
            output_for_interp = np.clip(output_wavelengths_nm, cal_min, cal_max)
            print("Modo fuera de rango: clip.")
        elif out_of_range == "nan":
            output_for_interp = output_wavelengths_nm.astype(np.float64)
            print("Modo fuera de rango: NaN.")
        else:
            raise ValueError("--out-of-range debe ser 'nan' o 'clip'.")
    else:
        output_for_interp = output_wavelengths_nm.astype(np.float64)

    order = np.argsort(calibration_wavelengths_nm)
    cal_w = calibration_wavelengths_nm[order].astype(np.float64)
    cal_x = calibration_x_pixels[order].astype(np.float64)

    target_x = np.interp(output_for_interp, cal_w, cal_x)
    target_x = np.clip(target_x, 0, spectral_width - 1)

    # Interpolacion lineal vectorizada sobre el eje espectral.
    # Evita recorrer millones de perfiles pixel a pixel en Python.
    x0 = np.floor(target_x).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, spectral_width - 1)
    weight = (target_x - x0).astype(np.float32)

    moved_f32 = moved.astype(np.float32, copy=False)
    left = np.take(moved_f32, x0, axis=-1)
    right = np.take(moved_f32, x1, axis=-1)
    result = left * (1.0 - weight) + right * weight
    if out_of_range == "nan" and np.any(outside):
        result[..., outside] = np.nan

    result = np.moveaxis(result, -1, spectral_axis)
    if dtype != "float32":
        result = result.astype(dtype)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruye un cubo espectral interpolando el eje de dispersion x "
            "hacia longitudes de onda usando una calibracion ya calculada."
        )
    )
    parser.add_argument("--input-npy", required=True, type=Path, help="Imagen/cubo .npy de entrada.")
    parser.add_argument("--output-npy", required=True, type=Path, help="Cubo reconstruido .npy de salida.")
    parser.add_argument("--calibration-csv", type=Path, help="CSV spectral_characterization.csv.")
    parser.add_argument("--calibration-npz", type=Path, help="NPZ spectral_calibration_model.npz.")
    parser.add_argument("--start-nm", required=True, type=float, help="Longitud de onda inicial.")
    parser.add_argument("--end-nm", required=True, type=float, help="Longitud de onda final.")
    parser.add_argument("--step-nm", required=True, type=float, help="Paso espectral en nm.")
    parser.add_argument(
        "--spectral-axis",
        default=-1,
        type=int,
        help="Eje de dispersion/pixeles en el .npy de entrada. Por defecto: -1.",
    )
    parser.add_argument(
        "--max-abs-residual-px",
        default=None,
        type=float,
        help="Descarta puntos con abs(residual_px) mayor a este umbral, si existe residual_px.",
    )
    parser.add_argument(
        "--max-wavelength-span-per-pixel-nm",
        default=80.0,
        type=float,
        help=(
            "Descarta grupos donde un mismo pixel representa demasiadas longitudes de onda. "
            "Usa un valor negativo para desactivar este filtro."
        ),
    )
    parser.add_argument(
        "--max-x-jump-px",
        default=30.0,
        type=float,
        help="Umbral para advertir saltos grandes en selected_x.",
    )
    parser.add_argument(
        "--out-of-range",
        choices=["nan", "clip"],
        default="nan",
        help="Que hacer fuera del rango calibrado: NaN o recortar al borde calibrado.",
    )
    parser.add_argument(
        "--output-dtype",
        default="float32",
        help="Tipo de dato de salida. Recomendado: float32.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.calibration_csv is not None and args.calibration_npz is not None:
        print("Se proporcionaron CSV y NPZ; se usara el NPZ.")

    input_npy = args.input_npy
    output_npy = args.output_npy

    if not input_npy.exists():
        raise FileNotFoundError(f"No existe input .npy: {input_npy}")

    max_span = args.max_wavelength_span_per_pixel_nm
    if max_span is not None and max_span < 0:
        max_span = None

    calibration = load_calibration(args.calibration_csv, args.calibration_npz)
    cal_w, cal_x = clean_calibration_points(
        calibration,
        max_abs_residual_px=args.max_abs_residual_px,
        max_x_jump_px=args.max_x_jump_px,
        max_wavelength_span_per_pixel_nm=max_span,
    )

    wavelengths_out = make_output_wavelength_axis(args.start_nm, args.end_nm, args.step_nm)

    data = np.load(input_npy, mmap_mode="r")
    print(f"Entrada: {input_npy}")
    print(f"Shape entrada: {data.shape}, dtype={data.dtype}")
    print(
        "Eje espectral salida: "
        f"{float(wavelengths_out[0]):.1f}-{float(wavelengths_out[-1]):.1f} nm, "
        f"{wavelengths_out.size} bandas"
    )

    reconstructed = reconstruct_spectral_cube(
        data,
        calibration_wavelengths_nm=cal_w,
        calibration_x_pixels=cal_x,
        output_wavelengths_nm=wavelengths_out,
        spectral_axis=args.spectral_axis,
        out_of_range=args.out_of_range,
        dtype=args.output_dtype,
    )

    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, reconstructed)

    wavelengths_path = output_npy.with_name(output_npy.stem + "_wavelengths_nm.npy")
    np.save(wavelengths_path, wavelengths_out.reshape(-1, 1))

    metadata = {
        "input_npy": str(input_npy),
        "output_npy": str(output_npy),
        "shape_input": tuple(int(v) for v in data.shape),
        "shape_output": tuple(int(v) for v in reconstructed.shape),
        "spectral_axis": int(args.spectral_axis),
        "wavelengths_path": str(wavelengths_path),
        "calibration_points_used": int(cal_w.size),
        "calibration_nm_range": [float(cal_w.min()), float(cal_w.max())],
        "calibration_x_range": [float(cal_x.min()), float(cal_x.max())],
        "out_of_range": args.out_of_range,
    }
    metadata_path = output_npy.with_name(output_npy.stem + "_metadata.npy")
    np.save(metadata_path, metadata)

    print(f"Salida guardada: {output_npy}")
    print(f"Eje espectral guardado: {wavelengths_path}")
    print(f"Metadata guardada: {metadata_path}")
    print(f"Shape salida: {reconstructed.shape}, dtype={reconstructed.dtype}")


if __name__ == "__main__":
    main()
