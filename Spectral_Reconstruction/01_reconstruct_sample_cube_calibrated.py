from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================
# CONFIGURACION
# =========================

SAMPLE_NPY = Path(r"Colorckeher\Cube_colorcheker1.npy")
CALIBRATION_CSV = Path(r"spectral_characterization_plots\spectral_characterization.csv")
OUTPUT_DIR = Path(r"preparado_muestra_colorckeher_calibrado")

# raw.shape = (y_sensor, x_sensor, frames)
SLIT_Y1 = 300
SLIT_Y2 = 850

# Si los dejas en None, el script elige automaticamente el rango espectral
# confiable a partir del CSV de caracterizacion.
SPECTRAL_X1 = None
SPECTRAL_X2 = None

FLIP_SCAN_AXIS = True
FLIP_SLIT_AXIS = False
FLIP_LAMBDA_AXIS = False

# Excluye pixeles donde muchas longitudes de onda colapsan en el mismo x.
# En tu CSV, x=452 agrupa aprox. 1181-1699 nm, asi que no es resoluble.
MAX_WAVELENGTH_SPAN_PER_PIXEL_NM = 120.0

# "uint8" conserva el cubo con el mismo tipo del .npy y ocupa menos espacio.
# Usa "float32" si vas a hacer operaciones radiometricas inmediatamente.
OUTPUT_DTYPE = "uint8"


def robust_limits(img: np.ndarray, low: float = 1, high: float = 99) -> tuple[float, float]:
    vmin, vmax = np.percentile(img, [low, high])
    if vmax <= vmin:
        vmin = float(np.min(img))
        vmax = float(np.max(img))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def load_npy_checked(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Archivo vacio: {path}")
    try:
        return np.load(path, mmap_mode="r")
    except ValueError as exc:
        raise ValueError(f"No pude leer {path}; el .npy parece incompleto/corrupto.") from exc


def load_calibration_table(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"No existe el CSV de calibracion: {path}")

    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    required = {"wavelength_nm", "selected_x"}
    missing = required.difference(table.dtype.names or ())
    if missing:
        raise ValueError(f"Al CSV le faltan columnas requeridas: {sorted(missing)}")
    return table


def grouped_pixel_calibration(table: np.ndarray) -> np.ndarray:
    selected_x = table["selected_x"].astype(np.int32)
    wavelengths = table["wavelength_nm"].astype(np.float64)

    rows = []
    for x in np.unique(selected_x):
        values = wavelengths[selected_x == x]
        rows.append(
            (
                int(x),
                float(np.median(values)),
                float(np.min(values)),
                float(np.max(values)),
                float(np.max(values) - np.min(values)),
                int(values.size),
            )
        )

    dtype = [
        ("sensor_x", "i4"),
        ("wavelength_nm", "f4"),
        ("wavelength_min_nm", "f4"),
        ("wavelength_max_nm", "f4"),
        ("wavelength_span_nm", "f4"),
        ("measurements", "i4"),
    ]
    grouped = np.array(rows, dtype=dtype)
    return np.sort(grouped, order="sensor_x")


def usable_calibration(grouped: np.ndarray) -> np.ndarray:
    ok = grouped["wavelength_span_nm"] <= MAX_WAVELENGTH_SPAN_PER_PIXEL_NM
    usable = grouped[ok].copy()
    if usable.size < 2:
        raise ValueError("No quedaron suficientes puntos confiables de calibracion.")

    order = np.argsort(usable["sensor_x"])
    usable = usable[order]
    usable["wavelength_nm"] = isotonic_increasing(usable["wavelength_nm"].astype(np.float64))
    return usable


def isotonic_increasing(values: np.ndarray) -> np.ndarray:
    """Ajuste monotono creciente con PAVA, sin depender de scikit-learn."""
    y = np.asarray(values, dtype=np.float64)
    levels: list[float] = []
    weights: list[float] = []

    for value in y:
        levels.append(float(value))
        weights.append(1.0)

        while len(levels) >= 2 and levels[-2] > levels[-1]:
            total_weight = weights[-2] + weights[-1]
            pooled = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / total_weight
            levels[-2] = pooled
            weights[-2] = total_weight
            levels.pop()
            weights.pop()

    return np.repeat(np.array(levels, dtype=np.float64), np.array(weights, dtype=np.int32))


def choose_spectral_range(usable: np.ndarray) -> tuple[int, int]:
    if SPECTRAL_X1 is not None and SPECTRAL_X2 is not None:
        return int(SPECTRAL_X1), int(SPECTRAL_X2)

    x1 = int(np.min(usable["sensor_x"]))
    x2 = int(np.max(usable["sensor_x"])) + 1
    return x1, x2


def wavelengths_for_columns(cols: np.ndarray, usable: np.ndarray) -> np.ndarray:
    x_cal = usable["sensor_x"].astype(np.float64)
    w_cal = usable["wavelength_nm"].astype(np.float64)

    order = np.argsort(x_cal)
    wavelengths = np.interp(cols.astype(np.float64), x_cal[order], w_cal[order])

    if FLIP_LAMBDA_AXIS:
        wavelengths = wavelengths[::-1]

    return wavelengths.astype(np.float32)


def reconstruct_cube(path: Path, x1: int, x2: int) -> np.ndarray:
    raw = load_npy_checked(path)
    print(f"{path.name} raw shape:", raw.shape, raw.dtype)

    if raw.ndim != 3:
        raise ValueError(f"Se esperaba raw 3D (y_sensor, x_sensor, frames), llego {raw.shape}")
    if not (0 <= SLIT_Y1 < SLIT_Y2 <= raw.shape[0]):
        raise ValueError(f"Recorte slit invalido {SLIT_Y1}:{SLIT_Y2} para raw.shape[0]={raw.shape[0]}")
    if not (0 <= x1 < x2 <= raw.shape[1]):
        raise ValueError(f"Recorte espectral invalido {x1}:{x2} para raw.shape[1]={raw.shape[1]}")

    raw_crop = raw[SLIT_Y1:SLIT_Y2, x1:x2, :]
    cube = np.transpose(raw_crop, (0, 2, 1))  # (y_slit, x_scan, lambda_pixel)

    if FLIP_SLIT_AXIS:
        cube = np.flip(cube, axis=0)
    if FLIP_SCAN_AXIS:
        cube = np.flip(cube, axis=1)
    if FLIP_LAMBDA_AXIS:
        cube = np.flip(cube, axis=2)

    if OUTPUT_DTYPE == "float32":
        cube = cube.astype(np.float32, copy=False)
    elif OUTPUT_DTYPE == "uint8":
        cube = cube.astype(np.uint8, copy=False)
    else:
        raise ValueError("OUTPUT_DTYPE debe ser 'uint8' o 'float32'")

    print(f"Cubo calibrado shape:", cube.shape, cube.dtype)
    return cube


def save_raw_slit_diagnostic(out_dir: Path, raw_path: Path, x1: int, x2: int) -> None:
    raw = load_npy_checked(raw_path)
    frame_index = raw.shape[2] // 2
    frame = raw[:, :, frame_index]
    vmin, vmax = robust_limits(frame, low=1, high=99.8)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(frame, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
    ax.axhline(SLIT_Y1, color="red", linestyle="--", linewidth=1.2, label="slit y")
    ax.axhline(SLIT_Y2, color="red", linestyle="--", linewidth=1.2)
    ax.axvline(x1, color="lime", linestyle="--", linewidth=1.2, label="rango espectral calibrado")
    ax.axvline(x2, color="lime", linestyle="--", linewidth=1.2)
    ax.set_title(f"Frame raw muestra {frame_index}: rojo=slit, verde=calibracion util")
    ax.set_xlabel("x_sensor / lambda_pixel")
    ax.set_ylabel("y_sensor")
    ax.legend(loc="best")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostico_frame_raw_muestra_calibrado.png", dpi=150)
    plt.close(fig)


def save_preview(out_dir: Path, cube: np.ndarray) -> None:
    preview = cube.mean(axis=2)
    vmin, vmax = robust_limits(preview)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(preview, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title("Preview cubo muestra reconstruido con calibracion espectral")
    ax.set_xlabel("x_scan / frames")
    ax.set_ylabel("y_slit")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "preview_sample_cube_calibrated.png", dpi=150)
    plt.close(fig)

    np.save(out_dir / "preview_sample_cube_calibrated.npy", preview.astype(np.float32))


def save_calibration_plot(out_dir: Path, grouped: np.ndarray, usable: np.ndarray, cols: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(
        grouped["sensor_x"],
        grouped["wavelength_nm"],
        s=15,
        alpha=0.35,
        label="CSV agrupado por pixel",
    )
    ax.plot(usable["sensor_x"], usable["wavelength_nm"], color="tab:blue", label="calibracion usada")
    ax.axvspan(cols[0], cols[-1], color="tab:green", alpha=0.12, label="rango reconstruido")
    ax.set_xlabel("Columna espectral del sensor")
    ax.set_ylabel("Longitud de onda (nm)")
    ax.set_title("Calibracion espectral usada para el cubo")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "calibracion_usada_wavelength_vs_x.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    calibration = load_calibration_table(CALIBRATION_CSV)
    grouped = grouped_pixel_calibration(calibration)
    usable = usable_calibration(grouped)

    x1, x2 = choose_spectral_range(usable)
    spectral_cols = np.arange(x1, x2, dtype=np.int32)
    wavelengths_nm = wavelengths_for_columns(spectral_cols, usable)

    cube = reconstruct_cube(SAMPLE_NPY, x1, x2)

    np.save(OUTPUT_DIR / "sample_cube_y_xscan_lambda_calibrated.npy", cube)
    np.save(OUTPUT_DIR / "spectral_pixel_columns.npy", spectral_cols.reshape(-1, 1))
    np.save(OUTPUT_DIR / "wavelengths_nm.npy", wavelengths_nm.reshape(-1, 1))
    np.save(OUTPUT_DIR / "calibration_grouped_by_pixel.npy", grouped)
    np.save(OUTPUT_DIR / "calibration_used.npy", usable)

    save_raw_slit_diagnostic(OUTPUT_DIR, SAMPLE_NPY, x1, x2)
    save_preview(OUTPUT_DIR, cube)
    save_calibration_plot(OUTPUT_DIR, grouped, usable, spectral_cols)

    metadata = {
        "sample_npy": str(SAMPLE_NPY),
        "calibration_csv": str(CALIBRATION_CSV),
        "sample_cube_shape": tuple(int(v) for v in cube.shape),
        "axis_order": "(y_slit, x_scan, lambda_pixel)",
        "slit_y_crop": [SLIT_Y1, SLIT_Y2],
        "spectral_x_crop": [x1, x2],
        "wavelength_nm_minmax": [float(wavelengths_nm.min()), float(wavelengths_nm.max())],
        "max_wavelength_span_per_pixel_nm": MAX_WAVELENGTH_SPAN_PER_PIXEL_NM,
        "excluded_note": "Pixeles con colapso espectral mayor al umbral no se usan para definir el rango.",
    }
    np.save(OUTPUT_DIR / "metadata_sample_calibrated.npy", metadata)

    print("Rango espectral usado:", x1, x2)
    print("Longitudes de onda aprox:", float(wavelengths_nm.min()), "a", float(wavelengths_nm.max()), "nm")
    print("Muestra calibrada guardada en:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
