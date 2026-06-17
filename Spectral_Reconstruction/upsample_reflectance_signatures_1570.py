from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.interpolate import PchipInterpolator


# =========================
# CONFIGURACION
# =========================

INPUT_DIR = Path(r"Capturas_soil\Firmas")
OUTPUT_DIR = Path(r"Capturas_soil\Firmas_1570")
MAT_DIR = OUTPUT_DIR / "MAT"
NPY_DIR = OUTPUT_DIR / "NPY"
CSV_DIR = OUTPUT_DIR / "CSV"
SUMMARY_DIR = OUTPUT_DIR / "RESUMEN"

SIGNATURE_VAR = "reflectanciaSoilRecortada"
X_VAR = "bandasRecorte"

TARGET_LENGTH = 1570
METHOD = "pchip"  # "pchip" o "linear"


def as_vector(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def interpolate_nan_values(y: np.ndarray) -> np.ndarray:
    """Rellena NaN internos antes del upsampling."""
    y = y.astype(np.float64, copy=True)
    finite = np.isfinite(y)
    if finite.sum() < 2:
        raise ValueError("La firma tiene menos de 2 valores finitos.")
    if np.all(finite):
        return y

    x = np.arange(y.size, dtype=np.float64)
    y[~finite] = np.interp(x[~finite], x[finite], y[finite])
    return y


def upsample_signature(x_original: np.ndarray, y_original: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_original = as_vector(x_original)
    y_original = as_vector(y_original)

    if x_original.size != y_original.size:
        raise ValueError(
            f"X e Y tienen tamaños diferentes: {x_original.size} vs {y_original.size}"
        )
    if x_original.size < 2:
        raise ValueError("La firma necesita al menos 2 puntos.")

    order = np.argsort(x_original)
    x_original = x_original[order]
    y_original = y_original[order]

    unique_x, unique_idx = np.unique(x_original, return_index=True)
    if unique_x.size != x_original.size:
        print("  Advertencia: hay bandas duplicadas; se conserva la primera aparicion.")
        x_original = unique_x
        y_original = y_original[unique_idx]

    y_original = interpolate_nan_values(y_original)

    x_target = np.linspace(float(x_original.min()), float(x_original.max()), TARGET_LENGTH)

    if METHOD == "pchip":
        interpolator = PchipInterpolator(x_original, y_original, extrapolate=False)
        y_target = interpolator(x_target)
    elif METHOD == "linear":
        y_target = np.interp(x_target, x_original, y_original)
    else:
        raise ValueError("METHOD debe ser 'pchip' o 'linear'")

    y_target = np.asarray(y_target, dtype=np.float64).reshape(TARGET_LENGTH, 1)
    x_target = np.asarray(x_target, dtype=np.float64).reshape(TARGET_LENGTH, 1)
    return x_target, y_target


def process_file(path: Path) -> tuple[str, np.ndarray]:
    data = sio.loadmat(path)
    if SIGNATURE_VAR not in data:
        raise KeyError(f"{path.name} no contiene la variable {SIGNATURE_VAR}")

    y_original = as_vector(data[SIGNATURE_VAR])
    if X_VAR in data:
        x_original = as_vector(data[X_VAR])
    else:
        print(f"  Advertencia: {path.name} no contiene {X_VAR}; se usan indices 1..N.")
        x_original = np.arange(1, y_original.size + 1, dtype=np.float64)

    x_1570, y_1570 = upsample_signature(x_original, y_original)

    stem = path.stem
    out_mat = MAT_DIR / f"{stem}_reflectancia_1570.mat"
    out_csv = CSV_DIR / f"{stem}_reflectancia_1570.csv"
    out_npy = NPY_DIR / f"{stem}_reflectancia_1570.npy"

    sio.savemat(
        out_mat,
        {
            "reflectanciaSoil1570": y_1570,
            "ejeBandas1570": x_1570,
            "reflectanciaSoilOriginal": y_original.reshape(-1, 1),
            "bandasOriginal": x_original.reshape(-1, 1),
            "metodoUpsampling": METHOD,
            "targetLength": TARGET_LENGTH,
            "archivoOrigen": str(path),
        },
    )

    table = np.column_stack([x_1570.reshape(-1), y_1570.reshape(-1)])
    np.savetxt(
        out_csv,
        table,
        delimiter=",",
        header="eje_bandas_1570,reflectancia_soil_1570",
        comments="",
    )
    np.save(out_npy, y_1570.astype(np.float32))

    return stem, y_1570


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {INPUT_DIR}")

    for folder in [OUTPUT_DIR, MAT_DIR, NPY_DIR, CSV_DIR, SUMMARY_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    mat_files = sorted(INPUT_DIR.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No se encontraron .mat en: {INPUT_DIR}")

    print("Entrada:", INPUT_DIR)
    print("Salida:", OUTPUT_DIR)
    print("  MAT:", MAT_DIR)
    print("  NPY:", NPY_DIR)
    print("  CSV:", CSV_DIR)
    print("  RESUMEN:", SUMMARY_DIR)
    print("Archivos .mat encontrados:", len(mat_files))
    print("Longitud objetivo:", TARGET_LENGTH)
    print("Metodo:", METHOD)

    names = []
    signatures = []
    for path in mat_files:
        print("Procesando:", path.name)
        name, y_1570 = process_file(path)
        names.append(name)
        signatures.append(y_1570.reshape(-1))

    matrix = np.vstack(signatures).astype(np.float32)
    np.save(SUMMARY_DIR / "todas_las_firmas_1570_muestras_x_bandas.npy", matrix)
    np.savetxt(
        SUMMARY_DIR / "todas_las_firmas_1570_muestras_x_bandas.csv",
        matrix,
        delimiter=",",
        header=",".join([f"banda_{i + 1}" for i in range(TARGET_LENGTH)]),
        comments="",
    )

    sio.savemat(
        SUMMARY_DIR / "todas_las_firmas_1570.mat",
        {
            "firmas1570": matrix,
            "nombres": np.array(names, dtype=object).reshape(-1, 1),
            "targetLength": TARGET_LENGTH,
            "metodoUpsampling": METHOD,
        },
    )

    print("\nListo.")
    print("Firmas individuales guardadas por formato:")
    print("  .mat:", MAT_DIR)
    print("  .npy:", NPY_DIR)
    print("  .csv:", CSV_DIR)
    print("Matriz global:", SUMMARY_DIR / "todas_las_firmas_1570_muestras_x_bandas.npy")
    print("Shape matriz global:", matrix.shape)


if __name__ == "__main__":
    main()
