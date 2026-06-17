from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================
# CONFIGURACION
# =========================

SAMPLE_NPY = Path(r"Soil_721\cube_20260430_220712.npy")
SPECTRALON_NPY = Path(r"Blanco_espectralon\cube_20260505_091738.npy")

OUTPUT_DIR = Path(r"diagnostico_rango_espectral")

# Slit / eje espacial vertical del sensor.
SLIT_Y1 = 170
SLIT_Y2 = 820

# Para ahorrar tiempo/memoria en percentiles, usa un frame cada N.
# 1 usa todos los frames.
FRAME_STEP = 5

# Umbral relativo sugerido sobre perfil normalizado para estimar rango util.
THRESHOLD = 0.10


def load_npy_mmap(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")
    return np.load(path, mmap_mode="r")


def compute_profiles(path: Path) -> dict[str, np.ndarray]:
    raw = load_npy_mmap(path)
    print(f"{path.name}: shape={raw.shape}, dtype={raw.dtype}")

    if raw.ndim != 3:
        raise ValueError(f"Se esperaba raw 3D (y_sensor, x_sensor, frames), llego {raw.shape}")

    # slit: (y_slit, x_sensor, frames_subsampled)
    slit = raw[SLIT_Y1:SLIT_Y2, :, ::FRAME_STEP]
    slit_f = slit.astype(np.float32, copy=False)

    mean_profile = slit_f.mean(axis=(0, 2))
    median_profile = np.median(slit_f, axis=(0, 2))

    p99 = np.percentile(slit_f, 99, axis=(0, 2))
    p01 = np.percentile(slit_f, 1, axis=(0, 2))
    contrast_profile = p99 - p01

    return {
        "mean": mean_profile.astype(np.float32),
        "median": median_profile.astype(np.float32),
        "contrast": contrast_profile.astype(np.float32),
    }


def normalize(profile: np.ndarray) -> np.ndarray:
    profile = profile.astype(np.float32)
    pmin = float(np.min(profile))
    pmax = float(np.max(profile))
    if pmax <= pmin:
        return np.zeros_like(profile)
    return (profile - pmin) / (pmax - pmin)


def estimate_ranges(profile_norm: np.ndarray, threshold: float = THRESHOLD) -> list[tuple[int, int]]:
    mask = profile_norm >= threshold
    ranges = []
    start = None

    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            ranges.append((start, i))
            start = None

    if start is not None:
        ranges.append((start, len(mask)))

    return ranges


def plot_profiles(sample: dict[str, np.ndarray], spectralon: dict[str, np.ndarray]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x_cols = np.arange(sample["mean"].size)

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)

    for ax, key, title in [
        (axes[0], "mean", "Media dentro del slit"),
        (axes[1], "median", "Mediana dentro del slit"),
        (axes[2], "contrast", "Contraste p99-p1 dentro del slit"),
    ]:
        sample_norm = normalize(sample[key])
        spectralon_norm = normalize(spectralon[key])

        ax.plot(x_cols, sample_norm, label="Muestra Soil_721", linewidth=1.4)
        ax.plot(x_cols, spectralon_norm, label="Spectralon", linewidth=1.4)
        ax.axhline(THRESHOLD, color="red", linestyle="--", linewidth=1, label=f"umbral {THRESHOLD}")
        ax.set_ylabel("Normalizado 0-1")
        ax.set_title(title)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Columna x_sensor / lambda_pixel")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "perfiles_muestra_vs_spectralon.png", dpi=150)
    plt.close(fig)

    # Grafica de perfiles crudos por separado
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    for ax, key, title in [
        (axes[0], "mean", "Media cruda"),
        (axes[1], "median", "Mediana cruda"),
        (axes[2], "contrast", "Contraste crudo"),
    ]:
        ax.plot(x_cols, sample[key], label="Muestra Soil_721", linewidth=1.2)
        ax.plot(x_cols, spectralon[key], label="Spectralon", linewidth=1.2)
        ax.set_title(title)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Columna x_sensor / lambda_pixel")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "perfiles_crudos_muestra_vs_spectralon.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sample_profiles = compute_profiles(SAMPLE_NPY)
    spectralon_profiles = compute_profiles(SPECTRALON_NPY)

    plot_profiles(sample_profiles, spectralon_profiles)

    for key in ["mean", "median", "contrast"]:
        sample_ranges = estimate_ranges(normalize(sample_profiles[key]))
        spectralon_ranges = estimate_ranges(normalize(spectralon_profiles[key]))
        print(f"\nPerfil: {key}")
        print(f"  Rangos muestra >= {THRESHOLD}: {sample_ranges}")
        print(f"  Rangos Spectralon >= {THRESHOLD}: {spectralon_ranges}")

    np.savez(
        OUTPUT_DIR / "profiles_muestra_spectralon.npz",
        sample_mean=sample_profiles["mean"],
        sample_median=sample_profiles["median"],
        sample_contrast=sample_profiles["contrast"],
        spectralon_mean=spectralon_profiles["mean"],
        spectralon_median=spectralon_profiles["median"],
        spectralon_contrast=spectralon_profiles["contrast"],
    )

    print(f"\nGuardado en: {OUTPUT_DIR}")
    print("Revisa:")
    print(f"  {OUTPUT_DIR / 'perfiles_muestra_vs_spectralon.png'}")
    print(f"  {OUTPUT_DIR / 'perfiles_crudos_muestra_vs_spectralon.png'}")


if __name__ == "__main__":
    main()
