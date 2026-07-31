"""Crea resultado.npz para firmas manuales guardadas desde MATLAB.

Este helper existe porque algunas versiones de MATLAB no pueden usar el Python
embebido con NumPy, pero si pueden llamar a `python` por consola.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def vector(data: dict[str, object], key: str) -> np.ndarray:
    return np.asarray(data[key], dtype=np.float32).ravel()


def matrix(data: dict[str, object], key: str, dtype: type) -> np.ndarray:
    return np.asarray(data[key], dtype=dtype)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", required=True, type=Path)
    parser.add_argument("--npz", required=True, type=Path)
    args = parser.parse_args()

    data = loadmat(args.mat)
    args.npz.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.npz,
        soil_signature=vector(data, "soil_signature"),
        white_signature=vector(data, "white_signature"),
        dark_signature=vector(data, "dark_signature"),
        soil_reflectance=vector(data, "soil_reflectance"),
        soil_mask=matrix(data, "maskSoil", bool),
        white_mask=matrix(data, "maskWhite", bool),
        dark_mask=matrix(data, "maskDark", bool),
        preview=matrix(data, "preview_manual", np.float32),
        preview_range=np.asarray(data["preview_range"], dtype=np.int32).ravel(),
        preview_subranges=np.asarray(data["preview_subranges"], dtype=np.int32),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
