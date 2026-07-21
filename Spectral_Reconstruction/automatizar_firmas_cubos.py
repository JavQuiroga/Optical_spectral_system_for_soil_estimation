r"""
Extraccion automatica de firmas SOIL / WHITE / DARK desde cubos .npy.

El programa:
1. Busca cubos recursivamente.
2. Construye una representacion multibanda robusta.
3. Segmenta regiones por K-means.
4. Clasifica componentes como SOIL, WHITE y DARK.
5. Contrae cada mascara para usar solo su zona interior.
6. Extrae firmas completas y calcula reflectancia.
7. Guarda diagnosticos, resultados por cubo y un resumen CSV.

Ejemplos:
    python automatizar_firmas_cubos.py --self-test

    python Spectral_Reconstruction\automatizar_firmas_cubos.py ^
        --input-dir "Spectral_Reconstruction\Capturas_soil" ^
        --output-dir "Spectral_Reconstruction\Firmas_automaticas" ^
        --layout y_lambda_x ^
        --preview-start 390 ^
        --preview-stop 550 ^
        --limit 20

Los indices siguen la convencion Python: start incluido, stop excluido.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans


ROLES = ("soil", "white", "dark")
ROLE_COLORS = {
    "soil": "#f0c419",
    "white": "#ffffff",
    "dark": "#00b7ff",
}


@dataclass
class Config:
    layout: str = "y_lambda_x"
    preview_start: int | None = 390
    preview_stop: int | None = 679
    n_subranges: int = 3
    n_clusters: int = 5
    k_values: tuple[int, ...] = (4, 5)
    sample_pixels: int = 80_000
    random_seed: int = 17
    spatial_smoothing_fraction: float = 0.008
    min_area_fraction: float = 0.0008
    max_area_fraction: float = 0.45
    border_margin_fraction: float = 0.015
    interior_fraction: float = 0.35
    circular_roi_radius_fraction: float = 0.22
    soil_roi_radius_pixels: float = 90.0
    dark_corner_search_fraction: float = 0.30
    dark_rectangle_height_fraction: float = 0.08
    dark_rectangle_width_fraction: float = 0.08
    dark_corner_margin_fraction: float = 0.01
    min_roi_pixels: int = 30
    reduction: str = "median"
    expected_soil: tuple[float, float] | None = (0.43, 0.55)
    expected_white: tuple[float, float] | None = None
    expected_dark: tuple[float, float] | None = None
    max_position_distance: float = 0.55
    soil_max_position_distance: float = 0.28
    min_soil_position_score: float = 0.18
    min_confidence: float = 0.42
    max_invalid_reflectance_fraction: float = 0.20


@dataclass
class Candidate:
    cluster: int
    component: int
    area: int
    area_fraction: float
    centroid_y: float
    centroid_x: float
    brightness: float
    border_fraction: float
    fill_fraction: float
    bbox_y0: int
    bbox_y1: int
    bbox_x0: int
    bbox_x1: int
    mask: np.ndarray


@dataclass
class CubeResult:
    cube_id: str
    input_path: str
    status: str
    confidence: float
    reason: str
    shape_y: int
    shape_x: int
    bands: int
    soil_pixels: int
    white_pixels: int
    dark_pixels: int
    invalid_reflectance_fraction: float
    reflectance_outside_fraction: float


def parse_point(text: str | None) -> tuple[float, float] | None:
    if text is None:
        return None
    pieces = [piece.strip() for piece in text.split(",")]
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("Use coordenadas normalizadas como x,y; ejemplo: 0.5,0.4")
    x, y = (float(value) for value in pieces)
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise argparse.ArgumentTypeError("Las coordenadas x,y deben estar entre 0 y 1.")
    return x, y


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(piece.strip()) for piece in text.split(",") if piece.strip())
    if not values:
        raise argparse.ArgumentTypeError("Debe indicar al menos un valor de K.")
    if any(value < 2 for value in values):
        raise argparse.ArgumentTypeError("Todos los valores de K deben ser >= 2.")
    return values


def load_cube(path: Path, layout: str) -> np.ndarray:
    cube = np.load(path, mmap_mode="r")
    if cube.ndim != 3:
        raise ValueError(f"Se esperaba un cubo 3D y se obtuvo {cube.shape}")

    if layout == "y_lambda_x":
        cube_y_x_lambda = np.moveaxis(cube, 1, 2)
    elif layout == "y_x_lambda":
        cube_y_x_lambda = cube
    else:
        raise ValueError(f"Layout desconocido: {layout}")

    if min(cube_y_x_lambda.shape) < 2:
        raise ValueError(f"Dimensiones no validas: {cube_y_x_lambda.shape}")
    return cube_y_x_lambda


def resolve_preview_range(n_bands: int, config: Config) -> tuple[int, int]:
    start = 0 if config.preview_start is None else config.preview_start
    stop = n_bands if config.preview_stop is None else config.preview_stop
    if start < 0:
        start += n_bands
    if stop < 0:
        stop += n_bands
    start = max(0, start)
    stop = min(n_bands, stop)
    if start >= stop:
        raise ValueError(
            f"Rango de preview invalido {start}:{stop} para un cubo con {n_bands} bandas."
        )
    return start, stop


def robust_normalize(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    normalized = (image - low) / (high - low)
    return np.clip(normalized, 0, 1).astype(np.float32)


def build_multiband_features(
    cube: np.ndarray, config: Config
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    n_bands = cube.shape[2]
    start, stop = resolve_preview_range(n_bands, config)
    edges = np.linspace(start, stop, config.n_subranges + 1, dtype=int)

    channels: list[np.ndarray] = []
    ranges: list[tuple[int, int]] = []
    for index in range(config.n_subranges):
        first, last = int(edges[index]), int(edges[index + 1])
        if first >= last:
            continue
        image = np.nanmedian(cube[:, :, first:last], axis=2)
        normalized = robust_normalize(image)
        sigma = max(0.0, min(normalized.shape) * config.spatial_smoothing_fraction)
        if sigma > 0:
            normalized = ndimage.gaussian_filter(normalized, sigma=sigma)
        channels.append(robust_normalize(normalized))
        ranges.append((first, last))

    if not channels:
        raise ValueError("No fue posible construir canales para el preview.")

    feature_cube = np.stack(channels, axis=2)
    broadband = robust_normalize(np.nanmedian(cube[:, :, start:stop], axis=2))
    return feature_cube, broadband, (start, stop), ranges


def segment_features(features: np.ndarray, config: Config, n_clusters: int) -> np.ndarray:
    height, width, n_features = features.shape
    flat = features.reshape(-1, n_features)
    valid = np.all(np.isfinite(flat), axis=1)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < n_clusters:
        raise ValueError("No hay suficientes pixeles validos para segmentar.")

    rng = np.random.default_rng(config.random_seed)
    if valid_indices.size > config.sample_pixels:
        training_indices = rng.choice(
            valid_indices, size=config.sample_pixels, replace=False
        )
    else:
        training_indices = valid_indices

    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=config.random_seed,
        n_init=5,
        batch_size=4096,
    )
    model.fit(flat[training_indices])

    labels = np.full(flat.shape[0], -1, dtype=np.int16)
    labels[valid] = model.predict(flat[valid]).astype(np.int16)
    return labels.reshape(height, width)


def border_mask(shape: tuple[int, int], fraction: float) -> np.ndarray:
    height, width = shape
    margin = max(1, int(round(min(height, width) * fraction)))
    result = np.zeros(shape, dtype=bool)
    result[:margin, :] = True
    result[-margin:, :] = True
    result[:, :margin] = True
    result[:, -margin:] = True
    return result


def extract_candidates(
    labels: np.ndarray, broadband: np.ndarray, config: Config
) -> list[Candidate]:
    height, width = labels.shape
    image_area = height * width
    min_area = max(config.min_roi_pixels, int(image_area * config.min_area_fraction))
    max_area = int(image_area * config.max_area_fraction)
    outer_border = border_mask(labels.shape, config.border_margin_fraction)
    candidates: list[Candidate] = []

    structure = np.ones((3, 3), dtype=np.uint8)
    valid_clusters = [int(value) for value in np.unique(labels) if value >= 0]
    for cluster in valid_clusters:
        cluster_mask = labels == cluster
        cluster_mask = ndimage.binary_opening(cluster_mask, structure=structure)
        cluster_mask = ndimage.binary_closing(cluster_mask, structure=structure)
        components, count = ndimage.label(cluster_mask, structure=structure)

        for component in range(1, count + 1):
            mask = components == component
            area = int(np.count_nonzero(mask))
            if area < min_area or area > max_area:
                continue

            ys, xs = np.nonzero(mask)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            bbox_area = max(1, (y1 - y0) * (x1 - x0))
            brightness = float(np.nanmedian(broadband[mask]))
            border_fraction_value = float(np.count_nonzero(mask & outer_border) / area)

            candidates.append(
                Candidate(
                    cluster=cluster,
                    component=component,
                    area=area,
                    area_fraction=area / image_area,
                    centroid_y=float(np.mean(ys) / max(height - 1, 1)),
                    centroid_x=float(np.mean(xs) / max(width - 1, 1)),
                    brightness=brightness,
                    border_fraction=border_fraction_value,
                    fill_fraction=area / bbox_area,
                    bbox_y0=y0,
                    bbox_y1=y1,
                    bbox_x0=x0,
                    bbox_x1=x1,
                    mask=mask,
                )
            )

    if len(candidates) < 3:
        raise ValueError(
            f"Solo se encontraron {len(candidates)} componentes candidatos; se necesitan al menos 3."
        )
    return candidates


def position_score(
    candidate: Candidate,
    expected: tuple[float, float] | None,
    max_distance: float,
) -> float:
    if expected is None:
        return 0.65
    expected_x, expected_y = expected
    distance = math.hypot(
        candidate.centroid_x - expected_x, candidate.centroid_y - expected_y
    )
    return float(np.clip(1 - distance / max(max_distance, 1e-6), 0, 1))


def candidate_score(
    candidate: Candidate,
    role: str,
    brightness_rank: float,
    expected: tuple[float, float] | None,
    config: Config,
) -> float:
    if role == "white":
        intensity_score = brightness_rank
    elif role == "dark":
        intensity_score = 1 - brightness_rank
    else:
        intensity_score = 1 - min(abs(brightness_rank - 0.5) * 2, 1)

    position_distance = (
        config.soil_max_position_distance if role == "soil" else config.max_position_distance
    )
    position = position_score(candidate, expected, position_distance)
    border = float(np.clip(1 - 2.5 * candidate.border_fraction, 0, 1))
    compactness = float(np.clip(candidate.fill_fraction / 0.55, 0, 1))
    bbox_height = max(1, candidate.bbox_y1 - candidate.bbox_y0)
    bbox_width = max(1, candidate.bbox_x1 - candidate.bbox_x0)
    roundness = min(bbox_height, bbox_width) / max(bbox_height, bbox_width)
    area_score = float(
        np.clip(candidate.area_fraction / max(config.min_area_fraction * 8, 1e-9), 0, 1)
    )

    if role == "soil":
        # SOIL debe corresponder al circulo grande de la izquierda/centro.
        # Por eso aqui la posicion y la forma pesan mas que la intensidad pura:
        # la intensidad ayuda, pero no debe permitir que la ROI salte a otra zona.
        large_soil_score = float(
            np.clip(candidate.area_fraction / max(config.min_area_fraction * 35, 1e-9), 0, 1)
        )
        return (
            0.40 * position
            + 0.20 * roundness
            + 0.18 * large_soil_score
            + 0.10 * intensity_score
            + 0.08 * compactness
            + 0.04 * border
        )

    return (
        0.34 * intensity_score
        + 0.16 * position
        + 0.12 * border
        + 0.12 * compactness
        + 0.18 * roundness
        + 0.08 * area_score
    )


def assign_foreground_roles(
    candidates: list[Candidate], config: Config
) -> tuple[dict[str, Candidate], dict[str, float]]:
    brightness = np.array([candidate.brightness for candidate in candidates])
    order = np.argsort(np.argsort(brightness))
    ranks = order / max(len(candidates) - 1, 1)
    rank_by_id = {id(candidate): float(rank) for candidate, rank in zip(candidates, ranks)}
    expected_by_role = {
        "soil": config.expected_soil,
        "white": config.expected_white,
    }

    best_total = -np.inf
    best_assignment: dict[str, Candidate] | None = None
    best_scores: dict[str, float] | None = None

    limited = sorted(candidates, key=lambda item: item.area, reverse=True)[:24]
    for soil in limited:
        if config.expected_soil is not None:
            soil_position = position_score(
                soil, config.expected_soil, config.soil_max_position_distance
            )
            if soil_position < config.min_soil_position_score:
                continue
        for white in limited:
            if white is soil:
                continue
            assignment = {"soil": soil, "white": white}
            if not white.brightness > soil.brightness:
                continue

            scores = {
                role: candidate_score(
                    candidate,
                    role,
                    rank_by_id[id(candidate)],
                    expected_by_role[role],
                    config,
                )
                for role, candidate in assignment.items()
            }
            separation = white.brightness - soil.brightness
            total = sum(scores.values()) + 0.35 * float(np.clip(separation / 0.15, 0, 1))
            if total > best_total:
                best_total = total
                best_assignment = assignment
                best_scores = scores

    if best_assignment is None or best_scores is None:
        raise ValueError("No se encontro una asignacion WHITE > SOIL coherente.")
    return best_assignment, best_scores


def make_circular_mask(candidate: Candidate, config: Config, role: str) -> np.ndarray:
    height, width = candidate.mask.shape
    if role == "soil":
        center_y = candidate.centroid_y * max(height - 1, 1)
        center_x = candidate.centroid_x * max(width - 1, 1)
    else:
        center_y = (candidate.bbox_y0 + candidate.bbox_y1 - 1) / 2
        center_x = (candidate.bbox_x0 + candidate.bbox_x1 - 1) / 2
    object_height = candidate.bbox_y1 - candidate.bbox_y0
    object_width = candidate.bbox_x1 - candidate.bbox_x0
    if role == "soil":
        radius = float(config.soil_roi_radius_pixels)
    else:
        radius = max(
            2.0,
            min(object_height, object_width) * config.circular_roi_radius_fraction,
        )
    yy, xx = np.indices((height, width))
    circle = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2
    if np.count_nonzero(circle) < config.min_roi_pixels:
        raise ValueError("La ROI circular resulto demasiado pequena.")
    return circle


def select_dark_corner_rectangle(
    broadband: np.ndarray, config: Config
) -> tuple[np.ndarray, dict[str, int | float | str], float]:
    height, width = broadband.shape
    rect_h = max(3, int(round(height * config.dark_rectangle_height_fraction)))
    rect_w = max(3, int(round(width * config.dark_rectangle_width_fraction)))
    search_h = max(rect_h, int(round(height * config.dark_corner_search_fraction)))
    search_w = max(rect_w, int(round(width * config.dark_corner_search_fraction)))
    margin_y = int(round(height * config.dark_corner_margin_fraction))
    margin_x = int(round(width * config.dark_corner_margin_fraction))
    step_y = max(1, rect_h // 3)
    step_x = max(1, rect_w // 3)

    corners = {
        "top_left": (margin_y, min(height, margin_y + search_h), margin_x, min(width, margin_x + search_w)),
        "top_right": (margin_y, min(height, margin_y + search_h), max(0, width - margin_x - search_w), width - margin_x),
        "bottom_left": (max(0, height - margin_y - search_h), height - margin_y, margin_x, min(width, margin_x + search_w)),
        "bottom_right": (
            max(0, height - margin_y - search_h),
            height - margin_y,
            max(0, width - margin_x - search_w),
            width - margin_x,
        ),
    }

    best: tuple[float, str, int, int] | None = None
    for corner_name, (y0, y1, x0, x1) in corners.items():
        y_starts = list(range(y0, max(y0 + 1, y1 - rect_h + 1), step_y))
        x_starts = list(range(x0, max(x0 + 1, x1 - rect_w + 1), step_x))
        if y1 - rect_h >= y0:
            y_starts.append(y1 - rect_h)
        if x1 - rect_w >= x0:
            x_starts.append(x1 - rect_w)

        for rect_y0 in sorted(set(y_starts)):
            for rect_x0 in sorted(set(x_starts)):
                patch = broadband[rect_y0 : rect_y0 + rect_h, rect_x0 : rect_x0 + rect_w]
                if patch.shape != (rect_h, rect_w):
                    continue
                darkness = float(np.nanmedian(patch))
                texture_penalty = float(np.nanstd(patch)) * 0.20
                objective = darkness + texture_penalty
                if best is None or objective < best[0]:
                    best = (objective, corner_name, rect_y0, rect_x0)

    if best is None:
        raise ValueError("No fue posible ubicar la ROI rectangular DARK.")

    objective, corner_name, y0, x0 = best
    mask = np.zeros_like(broadband, dtype=bool)
    mask[y0 : y0 + rect_h, x0 : x0 + rect_w] = True
    brightness = float(np.nanmedian(broadband[mask]))
    score = float(np.clip(1 - brightness, 0, 1))
    metadata = {
        "corner": corner_name,
        "y0": int(y0),
        "y1": int(y0 + rect_h),
        "x0": int(x0),
        "x1": int(x0 + rect_w),
        "brightness": brightness,
        "objective": float(objective),
    }
    return mask, metadata, score


def segmentation_quality(
    assignment: dict[str, Candidate], role_scores: dict[str, float]
) -> float:
    soil = assignment["soil"]
    white = assignment["white"]

    brightness_gap = max(0.0, white.brightness - soil.brightness)
    soil_shape = 0.5 * soil.fill_fraction + 0.5 * (
        min(soil.bbox_y1 - soil.bbox_y0, soil.bbox_x1 - soil.bbox_x0)
        / max(soil.bbox_y1 - soil.bbox_y0, soil.bbox_x1 - soil.bbox_x0, 1)
    )
    white_shape = 0.5 * white.fill_fraction + 0.5 * (
        min(white.bbox_y1 - white.bbox_y0, white.bbox_x1 - white.bbox_x0)
        / max(white.bbox_y1 - white.bbox_y0, white.bbox_x1 - white.bbox_x0, 1)
    )
    border_penalty = soil.border_fraction + white.border_fraction
    area_balance = min(soil.area, white.area) / max(max(soil.area, white.area), 1)

    return float(
        0.40 * min(role_scores.values())
        + 0.20 * np.clip(brightness_gap / 0.25, 0, 1)
        + 0.18 * np.clip(soil_shape, 0, 1)
        + 0.14 * np.clip(white_shape, 0, 1)
        + 0.08 * np.clip(area_balance / 0.15, 0, 1)
        - 0.18 * border_penalty
    )


def choose_best_segmentation(
    features: np.ndarray, broadband: np.ndarray, config: Config
) -> tuple[np.ndarray, list[Candidate], dict[str, Candidate], dict[str, float], dict[str, object]]:
    attempts: list[dict[str, object]] = []
    best: tuple[int, float, np.ndarray, list[Candidate], dict[str, Candidate], dict[str, float]] | None = None

    for k in config.k_values:
        try:
            labels = segment_features(features, config, n_clusters=k)
            candidates = extract_candidates(labels, broadband, config)
            assignment, role_scores = assign_foreground_roles(candidates, config)
            quality = segmentation_quality(assignment, role_scores)
            attempts.append(
                {
                    "k": int(k),
                    "status": "ok",
                    "quality": quality,
                    "num_candidates": len(candidates),
                    "role_scores": role_scores,
                    "soil_area": int(assignment["soil"].area),
                    "white_area": int(assignment["white"].area),
                    "soil_brightness": float(assignment["soil"].brightness),
                    "white_brightness": float(assignment["white"].brightness),
                    "soil_centroid_x": float(assignment["soil"].centroid_x),
                    "soil_centroid_y": float(assignment["soil"].centroid_y),
                }
            )
            if best is None or quality > best[1]:
                best = (int(k), quality, labels, candidates, assignment, role_scores)
        except Exception as exc:
            attempts.append({"k": int(k), "status": "error", "error": str(exc)})

    if best is None:
        raise ValueError("Ningun valor de K produjo una segmentacion usable.")

    selected_k, quality, labels, candidates, assignment, role_scores = best
    diagnostics = {
        "selected_k": selected_k,
        "selected_quality": quality,
        "attempts": attempts,
    }
    return labels, candidates, assignment, role_scores, diagnostics


def signature_from_mask(
    cube: np.ndarray, mask: np.ndarray, reduction: str
) -> np.ndarray:
    pixels = np.asarray(cube[mask, :], dtype=np.float32)
    if pixels.shape[0] == 0:
        raise ValueError("Una mascara no contiene pixeles.")
    if reduction == "median":
        return np.nanmedian(pixels, axis=0).astype(np.float32)
    if reduction == "mean":
        return np.nanmean(pixels, axis=0).astype(np.float32)
    raise ValueError("reduction debe ser 'mean' o 'median'.")


def calculate_reflectance(
    soil: np.ndarray, white: np.ndarray, dark: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    denominator = white - dark
    finite_abs = np.abs(denominator[np.isfinite(denominator)])
    scale = float(np.percentile(finite_abs, 5)) if finite_abs.size else 0.0
    epsilon = max(scale * 0.05, np.finfo(np.float32).eps)
    valid = np.isfinite(denominator) & (np.abs(denominator) > epsilon)
    reflectance = np.full_like(soil, np.nan, dtype=np.float32)
    reflectance[valid] = (soil[valid] - dark[valid]) / denominator[valid]
    return reflectance, valid


def cube_identifier(path: Path, input_dir: Path) -> str:
    relative = path.relative_to(input_dir)
    parts = list(relative.with_suffix("").parts)
    return "__".join(parts)


def save_diagnostic(
    path: Path,
    features: np.ndarray,
    broadband: np.ndarray,
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
    reflectance: np.ndarray,
    confidence: float,
    status: str,
) -> None:
    rgb = np.zeros((*broadband.shape, 3), dtype=np.float32)
    if features.shape[2] >= 3:
        rgb[:, :, :] = features[:, :, [2, 1, 0]]
    else:
        rgb[:, :, :] = broadband[:, :, None]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes[0, 0].imshow(broadband, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Preview robusto multibanda")
    axes[0, 1].imshow(rgb)
    axes[0, 1].set_title("Falso color de subrangos")
    axes[1, 0].imshow(labels, cmap="tab20")
    axes[1, 0].set_title("Segmentacion K-means")

    axes[1, 1].imshow(rgb)
    for role in ROLES:
        mask = masks[role]
        if np.any(mask):
            axes[1, 1].contour(
                mask,
                levels=[0.5],
                colors=[ROLE_COLORS[role]],
                linewidths=2,
            )
            ys, xs = np.nonzero(mask)
            axes[1, 1].text(
                float(np.mean(xs)),
                float(np.mean(ys)),
                role.upper(),
                color=ROLE_COLORS[role],
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
            )
    axes[1, 1].set_title(f"ROIs interiores | {status} | confianza={confidence:.3f}")

    for axis in axes.flat:
        axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(reflectance, color="#8b5a2b", linewidth=1.2)
    axis.axhline(0, color="black", linewidth=0.7)
    axis.axhline(1, color="gray", linewidth=0.7, linestyle="--")
    axis.set_xlabel("Indice de banda")
    axis.set_ylabel("Reflectancia relativa")
    axis.set_title("Firma de reflectancia SOIL")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path.with_name(path.stem + "_reflectancia.png"), dpi=150)
    plt.close(figure)


def evaluate_masks_and_signatures(
    cube: np.ndarray,
    broadband: np.ndarray,
    assignment: dict[str, Candidate],
    role_scores: dict[str, float],
    config: Config,
) -> dict[str, object]:
    masks = {
        "soil": make_circular_mask(assignment["soil"], config, "soil"),
        "white": make_circular_mask(assignment["white"], config, "white"),
    }
    role_scores = dict(role_scores)
    masks["dark"], dark_rectangle, role_scores["dark"] = select_dark_corner_rectangle(
        broadband, config
    )

    signatures = {
        role: signature_from_mask(cube, masks[role], config.reduction)
        for role in ROLES
    }
    roi_sizes = {role: int(np.count_nonzero(masks[role])) for role in ROLES}
    reflectance, valid = calculate_reflectance(
        signatures["soil"], signatures["white"], signatures["dark"]
    )

    invalid_fraction = float(1 - np.count_nonzero(valid) / valid.size)
    finite_reflectance = reflectance[np.isfinite(reflectance)]
    outside_fraction = (
        float(np.mean((finite_reflectance < -0.15) | (finite_reflectance > 1.5)))
        if finite_reflectance.size
        else 1.0
    )
    confidence = float(min(role_scores.values()))
    reasons: list[str] = []
    if confidence < config.min_confidence:
        reasons.append("confianza_baja")
    if invalid_fraction > config.max_invalid_reflectance_fraction:
        reasons.append("muchas_bandas_invalidas")
    if outside_fraction > 0.35:
        reasons.append("reflectancia_fuera_de_rango")
    roi_balance = min(roi_sizes.values()) / max(max(roi_sizes.values()), 1)
    if roi_balance < 0.02:
        reasons.append("roi_muy_pequena_frente_a_otras")
    status = "review" if reasons else "ok"

    return {
        "masks": masks,
        "signatures": signatures,
        "reflectance": reflectance,
        "valid": valid,
        "roi_sizes": roi_sizes,
        "invalid_fraction": invalid_fraction,
        "outside_fraction": outside_fraction,
        "confidence": confidence,
        "status": status,
        "reason": ";".join(reasons),
        "dark_rectangle": dark_rectangle,
        "role_scores": role_scores,
    }


def save_k_result(
    out_dir: Path,
    cube: np.ndarray,
    features: np.ndarray,
    broadband: np.ndarray,
    labels: np.ndarray,
    assignment: dict[str, Candidate],
    role_scores: dict[str, float],
    quality: float,
    k: int,
    preview_range: tuple[int, int],
    subranges: list[tuple[int, int]],
    config: Config,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluated = evaluate_masks_and_signatures(
        cube, broadband, assignment, dict(role_scores), config
    )
    masks = evaluated["masks"]
    signatures = evaluated["signatures"]
    reflectance = evaluated["reflectance"]

    np.savez_compressed(
        out_dir / f"resultado_k_{k:02d}.npz",
        soil_signature=signatures["soil"],
        white_signature=signatures["white"],
        dark_signature=signatures["dark"],
        soil_reflectance=reflectance,
        soil_mask=masks["soil"],
        white_mask=masks["white"],
        dark_mask=masks["dark"],
        labels=labels,
        preview=broadband,
        preview_range=np.array(preview_range),
        preview_subranges=np.array(subranges),
        k=np.array([k], dtype=np.int32),
    )

    metadata = {
        "k": int(k),
        "quality": float(quality),
        "role_scores": evaluated["role_scores"],
        "selected_components": {
            role: {
                key: value
                for key, value in asdict(candidate).items()
                if key != "mask"
            }
            for role, candidate in assignment.items()
        },
        "dark_rectangle": evaluated["dark_rectangle"],
        "confidence": evaluated["confidence"],
        "status": evaluated["status"],
        "reason": evaluated["reason"],
        "roi_sizes": evaluated["roi_sizes"],
        "invalid_reflectance_fraction": evaluated["invalid_fraction"],
        "reflectance_outside_fraction": evaluated["outside_fraction"],
        "preview_range_start_inclusive_stop_exclusive": list(preview_range),
        "preview_subranges": subranges,
        "soil_roi_radius_pixels": config.soil_roi_radius_pixels,
    }
    (out_dir / f"metadata_k_{k:02d}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    save_diagnostic(
        out_dir / f"diagnostico_k_{k:02d}.png",
        features,
        broadband,
        labels,
        masks,
        reflectance,
        float(evaluated["confidence"]),
        str(evaluated["status"]),
    )
    return {
        "k": int(k),
        "quality": float(quality),
        "status": evaluated["status"],
        "reason": evaluated["reason"],
        "confidence": evaluated["confidence"],
        "invalid_fraction": evaluated["invalid_fraction"],
        "outside_fraction": evaluated["outside_fraction"],
        "reflectance": reflectance,
        "metadata": metadata,
        "out_dir": str(out_dir),
    }


def save_all_k_results(
    cube_out: Path,
    cube: np.ndarray,
    features: np.ndarray,
    broadband: np.ndarray,
    preview_range: tuple[int, int],
    subranges: list[tuple[int, int]],
    config: Config,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    per_k_dir = cube_out / "por_k"

    for k in config.k_values:
        out_dir = per_k_dir / f"k_{k:02d}"
        try:
            labels = segment_features(features, config, n_clusters=k)
            candidates = extract_candidates(labels, broadband, config)
            assignment, role_scores = assign_foreground_roles(candidates, config)
            quality = segmentation_quality(assignment, role_scores)
            result = save_k_result(
                out_dir,
                cube,
                features,
                broadband,
                labels,
                assignment,
                role_scores,
                quality,
                int(k),
                preview_range,
                subranges,
                config,
            )
            results.append(result)
        except Exception as exc:
            out_dir.mkdir(parents=True, exist_ok=True)
            error_metadata = {"k": int(k), "status": "error", "error": str(exc)}
            (out_dir / f"metadata_k_{k:02d}.json").write_text(
                json.dumps(error_metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            results.append(error_metadata)

    save_reflectance_by_k_plot(cube_out / "firmas_reflectancia_por_k.png", results)
    return results


def save_reflectance_by_k_plot(path: Path, k_results: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = False
    for result in k_results:
        if result.get("status") == "error" or "reflectance" not in result:
            continue
        reflectance = np.asarray(result["reflectance"], dtype=np.float32)
        label = f"K={result['k']} | {result.get('status')} | invalid={float(result.get('invalid_fraction', 0)):.1%}"
        ax.plot(reflectance, linewidth=1.2, label=label)
        plotted = True

    ax.axhline(0, color="black", linewidth=0.7)
    ax.axhline(1, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Indice de banda")
    ax.set_ylabel("Reflectancia relativa")
    ax.set_title("Firmas de reflectancia SOIL por K")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def process_cube(
    path: Path,
    input_dir: Path,
    output_dir: Path,
    config: Config,
) -> CubeResult:
    cube_id = cube_identifier(path, input_dir)
    cube_out = output_dir / "cubos" / cube_id
    cube_out.mkdir(parents=True, exist_ok=True)

    cube = load_cube(path, config.layout)
    features, broadband, preview_range, subranges = build_multiband_features(cube, config)
    labels, candidates, assignment, role_scores, segmentation_diagnostics = choose_best_segmentation(
        features, broadband, config
    )
    evaluated = evaluate_masks_and_signatures(
        cube, broadband, assignment, dict(role_scores), config
    )
    role_scores = evaluated["role_scores"]
    masks = evaluated["masks"]
    signatures = evaluated["signatures"]
    reflectance = evaluated["reflectance"]
    roi_sizes = evaluated["roi_sizes"]
    invalid_fraction = float(evaluated["invalid_fraction"])
    outside_fraction = float(evaluated["outside_fraction"])
    confidence = float(evaluated["confidence"])
    status = str(evaluated["status"])
    reason = str(evaluated["reason"])
    dark_rectangle = evaluated["dark_rectangle"]

    per_k_results = save_all_k_results(
        cube_out,
        cube,
        features,
        broadband,
        preview_range,
        subranges,
        config,
    )

    np.savez_compressed(
        cube_out / "resultado.npz",
        soil_signature=signatures["soil"],
        white_signature=signatures["white"],
        dark_signature=signatures["dark"],
        soil_reflectance=reflectance,
        soil_mask=masks["soil"],
        white_mask=masks["white"],
        dark_mask=masks["dark"],
        preview=broadband,
        preview_range=np.array(preview_range),
        preview_subranges=np.array(subranges),
    )

    metadata = {
        "cube_id": cube_id,
        "input_path": str(path),
        "cube_shape_y_x_lambda": [int(value) for value in cube.shape],
        "preview_range_start_inclusive_stop_exclusive": list(preview_range),
        "preview_subranges": subranges,
        "segmentation": segmentation_diagnostics,
        "per_k_results": [
            {
                key: value
                for key, value in result.items()
                if key not in {"reflectance", "metadata"}
            }
            for result in per_k_results
        ],
        "role_scores": role_scores,
        "selected_components": {
            role: {
                key: value
                for key, value in asdict(candidate).items()
                if key != "mask"
            }
            for role, candidate in assignment.items()
        },
        "dark_rectangle": dark_rectangle,
        "confidence": confidence,
        "status": status,
        "reason": reason,
        "invalid_reflectance_fraction": invalid_fraction,
        "reflectance_outside_fraction": outside_fraction,
        "config": asdict(config),
    }
    (cube_out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    save_diagnostic(
        cube_out / "diagnostico.png",
        features,
        broadband,
        labels,
        masks,
        reflectance,
        confidence,
        status,
    )

    return CubeResult(
        cube_id=cube_id,
        input_path=str(path),
        status=status,
        confidence=confidence,
        reason=reason,
        shape_y=int(cube.shape[0]),
        shape_x=int(cube.shape[1]),
        bands=int(cube.shape[2]),
        soil_pixels=roi_sizes["soil"],
        white_pixels=roi_sizes["white"],
        dark_pixels=roi_sizes["dark"],
        invalid_reflectance_fraction=invalid_fraction,
        reflectance_outside_fraction=outside_fraction,
    )


def write_summary(path: Path, results: Iterable[CubeResult]) -> None:
    rows = [asdict(result) for result in results]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_signature_matrix(output_dir: Path, results: list[CubeResult]) -> None:
    valid_results = [result for result in results if result.status in {"ok", "review"}]
    if not valid_results:
        return

    signatures: list[np.ndarray] = []
    ids: list[str] = []
    expected_bands: int | None = None
    for result in valid_results:
        result_path = output_dir / "cubos" / result.cube_id / "resultado.npz"
        with np.load(result_path) as data:
            signature = data["soil_reflectance"].astype(np.float32)
        if expected_bands is None:
            expected_bands = signature.size
        if signature.size != expected_bands:
            continue
        signatures.append(signature)
        ids.append(result.cube_id)

    if not signatures:
        return
    matrix = np.stack(signatures)
    np.save(output_dir / "todas_las_reflectancias.npy", matrix)
    (output_dir / "ids_reflectancias.txt").write_text("\n".join(ids), encoding="utf-8")


def discover_cubes(input_dir: Path, pattern: str) -> list[Path]:
    files = sorted(path for path in input_dir.rglob(pattern) if path.is_file())
    return files


def create_synthetic_cube(path: Path, layout: str) -> None:
    rng = np.random.default_rng(12)
    height, width, bands = 180, 260, 120
    wavelengths = np.linspace(0, 1, bands, dtype=np.float32)
    dark = 0.05 + 0.01 * wavelengths
    white = 0.82 + 0.08 * wavelengths
    soil = 0.25 + 0.28 * wavelengths - 0.05 * np.exp(-((wavelengths - 0.62) / 0.07) ** 2)
    background = 0.12 + 0.03 * wavelengths

    cube = np.broadcast_to(background, (height, width, bands)).copy()
    yy, xx = np.indices((height, width))
    masks = {
        "soil": ((yy - 95) / 42) ** 2 + ((xx - 130) / 56) ** 2 <= 1,
        "white": ((yy - 48) / 25) ** 2 + ((xx - 55) / 30) ** 2 <= 1,
        "dark": ((yy - 48) / 24) ** 2 + ((xx - 210) / 28) ** 2 <= 1,
    }
    for role, signature in (("soil", soil), ("white", white), ("dark", dark)):
        cube[masks[role], :] = signature
    cube += rng.normal(0, 0.012, cube.shape).astype(np.float32)
    cube = np.clip(cube, 0, None).astype(np.float32)

    if layout == "y_lambda_x":
        cube = np.moveaxis(cube, 2, 1)
    np.save(path, cube)


def run_self_test(output_dir: Path) -> int:
    test_dir = output_dir / "_entrada_sintetica"
    test_dir.mkdir(parents=True, exist_ok=True)
    cube_path = test_dir / "cube_sintetico.npy"
    create_synthetic_cube(cube_path, "y_lambda_x")
    config = Config(
        layout="y_lambda_x",
        preview_start=10,
        preview_stop=110,
        min_area_fraction=0.002,
        soil_roi_radius_pixels=32.0,
        expected_soil=(0.50, 0.53),
        expected_white=(0.21, 0.27),
        expected_dark=(0.81, 0.27),
    )
    result = process_cube(cube_path, test_dir, output_dir, config)
    write_summary(output_dir / "resumen.csv", [result])
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return 0 if result.status == "ok" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extrae automaticamente firmas SOIL/WHITE/DARK desde cubos .npy."
    )
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resultados_firmas_automaticas"),
    )
    parser.add_argument("--pattern", default="cube_*.npy")
    parser.add_argument(
        "--layout",
        choices=("y_lambda_x", "y_x_lambda"),
        default="y_lambda_x",
    )
    parser.add_argument("--preview-start", type=int, default=390)
    parser.add_argument("--preview-stop", type=int, default=679)
    parser.add_argument(
        "--clusters",
        type=int,
        help="Fuerza un unico K. Si no se usa, se prueban los valores de --k-values.",
    )
    parser.add_argument(
        "--k-values",
        type=parse_int_list,
        default=parse_int_list("4,5"),
        help="Valores de K a probar, separados por coma. Por defecto: 4,5.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reduction", choices=("mean", "median"), default="median")
    parser.add_argument(
        "--soil-radius-pixels",
        type=float,
        default=90.0,
        help="Radio fijo, en pixeles, para la ROI circular de SOIL. Por defecto: 90.",
    )
    parser.add_argument("--circle-radius-fraction", type=float, default=0.22)
    parser.add_argument("--dark-search-fraction", type=float, default=0.30)
    parser.add_argument("--dark-height-fraction", type=float, default=0.08)
    parser.add_argument("--dark-width-fraction", type=float, default=0.08)
    parser.add_argument(
        "--expected-soil",
        type=parse_point,
        default=parse_point("0.43,0.55"),
        help=(
            "Centro esperado de SOIL como x,y normalizado. "
            "Por defecto 0.43,0.55, cerca del circulo grande izquierdo."
        ),
    )
    parser.add_argument("--expected-white", type=parse_point)
    parser.add_argument("--expected-dark", type=parse_point)
    parser.add_argument(
        "--soil-max-position-distance",
        type=float,
        default=0.28,
        help="Distancia normalizada maxima para favorecer SOIL alrededor del centro esperado.",
    )
    parser.add_argument(
        "--min-soil-position-score",
        type=float,
        default=0.18,
        help="Puntaje minimo de posicion para aceptar un candidato SOIL.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.self_test:
        return run_self_test(output_dir)
    if args.input_dir is None:
        print("Falta --input-dir. Use --self-test para ejecutar la prueba sintetica.", file=sys.stderr)
        return 2

    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        print(f"No existe la carpeta de entrada: {input_dir}", file=sys.stderr)
        return 2

    config = Config(
        layout=args.layout,
        preview_start=args.preview_start,
        preview_stop=args.preview_stop,
        n_clusters=args.clusters if args.clusters is not None else args.k_values[0],
        k_values=(args.clusters,) if args.clusters is not None else args.k_values,
        reduction=args.reduction,
        soil_roi_radius_pixels=args.soil_radius_pixels,
        circular_roi_radius_fraction=args.circle_radius_fraction,
        dark_corner_search_fraction=args.dark_search_fraction,
        dark_rectangle_height_fraction=args.dark_height_fraction,
        dark_rectangle_width_fraction=args.dark_width_fraction,
        expected_soil=args.expected_soil,
        expected_white=args.expected_white,
        expected_dark=args.expected_dark,
        soil_max_position_distance=args.soil_max_position_distance,
        min_soil_position_score=args.min_soil_position_score,
    )
    (output_dir / "config_usada.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    files = discover_cubes(input_dir, args.pattern)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        print(f"No se encontraron archivos '{args.pattern}' dentro de {input_dir}.")
        return 2

    print(f"Cubos encontrados: {len(files)}")
    results: list[CubeResult] = []
    failures: list[dict[str, str]] = []

    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path}")
        try:
            result = process_cube(path, input_dir, output_dir, config)
            results.append(result)
            print(
                f"  -> {result.status}, confianza={result.confidence:.3f}, "
                f"invalidas={result.invalid_reflectance_fraction:.1%}"
            )
        except Exception as exc:
            print(f"  -> ERROR: {exc}")
            failures.append(
                {
                    "input_path": str(path),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        write_summary(output_dir / "resumen.csv", results)
        (output_dir / "errores.json").write_text(
            json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    write_signature_matrix(output_dir, results)
    ok_count = sum(result.status == "ok" for result in results)
    review_count = sum(result.status == "review" for result in results)
    print(
        f"Finalizado: ok={ok_count}, revisar={review_count}, errores={len(failures)}. "
        f"Salida: {output_dir}"
    )
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
