"""Smart DoF V10 — finer, perceptually progressive focus falloff."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from cinematic_v7_natural import parse_point
from cinematic_v8_semantic_focus import parse_bbox, parse_points
from cinematic_v9_cinematic_pull import CinematicDoFv9


class CinematicDoFv10(CinematicDoFv9):
    """V9 focus tracking with a broad, fine-grained blur progression."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pipeline_name = "Smart DoF v10.0 — Progressive Cinematic Focus"
        print("  Fine-grained progressive blur ramp enabled")

    def _compute_blur_map(
        self, depth: np.ndarray, focus_depth: float, subject_mask: np.ndarray | None
    ) -> np.ndarray:
        """Spread blur values across the full range rather than compressing the midtones."""
        depth_delta = np.abs(depth - focus_depth)
        depth_t = np.clip((depth_delta - 0.012) / 0.32, 0.0, 1.0)
        # A near-linear perceptual ramp: avoids V9's quick mid-range jump.
        depth_progress = 0.62 * depth_t + 0.38 * np.power(depth_t, 0.82)

        if subject_mask is None:
            return cv2.GaussianBlur(
                (self.max_blur * depth_progress).astype(np.float32), (0, 0), 2.8
            )

        binary_mask = (subject_mask >= 0.5).astype(np.uint8)
        protection = self._subject_protection(subject_mask, feather_px=24.0)
        distance = cv2.distanceTransform(1 - binary_mask, cv2.DIST_L2, 5)
        distance_t = np.clip((distance - 8.0) / 460.0, 0.0, 1.0)
        # Spatial distance rises deliberately slowly close to the subject,
        # then keeps adding small amounts of blur across the whole frame.
        spatial_progress = 0.70 * distance_t + 0.30 * np.power(distance_t, 1.45)

        # Gentle residual blend instead of a hard maximum or aggressive union.
        combined = depth_progress + (1.0 - depth_progress) * spatial_progress * 0.55
        blur_map = self.max_blur * combined

        foreground_weight = self._smoothstep((depth - focus_depth - 0.015) / 0.18)
        foreground_cap = self.max_blur * 0.46
        foreground_blur = foreground_cap * np.tanh(blur_map / foreground_cap)
        blur_map = blur_map * (1.0 - foreground_weight) + foreground_blur * foreground_weight

        blur_map *= 1.0 - protection
        blur_map = cv2.GaussianBlur(blur_map.astype(np.float32), (0, 0), 3.8)
        blur_map[protection >= 0.985] = 0.0
        return np.clip(blur_map, 0.0, self.max_blur).astype(np.float32)

    @staticmethod
    def _render_variable_blur(image: np.ndarray, blur_map: np.ndarray) -> np.ndarray:
        """Interpolate every one-pixel blur radius to avoid visible blur bands."""
        max_radius = int(math.ceil(float(np.max(blur_map))))
        if max_radius <= 0:
            return image.copy()
        radii = np.arange(0, max_radius + 2, dtype=np.float32)
        levels = [image.astype(np.float32)]
        for radius in radii[1:]:
            kernel = int(radius) * 2 + 1
            levels.append(cv2.GaussianBlur(image, (kernel, kernel), float(radius) * 0.50).astype(np.float32))
        indices = np.clip(np.searchsorted(radii, blur_map), 1, len(radii) - 1)
        result = np.empty_like(image, dtype=np.float32)
        for index in range(1, len(radii)):
            pixels = indices == index
            if not np.any(pixels):
                continue
            low, high = radii[index - 1], radii[index]
            weight = np.clip((blur_map[pixels] - low) / (high - low + 1e-6), 0.0, 1.0)
            result[pixels] = levels[index - 1][pixels] * (1.0 - weight[:, None]) + levels[index][pixels] * weight[:, None]
        return np.clip(result, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart DoF V10 — progressive focus falloff")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--point", required=True, type=parse_point)
    parser.add_argument("--bbox", required=True, type=parse_bbox)
    parser.add_argument("--positive-points", default=None)
    parser.add_argument("--negative-points", default=None)
    parser.add_argument("--max-blur", type=float, default=14.0)
    parser.add_argument("--transition-sec", type=float, default=1.5)
    parser.add_argument("--sam-interval", type=int, default=2)
    parser.add_argument("--sam-model", type=Path, default=None)
    args = parser.parse_args()
    pipeline = CinematicDoFv10(
        init_point=args.point, focus_bbox=args.bbox,
        positive_points=parse_points(args.positive_points), negative_points=parse_points(args.negative_points),
        max_blur=args.max_blur, transition_sec=args.transition_sec,
        sam_interval=args.sam_interval, model_path=args.sam_model,
    )
    pipeline.process_video(args.input, args.output)


if __name__ == "__main__":
    main()
