"""Smart DoF V21 — optical Natural DoF with occlusion-aware layers.

V21 is the new parallel-focal-plane baseline:

  * The Lotte Mart mask is used only to measure and stabilize target distance.
  * Every final pixel CoC is computed only from metric depth Z and S(t).
  * Pixels at the same depth receive the same sharpness, regardless of object.
  * Foreground, focal, and background depth layers are rendered separately.
  * Layers are composited far-to-near with premultiplied alpha.

The layered compositor prevents the symmetric colour leakage of a whole-frame
Gaussian pyramid.  A sharp focal layer covers blurred background spill, while
blurred foreground is allowed to spread over and occlude the focal subject as
a real out-of-focus foreground object would.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from cinematic_v7_natural import parse_point
from cinematic_v8_semantic_focus import parse_bbox, parse_points
from cinematic_v17_target_kinematic_anchor import CinematicDoFv17


class CinematicDoFv21(CinematicDoFv17):
    """Target-referenced optical DoF with far-to-near layer compositing."""

    def __init__(
        self,
        *args,
        focus_core_radius_px: float = 2.50,
        layer_opacity_boost: float = 1.30,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if focus_core_radius_px <= 0.0:
            raise ValueError("--focus-core-radius-px must be positive")
        if layer_opacity_boost <= 0.0:
            raise ValueError("--layer-opacity-boost must be positive")

        self.focus_core_radius_px = float(focus_core_radius_px)
        self.layer_opacity_boost = float(layer_opacity_boost)
        self.current_layer_depth_m: np.ndarray | None = None
        self.current_layer_focus_m: float | None = None
        self.pipeline_name = (
            "Smart DoF v21.0 — Optical Natural DoF + Occlusion Layers"
        )

        print("  V21 occlusion-aware optical baseline enabled")
        print("  Layer order: far background -> focal layer -> near foreground")
        print("  Layer filtering: normalized premultiplied-alpha Gaussian")
        print(
            f"  Sharp focal core: CoC <= {self.focus_core_radius_px:.2f}px"
        )
        print(f"  Defocus coverage boost: {self.layer_opacity_boost:.2f}x")
        print("  Subject mask never protects final pixels")

    def _compute_blur_map(
        self,
        depth_m: np.ndarray,
        focus_distance_m: float,
    ) -> np.ndarray:
        """Store signed layer geometry while retaining physical thin-lens CoC."""
        self.current_layer_depth_m = depth_m.astype(np.float32, copy=True)
        self.current_layer_focus_m = float(focus_distance_m)
        return super()._compute_blur_map(depth_m, focus_distance_m)

    @staticmethod
    def _smoothstep_scalar_field(value: np.ndarray) -> np.ndarray:
        x = np.clip(value, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    @staticmethod
    def _normalized_blurred_layer(
        image_f: np.ndarray,
        mask: np.ndarray,
        radius_px: float,
        opacity_boost: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Blur colour and coverage without importing colours across layers."""
        alpha = mask.astype(np.float32)
        if radius_px <= 0.35:
            return image_f, alpha

        sigma = max(0.35, radius_px * 0.55)
        premultiplied = image_f * alpha[..., None]
        blurred_alpha = cv2.GaussianBlur(
            alpha,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REPLICATE,
        )
        blurred_premultiplied = cv2.GaussianBlur(
            premultiplied,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REPLICATE,
        )
        colour = blurred_premultiplied / np.maximum(
            blurred_alpha[..., None],
            1e-5,
        )
        coverage = np.clip(
            blurred_alpha * opacity_boost,
            0.0,
            1.0,
        )
        return colour, coverage

    @staticmethod
    def _composite_over(
        destination: np.ndarray,
        source_colour: np.ndarray,
        source_alpha: np.ndarray,
    ) -> np.ndarray:
        alpha = source_alpha[..., None]
        return source_colour * alpha + destination * (1.0 - alpha)

    def _layer_edges(self, max_radius: float) -> np.ndarray:
        """Use dense low-CoC layers and broader high-CoC layers."""
        canonical = np.asarray(
            (
                self.focus_core_radius_px,
                1.5,
                2.5,
                4.0,
                6.5,
                10.0,
                15.0,
                21.0,
                max(self.max_blur, 21.01),
            ),
            dtype=np.float32,
        )
        edges = canonical[canonical <= max(max_radius, self.focus_core_radius_px)]
        if edges.size == 0 or edges[0] > self.focus_core_radius_px:
            edges = np.insert(edges, 0, self.focus_core_radius_px)
        if edges[-1] < max_radius + 1e-3:
            edges = np.append(edges, max_radius + 1e-3)
        return np.unique(edges)

    def _render_variable_blur(
        self,
        image: np.ndarray,
        blur_map: np.ndarray,
    ) -> np.ndarray:
        """Render signed CoC layers far-to-near with correct occlusion order."""
        depth_m = self.current_layer_depth_m
        focus_m = self.current_layer_focus_m
        if depth_m is None or focus_m is None:
            return super()._render_variable_blur(image, blur_map)

        max_radius = float(np.max(blur_map))
        if max_radius <= 0.05:
            return image.copy()

        image_f = image.astype(np.float32)
        coc = np.clip(
            blur_map.astype(np.float32),
            0.0,
            self.max_blur,
        )
        depth = depth_m.astype(np.float32)
        background = depth > focus_m
        foreground = depth < focus_m
        edges = self._layer_edges(max_radius)

        # Original colour is a safe underlay for pixels not covered after
        # layer filtering.  Every actual defocus layer then replaces it.
        result = image_f.copy()

        def render_bin(
            signed_side: np.ndarray,
            low: float,
            high: float,
        ) -> None:
            nonlocal result
            mask = (
                signed_side
                & (coc > low)
                & (coc <= high + 1e-5)
            )
            if np.count_nonzero(mask) < 8:
                return
            values = coc[mask]
            radius = float(np.median(values))
            colour, alpha = self._normalized_blurred_layer(
                image_f,
                mask,
                radius,
                self.layer_opacity_boost,
            )
            result = self._composite_over(result, colour, alpha)

        # Background: deepest/highest-CoC layers first, approaching focus last.
        bin_pairs = list(zip(edges[:-1], edges[1:]))
        for low, high in reversed(bin_pairs):
            render_bin(background, float(low), float(high))

        # The focal layer is the conventional acceptable-CoC region used to
        # define photographic depth of field.  It depends only on CoC, never
        # on semantic membership: every pixel with the same depth receives
        # the same sharpness.  The feather avoids a hard synthetic boundary.
        feather_width = max(0.35, self.focus_core_radius_px * 0.65)
        focus_alpha = 1.0 - self._smoothstep_scalar_field(
            (coc - self.focus_core_radius_px)
            / feather_width
        )
        result = self._composite_over(
            result,
            image_f,
            focus_alpha.astype(np.float32),
        )

        # Foreground: layers nearest the focus plane first; the closest and
        # most defocused geometry is composited last and may physically cover
        # the sharp subject.
        for low, high in bin_pairs:
            render_bin(foreground, float(low), float(high))

        return np.clip(result, 0.0, 255.0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Smart DoF V21 — target-referenced optical Natural DoF with "
            "occlusion-aware foreground/focal/background layers"
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--depth-output", default=None)
    parser.add_argument("--coc-output", default=None)
    parser.add_argument("--focus-log", default=None)
    parser.add_argument("--point", required=True, type=parse_point)
    parser.add_argument("--bbox", required=True, type=parse_bbox)
    parser.add_argument("--positive-points", default=None)
    parser.add_argument("--negative-points", default=None)
    parser.add_argument("--metric-model", type=Path, default=None)
    parser.add_argument("--focal-length-mm", type=float, default=100.0)
    parser.add_argument("--f-number", type=float, default=2.0)
    parser.add_argument("--sensor-width-mm", type=float, default=36.0)
    parser.add_argument("--coc-gain", type=float, default=2.0)
    parser.add_argument("--max-blur", type=float, default=32.0)
    parser.add_argument("--transition-sec", type=float, default=1.5)
    parser.add_argument(
        "--kinematic-start-distance-m",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--kinematic-end-distance-m",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--kinematic-end-frame",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--focus-core-radius-px",
        type=float,
        default=2.50,
        help=(
            "acceptable image-space CoC radius treated as the sharp DoF core; "
            "applies equally to every pixel and is not a subject-mask override"
        ),
    )
    parser.add_argument("--layer-opacity-boost", type=float, default=1.30)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--sam-model", type=Path, default=None)
    args = parser.parse_args()

    pipeline = CinematicDoFv21(
        init_point=args.point,
        focus_bbox=args.bbox,
        positive_points=parse_points(args.positive_points),
        negative_points=parse_points(args.negative_points),
        max_blur=args.max_blur,
        transition_sec=args.transition_sec,
        sam_interval=1,
        model_path=args.sam_model,
        depth_mode="metric",
        metric_model_path=args.metric_model,
        focal_length_mm=args.focal_length_mm,
        f_number=args.f_number,
        sensor_width_mm=args.sensor_width_mm,
        coc_gain=args.coc_gain,
        kinematic_start_distance_m=args.kinematic_start_distance_m,
        kinematic_end_distance_m=args.kinematic_end_distance_m,
        kinematic_end_frame=args.kinematic_end_frame,
        focus_core_radius_px=args.focus_core_radius_px,
        layer_opacity_boost=args.layer_opacity_boost,
    )
    pipeline.process_video(
        args.input,
        args.output,
        args.depth_output,
        args.coc_output,
        args.focus_log,
        args.max_frames,
    )


if __name__ == "__main__":
    main()
