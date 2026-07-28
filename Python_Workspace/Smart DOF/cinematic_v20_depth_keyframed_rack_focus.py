"""Smart DoF V20 — depth-keyframed cinematic rack focus.

V20 replaces V19's uninterrupted 60m-to-3m linear sweep with an editorial
focus curve measured from representative scene regions:

  * 60 m: initial Lotte Mart / far subject
  * 32 m: mid-road and forward-vehicle region
  * 14 m: near-road region
  *  3 m: bonnet

Each focal distance is held long enough to be readable.  Transitions use a
quintic smootherstep curve, so focus-ring velocity and acceleration both reach
zero at every keyframe.  A restrained 85mm f/2.8 virtual lens replaces the
extreme V18/V19 settings.

Depth calibration remains first-frame-only and fixed for the entire video.
There is no object tracking, repeated SAM inference, or target-locked focus.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from cinematic_v7_natural import parse_point
from cinematic_v8_semantic_focus import parse_bbox, parse_points
from cinematic_v19_rack_focus import CinematicDoFv19


def parse_focus_keyframes(
    value: str,
) -> tuple[tuple[float, float], ...]:
    """Parse ``progress:distance`` pairs separated by semicolons."""
    pairs: list[tuple[float, float]] = []
    for item in value.split(";"):
        progress_text, distance_text = item.strip().split(":", 1)
        progress = float(progress_text)
        distance = float(distance_text)
        pairs.append((progress, distance))
    if len(pairs) < 2:
        raise argparse.ArgumentTypeError("At least two focus keyframes required")
    if abs(pairs[0][0]) > 1e-9 or abs(pairs[-1][0] - 1.0) > 1e-9:
        raise argparse.ArgumentTypeError(
            "Focus keyframes must start at 0 and end at 1"
        )
    if any(distance <= 0.0 for _, distance in pairs):
        raise argparse.ArgumentTypeError("Focus distances must be positive")
    if any(
        pairs[index + 1][0] <= pairs[index][0]
        for index in range(len(pairs) - 1)
    ):
        raise argparse.ArgumentTypeError(
            "Focus-keyframe progress values must strictly increase"
        )
    return tuple(pairs)


DEFAULT_FOCUS_KEYFRAMES = parse_focus_keyframes(
    "0.00:60;0.18:60;0.38:32;0.55:32;"
    "0.73:14;0.86:14;0.96:3;1.00:3"
)


class CinematicDoFv20(CinematicDoFv19):
    """First-frame calibrated depth with held, eased editorial focus keys."""

    def __init__(
        self,
        *args,
        focus_keyframes: tuple[tuple[float, float], ...] = (
            DEFAULT_FOCUS_KEYFRAMES
        ),
        coc_current_weight: float = 0.85,
        coc_spatial_sigma: float = 1.2,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 < coc_current_weight <= 1.0:
            raise ValueError("--coc-current-weight must be in (0, 1]")
        if coc_spatial_sigma < 0.0:
            raise ValueError("--coc-spatial-sigma must be non-negative")

        self.focus_keyframes = focus_keyframes
        self.coc_current_weight = float(coc_current_weight)
        self.coc_spatial_sigma = float(coc_spatial_sigma)
        self.previous_effective_coc: np.ndarray | None = None
        self.pipeline_name = (
            "Smart DoF v20.0 — Depth-Keyframed Natural Rack Focus"
        )

        print("  V20 depth-keyframed focus direction enabled")
        print(
            "  Focus keys: "
            + " -> ".join(
                f"{progress:.2f}:{distance:.1f}m"
                for progress, distance in self.focus_keyframes
            )
        )
        print("  Transition curve: quintic smootherstep")
        print(
            "  CoC stabilization: "
            f"current {self.coc_current_weight:.2f}, "
            f"previous {1.0 - self.coc_current_weight:.2f}, "
            f"spatial sigma {self.coc_spatial_sigma:.1f}"
        )
        print("  Design priority: readable focus subjects over maximum blur")

    @staticmethod
    def _smootherstep(value: float) -> float:
        x = float(np.clip(value, 0.0, 1.0))
        return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)

    def _rack_focus_distance(
        self,
        frame_index: int,
        end_frame: int,
    ) -> tuple[float, float]:
        progress = self._rack_progress(frame_index, end_frame)
        for index in range(len(self.focus_keyframes) - 1):
            left_t, left_distance = self.focus_keyframes[index]
            right_t, right_distance = self.focus_keyframes[index + 1]
            if progress <= right_t or index == len(self.focus_keyframes) - 2:
                local_progress = (
                    (progress - left_t)
                    / max(right_t - left_t, 1e-9)
                )
                eased = self._smootherstep(local_progress)
                focus_distance = (
                    left_distance
                    + (right_distance - left_distance) * eased
                )
                return float(focus_distance), progress
        return float(self.focus_keyframes[-1][1]), progress

    def _compute_blur_map(
        self,
        depth_m: np.ndarray,
        focus_distance_m: float,
    ) -> np.ndarray:
        """Compute moderate CoC and suppress frame-to-frame pumping."""
        current_coc = super()._compute_blur_map(
            depth_m,
            focus_distance_m,
        )
        if self.coc_spatial_sigma > 0.0:
            current_coc = cv2.GaussianBlur(
                current_coc,
                (0, 0),
                sigmaX=self.coc_spatial_sigma,
                sigmaY=self.coc_spatial_sigma,
                borderType=cv2.BORDER_REPLICATE,
            )
        if self.previous_effective_coc is None:
            effective_coc = current_coc
        else:
            effective_coc = (
                self.coc_current_weight * current_coc
                + (1.0 - self.coc_current_weight)
                * self.previous_effective_coc
            )
        effective_coc = np.clip(
            effective_coc,
            0.0,
            self.max_blur,
        ).astype(np.float32)
        self.previous_effective_coc = effective_coc
        return effective_coc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Smart DoF V20 — natural depth-keyframed rack focus with "
            "readable far, mid-road, near-road, and bonnet holds"
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
    parser.add_argument("--focal-length-mm", type=float, default=85.0)
    parser.add_argument("--f-number", type=float, default=2.8)
    parser.add_argument("--sensor-width-mm", type=float, default=36.0)
    parser.add_argument("--coc-gain", type=float, default=1.5)
    parser.add_argument("--max-blur", type=float, default=24.0)
    parser.add_argument(
        "--focus-keyframes",
        type=parse_focus_keyframes,
        default=DEFAULT_FOCUS_KEYFRAMES,
    )
    parser.add_argument("--coc-current-weight", type=float, default=0.85)
    parser.add_argument("--coc-spatial-sigma", type=float, default=1.2)
    parser.add_argument("--rack-end-frame", type=int, default=None)
    parser.add_argument("--scene-scale", type=float, default=None)
    parser.add_argument("--near-roi-y-offset", type=int, default=55)
    parser.add_argument("--near-roi-width", type=int, default=160)
    parser.add_argument("--near-roi-height", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--sam-model", type=Path, default=None)
    args = parser.parse_args()

    pipeline = CinematicDoFv20(
        init_point=args.point,
        focus_bbox=args.bbox,
        positive_points=parse_points(args.positive_points),
        negative_points=parse_points(args.negative_points),
        max_blur=args.max_blur,
        transition_sec=1.5,
        sam_interval=1,
        model_path=args.sam_model,
        depth_mode="metric",
        metric_model_path=args.metric_model,
        focal_length_mm=args.focal_length_mm,
        f_number=args.f_number,
        sensor_width_mm=args.sensor_width_mm,
        coc_gain=args.coc_gain,
        rack_start_distance_m=args.focus_keyframes[0][1],
        rack_end_distance_m=args.focus_keyframes[-1][1],
        rack_end_frame=args.rack_end_frame,
        scene_scale=args.scene_scale,
        near_roi_y_offset=args.near_roi_y_offset,
        near_roi_width=args.near_roi_width,
        near_roi_height=args.near_roi_height,
        focus_keyframes=args.focus_keyframes,
        coc_current_weight=args.coc_current_weight,
        coc_spatial_sigma=args.coc_spatial_sigma,
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
