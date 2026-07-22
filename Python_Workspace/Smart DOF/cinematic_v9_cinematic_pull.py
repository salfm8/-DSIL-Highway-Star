"""Smart DoF v9 — continuous cinematic focus pull.

V9 keeps V8's region-prompted semantic focus, then replaces its piecewise blur
composition and abrupt tracking-loss handoff with continuous blur blending and
a mask-preserving cinematic focus pull.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from cinematic_v7_natural import parse_point
from cinematic_v8_semantic_focus import CinematicDoFv8, parse_bbox, parse_points


class CinematicDoFv9(CinematicDoFv8):
    """V8 semantic target selection with continuous, lens-like focus changes."""

    def __init__(self, *args, sam_interval: int = 2, **kwargs) -> None:
        kwargs["pipeline_name"] = "Smart DoF v9.0 — Continuous Cinematic Focus Pull"
        super().__init__(*args, **kwargs)
        self.sam_interval = max(1, sam_interval)
        self.last_subject_mask: np.ndarray | None = None
        self.last_mask_velocity = np.zeros(2, dtype=np.float32)
        self.previous_blur_map: np.ndarray | None = None
        self.previous_blur_point: np.ndarray | None = None
        self.last_sam_mask: np.ndarray | None = None
        self.last_sam_point: np.ndarray | None = None

        print("  Continuous blur blending and mask-preserving focus pull enabled")

    @staticmethod
    def _warp_mask(mask: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """Move a recent SAM mask by optical-flow motion between SAM refreshes."""
        height, width = mask.shape[:2]
        transform = np.float32([[1.0, 0.0, delta[0]], [0.0, 1.0, delta[1]]])
        return cv2.warpAffine(
            mask,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def _stabilize_mask(self, current_mask: np.ndarray, point: np.ndarray) -> np.ndarray:
        prior_point = None if self.prev_mask_point is None else self.prev_mask_point.copy()
        stable = super()._stabilize_mask(current_mask, point)
        if prior_point is not None:
            velocity = point.reshape(-1, 2)[0] - prior_point.reshape(-1, 2)[0]
            # One-frame optical flow may spike at an edge.  Limit it rather than
            # letting a bad step tear the transition mask across the frame.
            self.last_mask_velocity = np.clip(velocity, -12.0, 12.0).astype(np.float32)
        self.last_subject_mask = stable.copy()
        return stable

    def _transition_mask(self, height: int, width: int) -> np.ndarray | None:
        """Carry the final stable subject mask through the start of focus pull."""
        if self.last_subject_mask is None:
            return None
        dx, dy = self.last_mask_velocity * self.transition_frame
        transform = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
        return cv2.warpAffine(
            self.last_subject_mask,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def _compute_blur_map(
        self,
        depth: np.ndarray,
        focus_depth: float,
        subject_mask: np.ndarray | None,
    ) -> np.ndarray:
        """Use continuous blur unions instead of V8's hard max/cap boundaries."""
        depth_difference = np.abs(depth - focus_depth)
        dead_zone = 0.022
        full_blur_difference = 0.19
        depth_progress = self._smoothstep(
            (depth_difference - dead_zone) / (full_blur_difference - dead_zone)
        )
        depth_blur = self.max_blur * depth_progress

        if subject_mask is None:
            return cv2.GaussianBlur(depth_blur.astype(np.float32), (0, 0), 2.0)

        binary_mask = (subject_mask >= 0.5).astype(np.uint8)
        protection = self._subject_protection(subject_mask, feather_px=15.0)
        distance_outside = cv2.distanceTransform(1 - binary_mask, cv2.DIST_L2, 5)
        semantic_progress = self._smoothstep((distance_outside - 14.0) / 250.0)
        semantic_blur = self.max_blur * 0.50 * semantic_progress

        # Probabilistic union is a soft maximum: it preserves the stronger cue
        # while remaining differentiable across their crossing point.
        combined = self.max_blur * (
            1.0
            - (1.0 - depth_blur / self.max_blur)
            * (1.0 - semantic_blur / self.max_blur)
        )

        # Near foreground (bonnet/road for this shot) is restrained using a
        # continuous saturation curve, not a boolean branch and hard cap.
        foreground_weight = self._smoothstep((depth - focus_depth - dead_zone) / 0.13)
        foreground_cap = self.max_blur * 0.38
        saturated_foreground = foreground_cap * np.tanh(combined / foreground_cap)
        blur_map = combined * (1.0 - foreground_weight) + saturated_foreground * foreground_weight

        # Wide blur-map smoothing makes road, vehicle, and bridge planes blend
        # into each other like a changing focal plane.  Re-protect the interior
        # after smoothing so the selected building remains crisply focused.
        blur_map *= 1.0 - protection
        blur_map = cv2.GaussianBlur(blur_map.astype(np.float32), (0, 0), 2.6)
        blur_map[protection >= 0.96] = 0.0
        return np.clip(blur_map, 0.0, self.max_blur).astype(np.float32)

    def _stabilize_blur_map(self, blur_map: np.ndarray, point: np.ndarray | None) -> np.ndarray:
        """Light temporal smoothing of the blur field without visible ghosting."""
        if self.previous_blur_map is None or point is None or self.previous_blur_point is None:
            stable = blur_map
        else:
            delta = point.reshape(-1, 2)[0] - self.previous_blur_point.reshape(-1, 2)[0]
            transform = np.float32([[1.0, 0.0, delta[0]], [0.0, 1.0, delta[1]]])
            warped_prior = cv2.warpAffine(
                self.previous_blur_map,
                transform,
                (blur_map.shape[1], blur_map.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            stable = blur_map * 0.80 + warped_prior * 0.20
        self.previous_blur_map = stable.astype(np.float32)
        if point is not None:
            self.previous_blur_point = point.copy()
        return self.previous_blur_map

    def process_video(self, input_path: str, output_path: str) -> None:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open input video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not output.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot create output video: {output_path}")

        frame_index = 0
        started_at = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            depth = self._get_depth_map(rgb, width, height)

            if not self.tracking_lost:
                tracking_ok = self._track_point(gray)
                if not tracking_ok or not self._point_in_bounds(self.current_point, width, height):
                    self._begin_focus_pull(fps)
                else:
                    refresh_sam = (
                        self.last_sam_mask is None
                        or self.last_sam_point is None
                        or (frame_index - 1) % self.sam_interval == 0
                    )
                    if refresh_sam:
                        raw_mask = self._get_sam_mask(frame, self.current_point)
                        self.last_sam_mask = raw_mask.copy()
                        self.last_sam_point = self.current_point.copy()
                    else:
                        delta = (
                            self.current_point.reshape(-1, 2)[0]
                            - self.last_sam_point.reshape(-1, 2)[0]
                        )
                        raw_mask = self._warp_mask(self.last_sam_mask, delta)
                    stable_mask = self._stabilize_mask(raw_mask, self.current_point)
                    observed_depth = self._subject_depth(depth, stable_mask)
                    if observed_depth is None:
                        self._begin_focus_pull(fps)
                    else:
                        self.focus_depth = (
                            observed_depth
                            if self.focus_depth is None
                            else self.focus_depth * self.depth_ema + observed_depth * (1.0 - self.depth_ema)
                        )
                        blur_map = self._compute_blur_map(depth, self.focus_depth, stable_mask)
                        blur_map = self._stabilize_blur_map(blur_map, self.current_point)
                        output.write(self._render_variable_blur(frame, blur_map))
                        sys.stdout.write(
                            f"\r  🎥 [{frame_index}/{total_frames}] "
                            f"{frame_index * 100 // max(total_frames, 1)}% | Continuous subject focus"
                        )
                        sys.stdout.flush()
                        continue

            # Crucially, the first transition frame still carries the last
            # subject mask and blur structure.  The focus pull therefore starts
            # from the image the viewer was already seeing, not a new map.
            if self.focus_depth is None:
                rendered = frame
            else:
                self.transition_frame += 1
                progress = min(1.0, self.transition_frame / self.transition_frames)
                blur_strength = 0.5 * (1.0 + math.cos(math.pi * progress))
                carried_mask = self._transition_mask(height, width)
                blur_map = self._compute_blur_map(depth, self.focus_depth, carried_mask)
                blur_map = self._stabilize_blur_map(blur_map, None)
                rendered = self._render_variable_blur(frame, blur_map * blur_strength)

            output.write(rendered)
            sys.stdout.write(
                f"\r  🎥 [{frame_index}/{total_frames}] "
                f"{frame_index * 100 // max(total_frames, 1)}% | Continuous focus pull"
            )
            sys.stdout.flush()

        cap.release()
        output.release()
        print(f"\n\n  ✅ {self.pipeline_name} rendering complete in {time.time() - started_at:.1f}s")
        print(f"  Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart DoF v9 — continuous cinematic focus pull")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--point", required=True, type=parse_point)
    parser.add_argument("--bbox", required=True, type=parse_bbox)
    parser.add_argument("--positive-points", default=None)
    parser.add_argument("--negative-points", default=None)
    parser.add_argument("--max-blur", type=float, default=14.0)
    parser.add_argument("--transition-sec", type=float, default=1.5)
    parser.add_argument(
        "--sam-interval",
        type=int,
        default=2,
        help="Run SAM every N frames and motion-warp its mask between refreshes (default: 2)",
    )
    parser.add_argument("--sam-model", type=Path, default=None)
    args = parser.parse_args()

    pipeline = CinematicDoFv9(
        init_point=args.point,
        focus_bbox=args.bbox,
        positive_points=parse_points(args.positive_points),
        negative_points=parse_points(args.negative_points),
        max_blur=args.max_blur,
        transition_sec=args.transition_sec,
        sam_interval=args.sam_interval,
        model_path=args.sam_model,
    )
    pipeline.process_video(args.input, args.output)


if __name__ == "__main__":
    main()
