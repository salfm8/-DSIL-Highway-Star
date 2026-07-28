"""Smart DoF V13 — current-frame metric-depth target locking.

Unlike the experimental V12 scene-reprojection path, V13 never carries a
synthetic focal plane forward from the first frame.  For every valid frame:

1. Track the selected target and obtain its current SAM sampling mask.
2. Read the current frame's metric depth inside the target mask.
3. Update the thin-lens focus distance S(t) from that observation.
4. Compute CoC for every pixel using only current Z_t(x, y) and S(t).

The target mask selects depth samples only. It never protects target pixels or
otherwise changes the per-pixel CoC result.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from cinematic_v7_natural import parse_point
from cinematic_v8_semantic_focus import parse_bbox, parse_points
from cinematic_v12_metric_coc import CinematicDoFv12


class CinematicDoFv13(CinematicDoFv12):
    """Parallel-plane DoF whose focal distance is re-observed every frame."""

    def __init__(
        self,
        *args,
        focus_current_weight: float = 0.85,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 < focus_current_weight <= 1.0:
            raise ValueError("--focus-current-weight must be in (0, 1]")
        self.focus_current_weight = float(focus_current_weight)
        self.pipeline_name = (
            "Smart DoF v13.0 — Current-Frame Dynamic Target Focus"
        )
        print("  V13 dynamic target lock enabled")
        print("  S(t): current tracked-target metric depth on every frame")
        print(
            "  Target-depth temporal weight: "
            f"{self.focus_current_weight:.2f}"
        )
        print("  No reference-frame depth or synthetic focal-plane reprojection")

    def _update_dynamic_focus(self, observed_focus_m: float) -> float:
        """Update S(t) from the current frame rather than an initial reference."""
        observed = float(
            np.clip(observed_focus_m, 0.1, self.model_max_depth_m)
        )
        if self.focus_depth is None:
            self.focus_depth = observed
        else:
            weight = self.focus_current_weight
            self.focus_depth = (
                self.focus_depth * (1.0 - weight) + observed * weight
            )
        return float(self.focus_depth)

    def process_video(
        self,
        input_path: str,
        output_path: str,
        depth_output_path: str | None = None,
        coc_output_path: str | None = None,
        focus_log_path: str | None = None,
    ) -> None:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open input video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not output.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot create output video: {output_path}")

        def optional_writer(
            path: str | None,
            label: str,
        ) -> cv2.VideoWriter | None:
            if not path:
                return None
            writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create {label} video: {path}")
            return writer

        depth_output = optional_writer(depth_output_path, "depth")
        coc_output = optional_writer(coc_output_path, "CoC")
        focus_rows: list[tuple[int, str, float | None, float | None]] = []

        frame_index = 0
        started_at = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # This is the current frame's metric depth. The inherited temporal
            # stabilizer only flow-warps the immediately preceding estimate; it
            # does not replace it with the first frame or a synthetic plane.
            current_depth_m = self._get_depth_map(rgb, width, height)
            effective_coc = np.zeros((height, width), dtype=np.float32)
            observed_focus_m: float | None = None

            if not self.tracking_lost:
                tracking_ok = self._track_point(gray)
                if not tracking_ok or not self._point_in_bounds(
                    self.current_point,
                    width,
                    height,
                ):
                    self._begin_focus_pull(fps)
                else:
                    refresh_sam = (
                        self.last_sam_mask is None
                        or self.last_sam_point is None
                        or (frame_index - 1) % self.sam_interval == 0
                    )
                    if refresh_sam:
                        raw_mask = self._get_sam_mask(
                            frame,
                            self.current_point,
                        )
                        self.last_sam_mask = raw_mask.copy()
                        self.last_sam_point = self.current_point.copy()
                    else:
                        delta = (
                            self.current_point.reshape(-1, 2)[0]
                            - self.last_sam_point.reshape(-1, 2)[0]
                        )
                        raw_mask = self._warp_mask(
                            self.last_sam_mask,
                            delta,
                        )

                    stable_mask = self._stabilize_mask(
                        raw_mask,
                        self.current_point,
                    )
                    # The only SAM-dependent operation in the renderer:
                    # selecting current-frame metric-depth samples for S(t).
                    # Use the newly generated/stabilized current SAM mask
                    # directly. A first-frame mask accumulated through scale
                    # flow gradually includes non-target background pixels.
                    observed_focus_m = self._subject_depth(
                        current_depth_m,
                        stable_mask,
                    )
                    if observed_focus_m is None:
                        self._begin_focus_pull(fps)
                    else:
                        focus_m = self._update_dynamic_focus(
                            observed_focus_m
                        )
                        effective_coc = self._compute_blur_map(
                            current_depth_m,
                            focus_m,
                        )
                        rendered = self._render_variable_blur(
                            frame,
                            effective_coc,
                        )
                        focus_rows.append(
                            (
                                frame_index,
                                "tracking",
                                observed_focus_m,
                                focus_m,
                            )
                        )
                        if depth_output is not None:
                            depth_output.write(
                                self._depth_visualization(current_depth_m)
                            )
                        output.write(rendered)
                        if coc_output is not None:
                            coc_output.write(
                                self._coc_visualization(effective_coc)
                            )
                        sys.stdout.write(
                            f"\r  🎥 [{frame_index}/{total_frames}] "
                            f"{frame_index * 100 // max(total_frames, 1)}%"
                            f" | Current target {observed_focus_m:.2f}m"
                            f" -> S(t) {focus_m:.2f}m"
                        )
                        sys.stdout.flush()
                        continue

            if self.focus_depth is None:
                rendered = frame
                state = "no-focus"
            else:
                self.transition_frame += 1
                progress = min(
                    1.0,
                    self.transition_frame / self.transition_frames,
                )
                strength = 0.5 * (1.0 + math.cos(math.pi * progress))
                effective_coc = (
                    self._compute_blur_map(
                        current_depth_m,
                        self.focus_depth,
                    )
                    * strength
                )
                rendered = self._render_variable_blur(
                    frame,
                    effective_coc,
                )
                state = "deep-focus-pull"

            focus_rows.append(
                (
                    frame_index,
                    state,
                    observed_focus_m,
                    self.focus_depth,
                )
            )
            if depth_output is not None:
                depth_output.write(
                    self._depth_visualization(current_depth_m)
                )
            output.write(rendered)
            if coc_output is not None:
                coc_output.write(
                    self._coc_visualization(effective_coc)
                )
            sys.stdout.write(
                f"\r  🎥 [{frame_index}/{total_frames}] "
                f"{frame_index * 100 // max(total_frames, 1)}%"
                " | Dynamic-target deep-focus pull"
            )
            sys.stdout.flush()

        cap.release()
        output.release()
        if depth_output is not None:
            depth_output.release()
        if coc_output is not None:
            coc_output.release()

        if focus_log_path:
            with open(focus_log_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(
                    (
                        "frame",
                        "state",
                        "observed_target_depth_m",
                        "effective_focus_distance_m",
                    )
                )
                writer.writerows(focus_rows)

        elapsed = time.time() - started_at
        print(f"\n\n  ✅ {self.pipeline_name} complete in {elapsed:.1f}s")
        print(f"  Output: {output_path}")
        if depth_output_path:
            print(f"  Current metric depth: {depth_output_path}")
        if coc_output_path:
            print(f"  Effective CoC: {coc_output_path}")
        if focus_log_path:
            print(f"  Focus-distance log: {focus_log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Smart DoF V13 — current-frame metric-depth target locking"
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
    parser.add_argument(
        "--depth-mode",
        choices=("metric", "inverse-calibrated"),
        default="metric",
    )
    parser.add_argument("--metric-model", type=Path, default=None)
    parser.add_argument("--near-anchor-m", type=float, default=1.5)
    parser.add_argument("--far-anchor-m", type=float, default=80.0)
    parser.add_argument("--focal-length-mm", type=float, default=135.0)
    parser.add_argument("--f-number", type=float, default=2.0)
    parser.add_argument("--sensor-width-mm", type=float, default=36.0)
    parser.add_argument("--coc-gain", type=float, default=4.0)
    parser.add_argument("--max-blur", type=float, default=28.0)
    parser.add_argument("--depth-current-weight", type=float, default=0.78)
    parser.add_argument("--focus-current-weight", type=float, default=0.85)
    parser.add_argument("--transition-sec", type=float, default=1.5)
    parser.add_argument("--sam-interval", type=int, default=2)
    parser.add_argument("--sam-model", type=Path, default=None)
    args = parser.parse_args()

    pipeline = CinematicDoFv13(
        init_point=args.point,
        focus_bbox=args.bbox,
        positive_points=parse_points(args.positive_points),
        negative_points=parse_points(args.negative_points),
        max_blur=args.max_blur,
        transition_sec=args.transition_sec,
        sam_interval=args.sam_interval,
        model_path=args.sam_model,
        depth_mode=args.depth_mode,
        metric_model_path=args.metric_model,
        near_anchor_m=args.near_anchor_m,
        far_anchor_m=args.far_anchor_m,
        focal_length_mm=args.focal_length_mm,
        f_number=args.f_number,
        sensor_width_mm=args.sensor_width_mm,
        coc_gain=args.coc_gain,
        depth_current_weight=args.depth_current_weight,
        focus_current_weight=args.focus_current_weight,
    )
    pipeline.process_video(
        args.input,
        args.output,
        args.depth_output,
        args.coc_output,
        args.focus_log,
    )


if __name__ == "__main__":
    main()
