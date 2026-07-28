"""Smart DoF V15 — clean per-frame metric depth and dynamic target focus.

V15 intentionally removes every form of depth propagation, warping,
reprojection, temporal depth blending, motion prior, and focal-distance
prediction.

For each frame:
  1. Run Depth Anything V2 Metric from scratch.
  2. Use optical flow only to track the target prompt coordinate.
  3. Run SAM on the current frame at the tracked coordinate.
  4. Set S(t) to the median clean metric depth inside the current SAM mask.
  5. Compute thin-lens CoC from the current clean depth and current S(t).
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


class CinematicDoFv15(CinematicDoFv12):
    """Current-frame-only metric DoF with no depth history."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.depth_mode != "metric":
            raise ValueError("V15 clean mode requires --depth-mode metric")
        self.pipeline_name = (
            "Smart DoF v15.0 — Clean Per-Frame Dynamic Target Focus"
        )
        print("  V15 clean current-frame pipeline enabled")
        print("  Depth: fresh Depth Anything V2 Metric inference every frame")
        print("  Optical flow: target point coordinates only")
        print("  S(t): current SAM-mask interior metric-depth median")
        print("  Disabled: depth warp/reprojection/history/motion prior/Kalman")

    def _get_clean_metric_depth(
        self,
        rgb: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Run the metric model without any temporal depth operation."""
        model_output = self._predict_model_depth(rgb, width, height)
        return np.clip(
            model_output,
            0.1,
            self.model_max_depth_m,
        ).astype(np.float32)

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
        focus_rows: list[tuple[int, str, float | None]] = []

        frame_index = 0
        started_at = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Deliberately independent inference: no call to V12's
            # _get_depth_map(), because that method temporally stabilizes depth.
            current_depth_m = self._get_clean_metric_depth(
                rgb,
                width,
                height,
            )
            effective_coc = np.zeros((height, width), dtype=np.float32)
            observed_focus_m: float | None = None

            if not self.tracking_lost:
                # Optical flow is used only here, to update current_point.
                tracking_ok = self._track_point(gray)
                if not tracking_ok or not self._point_in_bounds(
                    self.current_point,
                    width,
                    height,
                ):
                    self._begin_focus_pull(fps)
                else:
                    # Fresh SAM inference on every frame. No previous mask
                    # warp, temporal mask blend, or accumulated scale transform.
                    current_mask = self._get_sam_mask(
                        frame,
                        self.current_point,
                    )
                    observed_focus_m = self._subject_depth(
                        current_depth_m,
                        current_mask,
                    )
                    if observed_focus_m is None:
                        self._begin_focus_pull(fps)
                    else:
                        # No EMA or prediction: the current median is S(t).
                        self.focus_depth = float(observed_focus_m)
                        effective_coc = self._compute_blur_map(
                            current_depth_m,
                            self.focus_depth,
                        )
                        rendered = self._render_variable_blur(
                            frame,
                            effective_coc,
                        )
                        focus_rows.append(
                            (
                                frame_index,
                                "tracking",
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
                            f" | Clean current S(t) {self.focus_depth:.2f}m"
                        )
                        sys.stdout.flush()
                        continue

            # After target loss, keep running fresh metric inference. Hold the
            # final valid S only for the 1.5-second cosine deep-focus release.
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
                " | Clean deep-focus pull"
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
                        "current_sam_median_focus_distance_m",
                    )
                )
                writer.writerows(focus_rows)

        elapsed = time.time() - started_at
        print(f"\n\n  ✅ {self.pipeline_name} complete in {elapsed:.1f}s")
        print(f"  Output: {output_path}")
        if depth_output_path:
            print(f"  Clean per-frame metric depth: {depth_output_path}")
        if coc_output_path:
            print(f"  Clean dynamic CoC: {coc_output_path}")
        if focus_log_path:
            print(f"  Focus log: {focus_log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Smart DoF V15 — fresh per-frame metric depth and SAM-median S(t)"
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
        choices=("metric",),
        default="metric",
    )
    parser.add_argument("--metric-model", type=Path, default=None)
    parser.add_argument("--focal-length-mm", type=float, default=135.0)
    parser.add_argument("--f-number", type=float, default=2.0)
    parser.add_argument("--sensor-width-mm", type=float, default=36.0)
    parser.add_argument("--coc-gain", type=float, default=4.0)
    parser.add_argument("--max-blur", type=float, default=28.0)
    parser.add_argument("--transition-sec", type=float, default=1.5)
    parser.add_argument("--sam-model", type=Path, default=None)
    args = parser.parse_args()

    pipeline = CinematicDoFv15(
        init_point=args.point,
        focus_bbox=args.bbox,
        positive_points=parse_points(args.positive_points),
        negative_points=parse_points(args.negative_points),
        max_blur=args.max_blur,
        transition_sec=args.transition_sec,
        sam_interval=1,
        model_path=args.sam_model,
        depth_mode=args.depth_mode,
        metric_model_path=args.metric_model,
        focal_length_mm=args.focal_length_mm,
        f_number=args.f_number,
        sensor_width_mm=args.sensor_width_mm,
        coc_gain=args.coc_gain,
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
