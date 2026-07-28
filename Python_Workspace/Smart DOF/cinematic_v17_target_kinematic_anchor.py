"""Smart DoF V17 — target-kinematic scale anchoring.

V17 keeps V15's clean current-frame pipeline:

  * Depth Anything V2 Metric runs independently on every frame.
  * Optical flow is used only to update the target prompt coordinate.
  * SAM is used only to measure the target's raw median depth.
  * No depth warp, reprojection, temporal depth blend, or mask-based
    sharpness is used.

The monocular model's unstable global scale is replaced by an explicit
kinematic focus schedule:

    S(t)     = lerp(S_start, S_end, progress(t))
    scale(t) = S(t) / median(raw_depth_t[target_mask])
    Z_t      = raw_depth_t * scale(t)

All final pixel blur radii are computed from Z_t and S(t) with the inherited
thin-lens Circle-of-Confusion equation.  The SAM mask never participates in
the final sharp/blur decision.
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
from cinematic_v15_clean_dynamic_focus import CinematicDoFv15


class CinematicDoFv17(CinematicDoFv15):
    """Clean metric DoF whose moving focus distance follows a target schedule."""

    def __init__(
        self,
        *args,
        kinematic_start_distance_m: float | None = None,
        kinematic_end_distance_m: float = 15.0,
        kinematic_end_frame: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if (
            kinematic_start_distance_m is not None
            and kinematic_start_distance_m <= 0.0
        ):
            raise ValueError("--kinematic-start-distance-m must be positive")
        if kinematic_end_distance_m <= 0.0:
            raise ValueError("--kinematic-end-distance-m must be positive")
        if kinematic_end_frame is not None and kinematic_end_frame < 2:
            raise ValueError("--kinematic-end-frame must be at least 2")

        self.requested_start_distance_m = (
            None
            if kinematic_start_distance_m is None
            else float(kinematic_start_distance_m)
        )
        self.kinematic_start_distance_m: float | None = None
        self.kinematic_end_distance_m = float(kinematic_end_distance_m)
        self.kinematic_end_frame = kinematic_end_frame
        self.last_target_scale: float | None = None
        self.pipeline_name = (
            "Smart DoF v17.0 — Target-Kinematic Anchored Focus Pull"
        )

        print("  V17 target-kinematic anchoring enabled")
        if self.requested_start_distance_m is None:
            print("  S_start: first valid SAM-mask raw-depth median")
        else:
            print(
                "  S_start: "
                f"{self.requested_start_distance_m:.2f}m (user override)"
            )
        print(f"  S_end: {self.kinematic_end_distance_m:.2f}m")
        if self.kinematic_end_frame is None:
            print("  Kinematic end frame: final input frame")
        else:
            print(
                "  Kinematic end frame: "
                f"{self.kinematic_end_frame} (target-loss timing)"
            )
        print("  Scale(t) = forced S(t) / current target raw median")
        print("  SAM usage: target scale measurement only")
        print("  Final sharpness: thin-lens CoC from scaled depth only")

    @staticmethod
    def _schedule_progress(frame_index: int, end_frame: int) -> float:
        if end_frame <= 1:
            return 1.0
        return float(
            np.clip(
                (frame_index - 1) / float(end_frame - 1),
                0.0,
                1.0,
            )
        )

    def _scheduled_focus_distance(
        self,
        frame_index: int,
        end_frame: int,
    ) -> tuple[float, float]:
        if self.kinematic_start_distance_m is None:
            raise RuntimeError("S_start has not been initialized")
        progress = self._schedule_progress(frame_index, end_frame)
        focus_distance_m = (
            self.kinematic_start_distance_m
            + (
                self.kinematic_end_distance_m
                - self.kinematic_start_distance_m
            )
            * progress
        )
        return float(focus_distance_m), progress

    @staticmethod
    def _apply_target_scale(
        raw_depth_m: np.ndarray,
        scheduled_focus_m: float,
        raw_target_median_m: float,
    ) -> tuple[np.ndarray, float]:
        scale_factor = scheduled_focus_m / max(raw_target_median_m, 1e-6)
        scaled_depth_m = np.maximum(
            raw_depth_m.astype(np.float32) * scale_factor,
            0.1,
        ).astype(np.float32)
        return scaled_depth_m, float(scale_factor)

    def process_video(
        self,
        input_path: str,
        output_path: str,
        depth_output_path: str | None = None,
        coc_output_path: str | None = None,
        focus_log_path: str | None = None,
        max_frames: int | None = None,
    ) -> None:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open input video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        schedule_end_frame = self.kinematic_end_frame or total_frames
        if schedule_end_frame > total_frames:
            print(
                "  ⚠️ Kinematic end frame exceeds video length; "
                "the scheduled S_end will not be reached"
            )
        if max_frames is not None and max_frames < 1:
            cap.release()
            raise ValueError("--max-frames must be positive")

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
        log_rows: list[
            tuple[
                int,
                str,
                float,
                float | None,
                float | None,
                float | None,
                float | None,
                float | None,
            ]
        ] = []

        frame_index = 0
        started_at = time.time()
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Independent current-frame inference: deliberately no V12
            # temporal depth method, warp, or reprojection.
            raw_depth_m = self._get_clean_metric_depth(rgb, width, height)
            scaled_depth_m = raw_depth_m
            effective_coc = np.zeros((height, width), dtype=np.float32)
            raw_target_median_m: float | None = None
            scheduled_focus_m: float | None = None
            schedule_progress = self._schedule_progress(
                frame_index,
                schedule_end_frame,
            )
            scale_factor: float | None = None
            target_alignment_error_m: float | None = None

            if not self.tracking_lost:
                # Optical flow is restricted to target-coordinate tracking.
                tracking_ok = self._track_point(gray)
                if not tracking_ok or not self._point_in_bounds(
                    self.current_point,
                    width,
                    height,
                ):
                    self._begin_focus_pull(fps)
                else:
                    current_mask = self._get_sam_mask(
                        frame,
                        self.current_point,
                    )
                    # SAM is restricted to this raw target-depth measurement.
                    raw_target_median_m = self._subject_depth(
                        raw_depth_m,
                        current_mask,
                    )
                    if raw_target_median_m is None:
                        self._begin_focus_pull(fps)
                    else:
                        if self.kinematic_start_distance_m is None:
                            self.kinematic_start_distance_m = (
                                self.requested_start_distance_m
                                if self.requested_start_distance_m is not None
                                else float(raw_target_median_m)
                            )
                        scheduled_focus_m, schedule_progress = (
                            self._scheduled_focus_distance(
                                frame_index,
                                schedule_end_frame,
                            )
                        )
                        scaled_depth_m, scale_factor = (
                            self._apply_target_scale(
                                raw_depth_m,
                                scheduled_focus_m,
                                float(raw_target_median_m),
                            )
                        )
                        self.focus_depth = scheduled_focus_m
                        self.last_target_scale = scale_factor

                        # This second median is diagnostic only.  It should
                        # numerically equal the scheduled focus distance.
                        scaled_target_median_m = self._subject_depth(
                            scaled_depth_m,
                            current_mask,
                        )
                        if scaled_target_median_m is not None:
                            target_alignment_error_m = (
                                float(scaled_target_median_m)
                                - scheduled_focus_m
                            )

                        # Pixel membership in current_mask is not used here.
                        effective_coc = self._compute_blur_map(
                            scaled_depth_m,
                            scheduled_focus_m,
                        )
                        rendered = self._render_variable_blur(
                            frame,
                            effective_coc,
                        )
                        state = "tracking"

                        log_rows.append(
                            (
                                frame_index,
                                state,
                                schedule_progress,
                                raw_target_median_m,
                                scheduled_focus_m,
                                scale_factor,
                                target_alignment_error_m,
                                self.focus_depth,
                            )
                        )
                        if depth_output is not None:
                            depth_output.write(
                                self._depth_visualization(scaled_depth_m)
                            )
                        output.write(rendered)
                        if coc_output is not None:
                            coc_output.write(
                                self._coc_visualization(effective_coc)
                            )
                        sys.stdout.write(
                            f"\r  🎥 [{frame_index}/{total_frames}] "
                            f"{frame_index * 100 // max(total_frames, 1)}%"
                            f" | raw target {raw_target_median_m:.2f}m"
                            f" | S(t) {scheduled_focus_m:.2f}m"
                            f" | scale {scale_factor:.3f}"
                        )
                        sys.stdout.flush()
                        continue

            # After target loss, do not reuse a depth map.  Apply the final
            # valid target scale to each new clean current-frame depth and
            # release its current CoC field to deep focus over 1.5 seconds.
            if self.focus_depth is None or self.last_target_scale is None:
                rendered = frame
                state = "no-focus"
            else:
                scaled_depth_m = np.maximum(
                    raw_depth_m.astype(np.float32) * self.last_target_scale,
                    0.1,
                ).astype(np.float32)
                self.transition_frame += 1
                progress = min(
                    1.0,
                    self.transition_frame / self.transition_frames,
                )
                strength = 0.5 * (1.0 + math.cos(math.pi * progress))
                effective_coc = (
                    self._compute_blur_map(
                        scaled_depth_m,
                        self.focus_depth,
                    )
                    * strength
                )
                rendered = self._render_variable_blur(
                    frame,
                    effective_coc,
                )
                state = "deep-focus-pull"
                scale_factor = self.last_target_scale
                scheduled_focus_m = self.focus_depth

            log_rows.append(
                (
                    frame_index,
                    state,
                    schedule_progress,
                    raw_target_median_m,
                    scheduled_focus_m,
                    scale_factor,
                    target_alignment_error_m,
                    self.focus_depth,
                )
            )
            if depth_output is not None:
                depth_output.write(
                    self._depth_visualization(scaled_depth_m)
                )
            output.write(rendered)
            if coc_output is not None:
                coc_output.write(
                    self._coc_visualization(effective_coc)
                )
            sys.stdout.write(
                f"\r  🎥 [{frame_index}/{total_frames}] "
                f"{frame_index * 100 // max(total_frames, 1)}%"
                " | Target-anchored deep-focus pull"
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
                        "schedule_progress",
                        "raw_target_median_m",
                        "forced_target_s_m",
                        "target_scale_factor",
                        "scaled_target_alignment_error_m",
                        "focus_distance_used_for_coc_m",
                    )
                )
                writer.writerows(log_rows)

        elapsed = time.time() - started_at
        print(f"\n\n  ✅ {self.pipeline_name} complete in {elapsed:.1f}s")
        print(f"  Output: {output_path}")
        if depth_output_path:
            print(f"  Target-anchored scene depth: {depth_output_path}")
        if coc_output_path:
            print(f"  Target-anchored effective CoC: {coc_output_path}")
        if focus_log_path:
            print(f"  Kinematic scale/focus log: {focus_log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Smart DoF V17 — target-anchored kinematic focus schedule "
            "with clean per-frame metric depth"
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
    parser.add_argument("--focal-length-mm", type=float, default=135.0)
    parser.add_argument("--f-number", type=float, default=2.0)
    parser.add_argument("--sensor-width-mm", type=float, default=36.0)
    parser.add_argument("--coc-gain", type=float, default=4.0)
    parser.add_argument("--max-blur", type=float, default=28.0)
    parser.add_argument("--transition-sec", type=float, default=1.5)
    parser.add_argument(
        "--kinematic-start-distance-m",
        type=float,
        default=None,
        help=(
            "Forced first-frame focus distance. Default: first valid "
            "SAM-mask raw-depth median."
        ),
    )
    parser.add_argument(
        "--kinematic-end-distance-m",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--kinematic-end-frame",
        type=int,
        default=None,
        help=(
            "Frame at which S(t) reaches its end distance. For this sample, "
            "V15 tracking shows the last valid target frame is 120."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional validation limit; does not change the focus schedule.",
    )
    parser.add_argument("--sam-model", type=Path, default=None)
    args = parser.parse_args()

    pipeline = CinematicDoFv17(
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
