"""Smart DoF V14 — object-locked dynamic focal plane.

V14 separates two different measurements that must not be confused:

* Optical expansion estimates camera approach and reconstructs the current
  metric scene depth from the prior view.
* The focus distance S(t) is then re-read from the tracked target in that
  reconstructed *current-frame* depth.

The motion estimate is never used directly as the CoC focus distance. This
keeps the focal plane attached to the tracked object while allowing its Z
position to approach the camera.
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
from cinematic_v13_dynamic_target import CinematicDoFv13


class CinematicDoFv14(CinematicDoFv13):
    """Reconstruct current scene depth, then measure S(t) at the target."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pipeline_name = "Smart DoF v14.0 — Object-Locked Focus Puller"
        self.motion_focus_m: float | None = None
        print("  V14 object-locked physical focus puller enabled")
        print("  Optical flow: current scene-depth reconstruction")
        print("  SAM: current target S(t) measurement after reconstruction")
        print("  CoC focus: measured S(t), never the motion prior")

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
        log_rows: list[
            tuple[
                int,
                str,
                float | None,
                float | None,
                float | None,
            ]
        ] = []

        frame_index = 0
        started_at = time.time()
        current_scene_depth: np.ndarray | None = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            raw_metric_depth = self._get_depth_map(rgb, width, height)
            current_scene_depth = raw_metric_depth
            effective_coc = np.zeros((height, width), dtype=np.float32)
            raw_target_m: float | None = None
            current_target_m: float | None = None

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
                    motion_mask, scale_step = (
                        self._propagate_scene_focus_mask(
                            gray,
                            stable_mask,
                        )
                    )
                    raw_target_m = self._subject_depth(
                        raw_metric_depth,
                        stable_mask,
                    )
                    if raw_target_m is None:
                        self._begin_focus_pull(fps)
                    else:
                        # The optical scale supplies only a scene-motion prior.
                        self.motion_focus_m = (
                            self._update_scene_focus_distance(
                                raw_target_m,
                                scale_step,
                            )
                        )
                        current_scene_depth = self._update_scene_depth(
                            raw_metric_depth,
                            gray,
                            self.motion_focus_m,
                            motion_mask,
                        )

                        # Critical V14 rule: measure the current target again
                        # after scene reconstruction. This measured value, not
                        # motion_focus_m, is the CoC focal distance.
                        current_target_m = self._subject_depth(
                            current_scene_depth,
                            stable_mask,
                        )
                        if current_target_m is None:
                            self._begin_focus_pull(fps)
                        else:
                            focus_m = self._update_dynamic_focus(
                                current_target_m
                            )
                            effective_coc = self._compute_blur_map(
                                current_scene_depth,
                                focus_m,
                            )
                            rendered = self._render_variable_blur(
                                frame,
                                effective_coc,
                            )
                            log_rows.append(
                                (
                                    frame_index,
                                    "tracking",
                                    raw_target_m,
                                    self.motion_focus_m,
                                    focus_m,
                                )
                            )
                            if depth_output is not None:
                                depth_output.write(
                                    self._depth_visualization(
                                        current_scene_depth
                                    )
                                )
                            output.write(rendered)
                            if coc_output is not None:
                                coc_output.write(
                                    self._coc_visualization(effective_coc)
                                )
                            sys.stdout.write(
                                f"\r  🎥 [{frame_index}/{total_frames}] "
                                f"{frame_index * 100 // max(total_frames, 1)}%"
                                f" | Raw {raw_target_m:.1f}m"
                                f" | motion {self.motion_focus_m:.1f}m"
                                f" | measured S(t) {focus_m:.1f}m"
                            )
                            sys.stdout.flush()
                            continue

            if self.focus_depth is None:
                rendered = frame
                state = "no-focus"
            else:
                # Keep reconstructing the abandoned plane during the short
                # focus pull, but hold the final measured object focus and
                # reduce its CoC strength to zero.
                if self.motion_focus_m is not None:
                    self.motion_focus_m = max(
                        0.5,
                        self.motion_focus_m - self.last_camera_advance_m,
                    )
                    current_scene_depth = self._update_scene_depth(
                        raw_metric_depth,
                        gray,
                        self.motion_focus_m,
                        None,
                    )
                self.transition_frame += 1
                progress = min(
                    1.0,
                    self.transition_frame / self.transition_frames,
                )
                strength = 0.5 * (1.0 + math.cos(math.pi * progress))
                effective_coc = (
                    self._compute_blur_map(
                        current_scene_depth,
                        self.focus_depth,
                    )
                    * strength
                )
                rendered = self._render_variable_blur(
                    frame,
                    effective_coc,
                )
                state = "deep-focus-pull"

            log_rows.append(
                (
                    frame_index,
                    state,
                    raw_target_m,
                    self.motion_focus_m,
                    self.focus_depth,
                )
            )
            if depth_output is not None:
                depth_output.write(
                    self._depth_visualization(current_scene_depth)
                )
            output.write(rendered)
            if coc_output is not None:
                coc_output.write(
                    self._coc_visualization(effective_coc)
                )
            sys.stdout.write(
                f"\r  🎥 [{frame_index}/{total_frames}] "
                f"{frame_index * 100 // max(total_frames, 1)}%"
                " | Object-lock deep-focus pull"
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
                        "raw_model_target_depth_m",
                        "motion_prior_depth_m",
                        "measured_current_focus_distance_m",
                    )
                )
                writer.writerows(log_rows)

        elapsed = time.time() - started_at
        print(f"\n\n  ✅ {self.pipeline_name} complete in {elapsed:.1f}s")
        print(f"  Output: {output_path}")
        if depth_output_path:
            print(f"  Reconstructed current depth: {depth_output_path}")
        if coc_output_path:
            print(f"  Object-locked CoC: {coc_output_path}")
        if focus_log_path:
            print(f"  Focus audit log: {focus_log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Smart DoF V14 — reconstructed current depth with object-locked S(t)"
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

    pipeline = CinematicDoFv14(
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
