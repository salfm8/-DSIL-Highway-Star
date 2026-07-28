"""Smart DoF V19 — pure lens rack focus without target tracking.

V19 abandons object-locked focus.  The far target and near bonnet are sampled
only once on the first frame to establish a fixed affine scene calibration:

    60 m = a * first_target_raw_median + b
     3 m = a * first_bonnet_raw_median + b
    Z_t  = a * raw_metric_depth_t + b

The coefficients are never updated.  No optical flow, repeated SAM inference,
target tracking, or target-driven focus distance is used after the first
frame.

The virtual focus ring then moves independently through the scene:

    S(t) = lerp(60 m, 3 m, decoded_video_progress)

All pixels are rendered only from their fixed-scale metric depth and the
current thin-lens Circle of Confusion.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from cinematic_v7_natural import parse_point
from cinematic_v8_semantic_focus import parse_bbox, parse_points
from cinematic_v15_clean_dynamic_focus import CinematicDoFv15


class CinematicDoFv19(CinematicDoFv15):
    """A fixed-scale scene swept by an independent 60m-to-3m focal plane."""

    def __init__(
        self,
        *args,
        rack_start_distance_m: float = 60.0,
        rack_end_distance_m: float = 3.0,
        rack_end_frame: int | None = None,
        scene_scale: float | None = None,
        near_roi_y_offset: int = 55,
        near_roi_width: int = 160,
        near_roi_height: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if rack_start_distance_m <= 0.0:
            raise ValueError("--rack-start-distance-m must be positive")
        if rack_end_distance_m <= 0.0:
            raise ValueError("--rack-end-distance-m must be positive")
        if rack_end_distance_m >= rack_start_distance_m:
            raise ValueError(
                "--rack-end-distance-m must be smaller than "
                "--rack-start-distance-m"
            )
        if rack_end_frame is not None and rack_end_frame < 2:
            raise ValueError("--rack-end-frame must be at least 2")
        if scene_scale is not None and scene_scale <= 0.0:
            raise ValueError("--scene-scale must be positive")
        if near_roi_y_offset < 1:
            raise ValueError("--near-roi-y-offset must be positive")
        if near_roi_width < 4 or near_roi_height < 4:
            raise ValueError("Near calibration ROI must be at least 4x4")

        self.rack_start_distance_m = float(rack_start_distance_m)
        self.rack_end_distance_m = float(rack_end_distance_m)
        self.rack_end_frame = rack_end_frame
        self.requested_scene_scale = (
            None if scene_scale is None else float(scene_scale)
        )
        self.near_roi_y_offset = int(near_roi_y_offset)
        self.near_roi_width = int(near_roi_width)
        self.near_roi_height = int(near_roi_height)
        self.depth_scale_a: float | None = None
        self.depth_offset_b: float | None = None
        self.initial_target_raw_median_m: float | None = None
        self.initial_near_raw_median_m: float | None = None
        self.pipeline_name = "Smart DoF v19.0 — Pure 60m-to-3m Rack Focus"

        print("  V19 pure rack-focus mode enabled")
        print(
            f"  Focus-ring sweep: {self.rack_start_distance_m:.2f}m "
            f"-> {self.rack_end_distance_m:.2f}m"
        )
        print("  Tracking after frame 1: disabled")
        print("  Optical flow: disabled")
        print("  Repeated SAM inference: disabled")
        if self.requested_scene_scale is None:
            print(
                "  Scene calibration: first-frame target "
                f"={self.rack_start_distance_m:.2f}m and bonnet "
                f"={self.rack_end_distance_m:.2f}m, then frozen"
            )
        else:
            print(
                f"  Scene scale: fixed user value "
                f"{self.requested_scene_scale:.6f}"
            )
        print(
            "  Final sharpness: fixed calibrated depth + independent "
            "thin-lens CoC"
        )

    def _near_roi_median(self, raw_depth_m: np.ndarray) -> float:
        """Measure a robust first-frame bonnet depth near the bottom centre."""
        height, width = raw_depth_m.shape
        center_x = width // 2
        center_y = int(
            np.clip(
                height - self.near_roi_y_offset,
                self.near_roi_height // 2,
                height - 1 - self.near_roi_height // 2,
            )
        )
        x0 = max(0, center_x - self.near_roi_width // 2)
        x1 = min(width, x0 + self.near_roi_width)
        y0 = max(0, center_y - self.near_roi_height // 2)
        y1 = min(height, y0 + self.near_roi_height)
        samples = raw_depth_m[y0:y1, x0:x1].reshape(-1)
        samples = samples[np.isfinite(samples) & (samples > 0.1)]
        if samples.size < 16:
            raise RuntimeError("Bonnet calibration ROI has insufficient depth")
        low, high = np.percentile(samples, (10.0, 90.0))
        trimmed = samples[(samples >= low) & (samples <= high)]
        return float(np.median(trimmed))

    @staticmethod
    def _count_decodable_frames(input_path: str) -> int:
        """Count frames that can actually be decoded, not unreliable metadata."""
        probe = cv2.VideoCapture(input_path)
        if not probe.isOpened():
            raise RuntimeError(f"Cannot open input video: {input_path}")
        count = 0
        while probe.grab():
            count += 1
        probe.release()
        if count < 1:
            raise RuntimeError(f"No decodable frames in input: {input_path}")
        return count

    @staticmethod
    def _rack_progress(frame_index: int, end_frame: int) -> float:
        return float(
            np.clip(
                (frame_index - 1) / max(float(end_frame - 1), 1.0),
                0.0,
                1.0,
            )
        )

    def _rack_focus_distance(
        self,
        frame_index: int,
        end_frame: int,
    ) -> tuple[float, float]:
        progress = self._rack_progress(frame_index, end_frame)
        focus_distance_m = (
            self.rack_start_distance_m
            + (self.rack_end_distance_m - self.rack_start_distance_m)
            * progress
        )
        return float(focus_distance_m), progress

    def process_video(
        self,
        input_path: str,
        output_path: str,
        depth_output_path: str | None = None,
        coc_output_path: str | None = None,
        focus_log_path: str | None = None,
        max_frames: int | None = None,
    ) -> None:
        decoded_frames = self._count_decodable_frames(input_path)
        schedule_end_frame = self.rack_end_frame or decoded_frames
        if schedule_end_frame > decoded_frames:
            print(
                "  ⚠️ Rack end frame exceeds the decodable video length; "
                "3m will not be reached"
            )
        render_limit = (
            decoded_frames
            if max_frames is None
            else min(decoded_frames, max_frames)
        )
        if render_limit < 1:
            raise ValueError("--max-frames must be positive")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open input video: {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
                float,
                float,
                float,
                float,
                float | None,
                float | None,
            ]
        ] = []

        frame_index = 0
        started_at = time.time()
        while frame_index < render_limit:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Clean current-frame Depth Anything V2 Metric inference.  No
            # depth history, temporal blend, warp, or reprojection.
            raw_depth_m = self._get_clean_metric_depth(rgb, width, height)

            if self.depth_scale_a is None or self.depth_offset_b is None:
                if self.requested_scene_scale is not None:
                    self.depth_scale_a = self.requested_scene_scale
                    self.depth_offset_b = 0.0
                else:
                    # First-frame-only endpoints: SAM supplies the far target,
                    # and a bottom-centre bonnet ROI supplies the near end.
                    # Neither measurement is repeated on later frames.
                    initial_mask = self._get_sam_mask(
                        frame,
                        self.current_point,
                    )
                    self.initial_target_raw_median_m = self._subject_depth(
                        raw_depth_m,
                        initial_mask,
                    )
                    if self.initial_target_raw_median_m is None:
                        raise RuntimeError(
                            "Could not calibrate first-frame target scale"
                        )
                    self.initial_near_raw_median_m = self._near_roi_median(
                        raw_depth_m
                    )
                    raw_span = (
                        self.initial_target_raw_median_m
                        - self.initial_near_raw_median_m
                    )
                    if raw_span <= 1e-6:
                        raise RuntimeError(
                            "Far target must be deeper than bonnet ROI"
                        )
                    self.depth_scale_a = (
                        self.rack_start_distance_m
                        - self.rack_end_distance_m
                    ) / raw_span
                    self.depth_offset_b = (
                        self.rack_end_distance_m
                        - self.depth_scale_a
                        * self.initial_near_raw_median_m
                    )
                    print(
                        "\n  First-frame calibration: "
                        f"raw target {self.initial_target_raw_median_m:.2f}m "
                        f"-> {self.rack_start_distance_m:.2f}m, "
                        f"raw bonnet {self.initial_near_raw_median_m:.2f}m "
                        f"-> {self.rack_end_distance_m:.2f}m"
                    )
                    print(
                        "  Frozen affine depth: "
                        f"Z = {self.depth_scale_a:.6f} * D_raw "
                        f"{self.depth_offset_b:+.6f}"
                    )

            scene_depth_m = np.maximum(
                raw_depth_m.astype(np.float32) * self.depth_scale_a
                + self.depth_offset_b,
                0.1,
            ).astype(np.float32)
            focus_distance_m, progress = self._rack_focus_distance(
                frame_index,
                schedule_end_frame,
            )

            # No target mask is used.  The zero-CoC surface is the global
            # parallel plane Z == S(t), which sweeps through the whole scene.
            effective_coc = self._compute_blur_map(
                scene_depth_m,
                focus_distance_m,
            )
            rendered = self._render_variable_blur(
                frame,
                effective_coc,
            )

            output.write(rendered)
            if depth_output is not None:
                depth_output.write(
                    self._depth_visualization(scene_depth_m)
                )
            if coc_output is not None:
                coc_output.write(
                    self._coc_visualization(effective_coc)
                )
            log_rows.append(
                (
                    frame_index,
                    progress,
                    focus_distance_m,
                    self.depth_scale_a,
                    self.depth_offset_b,
                    self.initial_target_raw_median_m,
                    self.initial_near_raw_median_m,
                )
            )

            sys.stdout.write(
                f"\r  🎥 [{frame_index}/{render_limit}] "
                f"{frame_index * 100 // max(render_limit, 1)}%"
                f" | Rack S(t) {focus_distance_m:.2f}m"
                f" | fixed depth a={self.depth_scale_a:.4f}"
                f" b={self.depth_offset_b:+.2f}"
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
                        "rack_progress",
                        "rack_focus_distance_m",
                        "fixed_depth_scale_a",
                        "fixed_depth_offset_b",
                        "first_frame_target_raw_median_m",
                        "first_frame_bonnet_raw_median_m",
                    )
                )
                writer.writerows(log_rows)

        elapsed = time.time() - started_at
        print(f"\n\n  ✅ {self.pipeline_name} complete in {elapsed:.1f}s")
        print(f"  Output: {output_path}")
        if depth_output_path:
            print(f"  Fixed-scale metric depth: {depth_output_path}")
        if coc_output_path:
            print(f"  Sweeping rack-focus CoC: {coc_output_path}")
        if focus_log_path:
            print(f"  Rack-focus log: {focus_log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Smart DoF V19 — pure 60m-to-3m rack focus without "
            "object tracking"
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
    parser.add_argument("--f-number", type=float, default=1.4)
    parser.add_argument("--sensor-width-mm", type=float, default=36.0)
    parser.add_argument("--coc-gain", type=float, default=5.0)
    parser.add_argument("--max-blur", type=float, default=60.0)
    parser.add_argument("--rack-start-distance-m", type=float, default=60.0)
    parser.add_argument("--rack-end-distance-m", type=float, default=3.0)
    parser.add_argument(
        "--rack-end-frame",
        type=int,
        default=None,
        help=(
            "Frame at which the focus ring reaches 3m. Default: count all "
            "actually decodable frames, avoiding unreliable container metadata."
        ),
    )
    parser.add_argument(
        "--scene-scale",
        type=float,
        default=None,
        help=(
            "Optional fixed global scale. Default: calibrate once from the "
            "first target mask so its initial depth equals the 60m start."
        ),
    )
    parser.add_argument("--near-roi-y-offset", type=int, default=55)
    parser.add_argument("--near-roi-width", type=int, default=160)
    parser.add_argument("--near-roi-height", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--sam-model", type=Path, default=None)
    args = parser.parse_args()

    pipeline = CinematicDoFv19(
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
        rack_start_distance_m=args.rack_start_distance_m,
        rack_end_distance_m=args.rack_end_distance_m,
        rack_end_frame=args.rack_end_frame,
        scene_scale=args.scene_scale,
        near_roi_y_offset=args.near_roi_y_offset,
        near_roi_width=args.near_roi_width,
        near_roi_height=args.near_roi_height,
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
