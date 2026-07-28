"""Smart DoF V16 — clean per-frame depth with road-based scale anchoring.

Every frame is inferred independently by Depth Anything V2 Metric. A fixed
bottom-centre road patch anchors the monocular scale:

    scale(t) = known_anchor_distance / mean(raw_depth_t[anchor_roi])
    Z_t      = raw_depth_t * scale(t)

Optical flow remains restricted to target-point tracking. SAM runs on the
current frame and S(t) is the median anchored depth inside its current mask.
No depth history, warp, reprojection, or camera-motion prior is used.
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


class CinematicDoFv16(CinematicDoFv15):
    """Per-frame metric DoF with a constant-distance road scale anchor."""

    def __init__(
        self,
        *args,
        anchor_distance_m: float = 5.0,
        anchor_y_offset: int = 80,
        anchor_width: int = 80,
        anchor_height: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if anchor_distance_m <= 0.0:
            raise ValueError("--anchor-distance-m must be positive")
        if anchor_y_offset < 1:
            raise ValueError("--anchor-y-offset must be positive")
        if anchor_width < 4 or anchor_height < 4:
            raise ValueError("Anchor ROI dimensions must be at least 4 pixels")

        self.anchor_distance_m = float(anchor_distance_m)
        self.anchor_y_offset = int(anchor_y_offset)
        self.anchor_width = int(anchor_width)
        self.anchor_height = int(anchor_height)
        self.pipeline_name = "Smart DoF v16.0 — Road-Anchored Dynamic Focus"

        print("  V16 road scale anchoring enabled")
        print(f"  Fixed anchor distance: {self.anchor_distance_m:.2f}m")
        print(
            "  Anchor ROI: bottom-centre, "
            f"y offset {self.anchor_y_offset}px, "
            f"{self.anchor_width}x{self.anchor_height}px"
        )
        print("  Scale(t) = fixed distance / current raw anchor mean")
        print("  Depth remains independent per frame; no warp/reprojection")

    def _anchor_bounds(
        self,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        center_x = width // 2
        center_y = int(
            np.clip(
                height - self.anchor_y_offset,
                self.anchor_height // 2,
                height - 1 - self.anchor_height // 2,
            )
        )
        x0 = max(0, center_x - self.anchor_width // 2)
        x1 = min(width, x0 + self.anchor_width)
        y0 = max(0, center_y - self.anchor_height // 2)
        y1 = min(height, y0 + self.anchor_height)
        return x0, y0, x1, y1

    def _scale_anchor_depth(
        self,
        raw_depth_m: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Return anchored depth, trimmed ROI mean, and scale factor."""
        height, width = raw_depth_m.shape
        x0, y0, x1, y1 = self._anchor_bounds(width, height)
        samples = raw_depth_m[y0:y1, x0:x1].reshape(-1)
        samples = samples[np.isfinite(samples) & (samples > 0.1)]
        if samples.size < 16:
            raise RuntimeError("Road anchor ROI has insufficient valid depth")

        # Reject lane-edge/object outliers while retaining an actual arithmetic
        # mean for the road-anchor measurement requested by the design.
        low, high = np.percentile(samples, (10.0, 90.0))
        trimmed = samples[(samples >= low) & (samples <= high)]
        anchor_mean_m = float(np.mean(trimmed))
        scale_factor = self.anchor_distance_m / max(anchor_mean_m, 1e-6)
        anchored = np.clip(
            raw_depth_m * scale_factor,
            0.1,
            self.model_max_depth_m,
        ).astype(np.float32)
        return anchored, anchor_mean_m, float(scale_factor)

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
            tuple[int, str, float, float, float | None]
        ] = []

        frame_index = 0
        started_at = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            raw_depth_m = self._get_clean_metric_depth(
                rgb,
                width,
                height,
            )
            current_depth_m, anchor_mean_m, scale_factor = (
                self._scale_anchor_depth(raw_depth_m)
            )
            effective_coc = np.zeros((height, width), dtype=np.float32)
            observed_focus_m: float | None = None

            if not self.tracking_lost:
                # Optical flow is used only to update this target coordinate.
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
                    observed_focus_m = self._subject_depth(
                        current_depth_m,
                        current_mask,
                    )
                    if observed_focus_m is None:
                        self._begin_focus_pull(fps)
                    else:
                        self.focus_depth = float(observed_focus_m)
                        effective_coc = self._compute_blur_map(
                            current_depth_m,
                            self.focus_depth,
                        )
                        rendered = self._render_variable_blur(
                            frame,
                            effective_coc,
                        )
                        log_rows.append(
                            (
                                frame_index,
                                "tracking",
                                anchor_mean_m,
                                scale_factor,
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
                            f" | anchor {anchor_mean_m:.2f}m"
                            f" x{scale_factor:.3f}"
                            f" | S(t) {self.focus_depth:.2f}m"
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

            log_rows.append(
                (
                    frame_index,
                    state,
                    anchor_mean_m,
                    scale_factor,
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
                " | Road-anchored deep-focus pull"
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
                        "raw_anchor_mean_m",
                        "scale_factor",
                        "anchored_target_focus_distance_m",
                    )
                )
                writer.writerows(log_rows)

        elapsed = time.time() - started_at
        print(f"\n\n  ✅ {self.pipeline_name} complete in {elapsed:.1f}s")
        print(f"  Output: {output_path}")
        if depth_output_path:
            print(f"  Road-anchored metric depth: {depth_output_path}")
        if coc_output_path:
            print(f"  Road-anchored CoC: {coc_output_path}")
        if focus_log_path:
            print(f"  Scale/focus log: {focus_log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Smart DoF V16 — fresh metric depth with constant road scale anchor"
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
    parser.add_argument("--anchor-distance-m", type=float, default=5.0)
    parser.add_argument("--anchor-y-offset", type=int, default=80)
    parser.add_argument("--anchor-width", type=int, default=80)
    parser.add_argument("--anchor-height", type=int, default=20)
    parser.add_argument("--sam-model", type=Path, default=None)
    args = parser.parse_args()

    pipeline = CinematicDoFv16(
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
        anchor_distance_m=args.anchor_distance_m,
        anchor_y_offset=args.anchor_y_offset,
        anchor_width=args.anchor_width,
        anchor_height=args.anchor_height,
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
