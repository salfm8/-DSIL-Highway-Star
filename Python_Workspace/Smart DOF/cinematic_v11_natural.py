"""Smart DoF V11 — parallel-focal-plane Natural DoF baseline.

V11 deliberately separates subject selection from depth-of-field rendering:

* MobileSAM is used only to obtain a robust target-depth observation.
* The mask never protects, sharpens, feathers, or spatially weights output pixels.
* Every output pixel receives a blur radius computed only from the absolute
  difference between its temporally stabilized depth and the target depth.

Depth Anything provides relative depth rather than calibrated metric distance,
so the resulting radius is a relative CoC approximation.  The focal-plane
constraint is nevertheless strict: equal stabilized depths receive equal blur
radii regardless of object identity or image position.
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
from cinematic_v8_semantic_focus import parse_bbox, parse_points
from cinematic_v10_progressive_focus import CinematicDoFv10


class CinematicDoFv11(CinematicDoFv10):
    """Natural DoF whose focal plane is parallel to the image sensor."""

    def __init__(
        self,
        *args,
        dead_zone: float = 0.018,
        full_blur_delta: float = 0.30,
        depth_current_weight: float = 0.76,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pipeline_name = "Smart DoF v11.0 — Parallel Focal Plane Baseline"

        self.dead_zone = float(dead_zone)
        self.full_blur_delta = float(full_blur_delta)
        self.depth_current_weight = float(depth_current_weight)
        if self.dead_zone < 0.0:
            raise ValueError("--dead-zone must be non-negative")
        if self.full_blur_delta <= self.dead_zone:
            raise ValueError("--full-blur-delta must be greater than --dead-zone")
        if not 0.0 < self.depth_current_weight <= 1.0:
            raise ValueError("--depth-current-weight must be in (0, 1]")

        self.previous_stable_depth: np.ndarray | None = None
        self.previous_depth_gray: np.ndarray | None = None

        # V9/V10 smooth their final blur field after it has been computed.
        # V11 disables that path: temporal stability is applied to depth first,
        # and CoC is then recomputed from the stabilized depth every frame.
        self.previous_blur_map = None
        self.previous_blur_point = None

        print("  Parallel focal-plane baseline enabled")
        print("  SAM usage: target-depth estimation only")
        print(
            f"  CoC curve: |depth-target| -> dead zone {self.dead_zone:.3f}"
            f" -> smootherstep -> {self.max_blur:.1f}px"
        )
        print(f"  Full blur at relative-depth delta: {self.full_blur_delta:.3f}")
        print(f"  Current-frame depth weight: {self.depth_current_weight:.2f}")

    @staticmethod
    def _smootherstep(value: np.ndarray) -> np.ndarray:
        """Quintic interpolation with zero slope at both ends."""
        value = np.clip(value, 0.0, 1.0)
        return value**3 * (value * (value * 6.0 - 15.0) + 10.0)

    def _stabilize_depth(self, current_depth: np.ndarray, current_gray: np.ndarray) -> np.ndarray:
        """Warp the previous stable depth into the current frame and blend it safely.

        Farneback flow is evaluated from the current image back to the previous
        image so each current pixel can sample its previous-frame correspondence.
        Photometric disagreement automatically reduces the prior-frame weight.
        """
        if self.previous_stable_depth is None or self.previous_depth_gray is None:
            stable = current_depth.astype(np.float32)
        else:
            backward_flow = cv2.calcOpticalFlowFarneback(
                current_gray,
                self.previous_depth_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=21,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            height, width = current_depth.shape
            grid_x, grid_y = np.meshgrid(
                np.arange(width, dtype=np.float32),
                np.arange(height, dtype=np.float32),
            )
            map_x = grid_x + backward_flow[..., 0]
            map_y = grid_y + backward_flow[..., 1]
            warped_depth = cv2.remap(
                self.previous_stable_depth,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            warped_gray = cv2.remap(
                self.previous_depth_gray,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            photometric_error = np.abs(
                current_gray.astype(np.float32) - warped_gray.astype(np.float32)
            )
            confidence = np.exp(-photometric_error / 18.0).astype(np.float32)
            prior_weight = (1.0 - self.depth_current_weight) * confidence
            stable = current_depth * (1.0 - prior_weight) + warped_depth * prior_weight

        stable = np.clip(stable, 0.0, 1.0).astype(np.float32)
        self.previous_stable_depth = stable.copy()
        self.previous_depth_gray = current_gray.copy()
        return stable

    def _get_depth_map(self, rgb: np.ndarray, width: int, height: int) -> np.ndarray:
        raw_depth = super()._get_depth_map(rgb, width, height)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return self._stabilize_depth(raw_depth, gray)

    def _compute_blur_map(self, depth: np.ndarray, focus_depth: float) -> np.ndarray:
        """Compute relative CoC from depth difference only.

        Let d be stabilized relative depth and d_f be the EMA-smoothed target
        depth.  The blur radius is

            delta = |d - d_f|
            u = clamp((delta - dead_zone) /
                      (full_blur_delta - dead_zone), 0, 1)
            radius = max_blur * smootherstep(u)

        No pixel coordinate, mask membership, object class, or distance to the
        selected object appears in this equation.
        """
        depth_delta = np.abs(depth.astype(np.float32) - np.float32(focus_depth))
        normalized = (depth_delta - self.dead_zone) / (
            self.full_blur_delta - self.dead_zone
        )
        coc_radius = self.max_blur * self._smootherstep(normalized)
        return np.clip(coc_radius, 0.0, self.max_blur).astype(np.float32)

    def _stabilize_blur_map(
        self, blur_map: np.ndarray, point: np.ndarray | None
    ) -> np.ndarray:
        """Do not add mask-, position-, or stale-map-dependent blur."""
        return np.asarray(blur_map, dtype=np.float32)

    @staticmethod
    def _depth_visualization(depth: np.ndarray) -> np.ndarray:
        depth_u8 = np.clip(depth * 255.0, 0.0, 255.0).astype(np.uint8)
        return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)

    def process_video(
        self,
        input_path: str,
        output_path: str,
        depth_output_path: str | None = None,
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

        depth_output = None
        if depth_output_path:
            depth_output = cv2.VideoWriter(
                depth_output_path, fourcc, fps, (width, height)
            )
            if not depth_output.isOpened():
                cap.release()
                output.release()
                raise RuntimeError(
                    f"Cannot create depth visualization video: {depth_output_path}"
                )

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

            if depth_output is not None:
                depth_output.write(self._depth_visualization(depth))

            if not self.tracking_lost:
                tracking_ok = self._track_point(gray)
                if not tracking_ok or not self._point_in_bounds(
                    self.current_point, width, height
                ):
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

                    # This is the only place where the SAM mask affects V11:
                    # it selects samples used to observe the focal depth.
                    observed_depth = self._subject_depth(depth, stable_mask)
                    if observed_depth is None:
                        self._begin_focus_pull(fps)
                    else:
                        self.focus_depth = (
                            observed_depth
                            if self.focus_depth is None
                            else self.focus_depth * self.depth_ema
                            + observed_depth * (1.0 - self.depth_ema)
                        )
                        blur_map = self._compute_blur_map(depth, self.focus_depth)
                        rendered = self._render_variable_blur(frame, blur_map)
                        output.write(rendered)
                        sys.stdout.write(
                            f"\r  🎥 [{frame_index}/{total_frames}] "
                            f"{frame_index * 100 // max(total_frames, 1)}%"
                            " | Parallel-plane Natural DoF"
                        )
                        sys.stdout.flush()
                        continue

            # After target loss, retain only the last stable target depth.
            # Recompute CoC from each new frame's stabilized depth, then reduce
            # its strength to zero with cosine easing over transition_sec.
            if self.focus_depth is None:
                rendered = frame
            else:
                self.transition_frame += 1
                progress = min(1.0, self.transition_frame / self.transition_frames)
                strength = 0.5 * (1.0 + math.cos(math.pi * progress))
                blur_map = self._compute_blur_map(depth, self.focus_depth)
                rendered = self._render_variable_blur(frame, blur_map * strength)

            output.write(rendered)
            sys.stdout.write(
                f"\r  🎥 [{frame_index}/{total_frames}] "
                f"{frame_index * 100 // max(total_frames, 1)}%"
                " | Natural deep-focus pull"
            )
            sys.stdout.flush()

        cap.release()
        output.release()
        if depth_output is not None:
            depth_output.release()
        elapsed = time.time() - started_at
        print(f"\n\n  ✅ {self.pipeline_name} rendering complete in {elapsed:.1f}s")
        print(f"  Output: {output_path}")
        if depth_output_path:
            print(f"  Depth visualization: {depth_output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Smart DoF V11 — depth-only parallel focal-plane Natural DoF baseline"
        )
    )
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Natural DoF output MP4")
    parser.add_argument(
        "--depth-output",
        default=None,
        help="Optional temporally stabilized depth visualization MP4",
    )
    parser.add_argument(
        "--point",
        required=True,
        type=parse_point,
        help="Reliable tracking point inside target: x,y",
    )
    parser.add_argument(
        "--bbox",
        required=True,
        type=parse_bbox,
        help="First-frame target box used only for target-depth sampling",
    )
    parser.add_argument(
        "--positive-points",
        default=None,
        help="Optional positive SAM prompts: x,y;x,y",
    )
    parser.add_argument(
        "--negative-points",
        default=None,
        help="Optional negative SAM prompts: x,y;x,y",
    )
    parser.add_argument(
        "--max-blur",
        type=float,
        default=14.0,
        help="Maximum relative CoC/blur radius in pixels",
    )
    parser.add_argument(
        "--dead-zone",
        type=float,
        default=0.018,
        help="Relative-depth interval rendered fully sharp",
    )
    parser.add_argument(
        "--full-blur-delta",
        type=float,
        default=0.30,
        help="Relative-depth difference that reaches maximum blur",
    )
    parser.add_argument(
        "--depth-current-weight",
        type=float,
        default=0.76,
        help="Current-frame weight in flow-warped temporal depth stabilization",
    )
    parser.add_argument(
        "--transition-sec",
        type=float,
        default=1.5,
        help="Deep-focus transition duration in seconds",
    )
    parser.add_argument(
        "--sam-interval",
        type=int,
        default=2,
        help="Run SAM every N frames; warp its mask between refreshes",
    )
    parser.add_argument(
        "--sam-model",
        type=Path,
        default=None,
        help="Optional MobileSAM model path",
    )
    args = parser.parse_args()

    pipeline = CinematicDoFv11(
        init_point=args.point,
        focus_bbox=args.bbox,
        positive_points=parse_points(args.positive_points),
        negative_points=parse_points(args.negative_points),
        max_blur=args.max_blur,
        transition_sec=args.transition_sec,
        sam_interval=args.sam_interval,
        model_path=args.sam_model,
        dead_zone=args.dead_zone,
        full_blur_delta=args.full_blur_delta,
        depth_current_weight=args.depth_current_weight,
    )
    pipeline.process_video(args.input, args.output, args.depth_output)


if __name__ == "__main__":
    main()
