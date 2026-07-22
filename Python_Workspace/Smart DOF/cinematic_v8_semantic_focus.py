"""Smart DoF v8 — region-prompted semantic focus with depth-aware falloff.

V8 extends V7 for shots where the selected focus subject is a large object
(for example a building), not the small logo or texture under a single click.
The target is defined by a first-frame bounding box plus optional positive and
negative SAM points.  The subject mask remains fully sharp; outside it, depth
falloff is combined with a restrained semantic-focus falloff.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from cinematic_v7_natural import CinematicDoFv7, parse_point


def parse_bbox(value: str) -> tuple[int, int, int, int]:
    try:
        x1, y1, x2, y2 = (int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("--bbox must use the format x1,y1,x2,y2") from error
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("--bbox requires x2>x1 and y2>y1")
    return x1, y1, x2, y2


def parse_points(value: str | None) -> list[tuple[int, int]]:
    if not value:
        return []
    try:
        return [parse_point(item) for item in value.split(";") if item.strip()]
    except argparse.ArgumentTypeError as error:
        raise argparse.ArgumentTypeError(
            "Point lists must use x,y;x,y (semicolon-separated)"
        ) from error


class CinematicDoFv8(CinematicDoFv7):
    """V7 tracking with a region-prompted semantic subject mask."""

    def __init__(
        self,
        init_point: tuple[int, int],
        focus_bbox: tuple[int, int, int, int],
        positive_points: list[tuple[int, int]] | None = None,
        negative_points: list[tuple[int, int]] | None = None,
        max_blur: float = 14.0,
        transition_sec: float = 1.5,
        model_path: Path | None = None,
        pipeline_name: str = "Smart DoF v8.0 — Semantic Focus Rendering",
    ) -> None:
        super().__init__(
            init_point=init_point,
            max_blur=max_blur,
            transition_sec=transition_sec,
            model_path=model_path,
            pipeline_name=pipeline_name,
        )
        self.initial_point = np.asarray(init_point, dtype=np.float32)
        self.focus_bbox = np.asarray(focus_bbox, dtype=np.float32)
        self.positive_points = positive_points or [init_point]
        self.negative_points = negative_points or []

        print("  V8 semantic focus region enabled")
        print(f"  Focus bounding box: {focus_bbox}")
        print(f"  Positive/negative prompts: {len(self.positive_points)}/{len(self.negative_points)}")

    @staticmethod
    def _translate_and_clip(
        point: tuple[int, int] | np.ndarray,
        delta: np.ndarray,
        width: int,
        height: int,
    ) -> list[int]:
        shifted = np.asarray(point, dtype=np.float32) + delta
        return [
            int(np.clip(round(float(shifted[0])), 0, width - 1)),
            int(np.clip(round(float(shifted[1])), 0, height - 1)),
        ]

    def _get_sam_mask(self, frame: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Segment the original subject region after translating prompts by tracking motion."""
        height, width = frame.shape[:2]
        delta = point.reshape(-1, 2)[0] - self.initial_point

        x1, y1, x2, y2 = self.focus_bbox + np.array([delta[0], delta[1], delta[0], delta[1]])
        bbox = [
            int(np.clip(round(float(x1)), 0, width - 1)),
            int(np.clip(round(float(y1)), 0, height - 1)),
            int(np.clip(round(float(x2)), 0, width - 1)),
            int(np.clip(round(float(y2)), 0, height - 1)),
        ]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return np.zeros((height, width), dtype=np.float32)

        positive = [self._translate_and_clip(item, delta, width, height) for item in self.positive_points]
        negative = [self._translate_and_clip(item, delta, width, height) for item in self.negative_points]
        prompts = positive + negative
        labels = [1] * len(positive) + [0] * len(negative)

        # Ultralytics SAM expects one batch of point prompts when a box prompt
        # is present, hence the additional outer list around prompts/labels.
        results = self.sam_model.predict(
            frame,
            bboxes=[bbox],
            points=[prompts],
            labels=[labels],
            verbose=False,
        )
        if not results or results[0].masks is None:
            return np.zeros((height, width), dtype=np.float32)

        mask = results[0].masks.data[0].cpu().numpy().astype(np.float32)
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        return (mask > 0.5).astype(np.float32)

    def _compute_blur_map(
        self,
        depth: np.ndarray,
        focus_depth: float,
        subject_mask: np.ndarray | None,
    ) -> np.ndarray:
        """Keep the selected object sharp and add restrained semantic falloff."""
        depth_difference = np.abs(depth - focus_depth)
        dead_zone = 0.025
        full_blur_difference = 0.20
        normalized = (depth_difference - dead_zone) / (full_blur_difference - dead_zone)
        depth_blur = self.max_blur * self._smoothstep(normalized)

        if subject_mask is None:
            return np.clip(depth_blur, 0.0, self.max_blur).astype(np.float32)

        binary_mask = (subject_mask >= 0.5).astype(np.uint8)
        protection = self._subject_protection(subject_mask, feather_px=11.0)
        distance_outside = cv2.distanceTransform(1 - binary_mask, cv2.DIST_L2, 5)

        # Monocular depth cannot always separate distant urban planes.  This
        # modest object-centred falloff ensures that an explicitly selected
        # building remains the visual focus without returning to V6's blanket
        # background blur.
        semantic_progress = self._smoothstep((distance_outside - 18.0) / 220.0)
        semantic_blur = self.max_blur * 0.52 * semantic_progress

        # In this model's relative-depth convention, higher values represent
        # near foreground for this shot.  Keep bonnet/road blur restrained even
        # when their depth differs greatly from the selected building.
        foreground = depth > focus_depth + dead_zone
        blur_map = np.maximum(depth_blur, semantic_blur)
        foreground_cap = self.max_blur * 0.34
        blur_map[foreground] = np.minimum(
            foreground_cap,
            depth_blur[foreground] * 0.45 + semantic_blur[foreground] * 0.35,
        )

        blur_map *= 1.0 - protection
        blur_map = cv2.GaussianBlur(blur_map.astype(np.float32), (0, 0), 1.1)
        blur_map[protection >= 0.98] = 0.0
        return np.clip(blur_map, 0.0, self.max_blur).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart DoF v8 — region-prompted semantic focus with depth-aware falloff"
    )
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--point", required=True, type=parse_point, help="Reliable tracking point inside target: x,y")
    parser.add_argument("--bbox", required=True, type=parse_bbox, help="First-frame focus box: x1,y1,x2,y2")
    parser.add_argument(
        "--positive-points",
        default=None,
        help="Optional positive SAM prompts: x,y;x,y (default: --point)",
    )
    parser.add_argument(
        "--negative-points",
        default=None,
        help="Optional negative SAM prompts: x,y;x,y",
    )
    parser.add_argument("--max-blur", type=float, default=14.0, help="Maximum background blur radius")
    parser.add_argument("--transition-sec", type=float, default=1.5, help="Deep-focus transition duration")
    parser.add_argument("--sam-model", type=Path, default=None, help="Optional MobileSAM model path")
    args = parser.parse_args()

    pipeline = CinematicDoFv8(
        init_point=args.point,
        focus_bbox=args.bbox,
        positive_points=parse_points(args.positive_points),
        negative_points=parse_points(args.negative_points),
        max_blur=args.max_blur,
        transition_sec=args.transition_sec,
        model_path=args.sam_model,
    )
    pipeline.process_video(args.input, args.output)


if __name__ == "__main__":
    main()
