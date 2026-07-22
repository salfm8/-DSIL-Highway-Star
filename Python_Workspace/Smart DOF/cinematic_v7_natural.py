"""
[DSIL] Highway Star — Smart DoF v7.0 (Natural Lens Rendering)
================================================================

Object- and depth-aware cinematic DoF renderer for natural focus control.

Design goals
------------
* Keep the selected subject reliably sharp without isolating it unnaturally.
* Apply restrained, depth-relative blur mainly to clearly separated depth planes.
* Prevent mask/depth flicker with temporal smoothing and optical-flow validation.
* When the target leaves the frame, return to deep focus over a configurable
  cinematic transition instead of reusing a stale blur map.

This renderer is a perceptual cinematic DoF approximation.  It does not claim
to be a calibrated physical thin-lens or Circle-of-Confusion simulation.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import SAM


class CinematicDoFv7:
    """Natural, temporally stable object-aware DoF pipeline."""

    def __init__(
        self,
        init_point: tuple[int, int],
        max_blur: float = 12.0,
        transition_sec: float = 1.5,
        model_path: Path | None = None,
        pipeline_name: str = "Smart DoF v7.0 — Natural Lens Rendering",
    ) -> None:
        self.init_point = init_point
        self.max_blur = max(0.0, float(max_blur))
        self.transition_sec = max(0.0, float(transition_sec))
        self.pipeline_name = pipeline_name

        script_dir = Path(__file__).resolve().parent
        self.model_path = model_path or script_dir.parent / "0720 Pitch" / "mobile_sam.pt"
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MobileSAM model not found: {self.model_path}")

        print("\n============================================================")
        print(f"  🎬 {self.pipeline_name}")
        print("============================================================")
        print(f"  Target point: {init_point}")
        print(f"  Maximum blur: {self.max_blur:.1f}px")
        print(f"  Deep-focus transition: {self.transition_sec:.2f}s")
        print("============================================================\n")

        print("  [1/3] Loading Depth Anything from the local model cache...")
        # Transformers 5.x looks for a processor_config.json when constructing
        # the generic pipeline.  This project cache contains the compatible
        # preprocessor_config.json instead, so load the processor/model directly
        # from the pinned local snapshot and never require a network request.
        try:
            depth_snapshot = snapshot_download(
                repo_id="LiheYoung/depth-anything-base-hf", local_files_only=True
            )
        except Exception as error:
            raise RuntimeError(
                "Depth Anything Base is not fully available in the local Hugging Face cache. "
                "Download the model once, then rerun V7 offline."
            ) from error
        self.depth_processor = AutoImageProcessor.from_pretrained(
            depth_snapshot, local_files_only=True
        )
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(
            depth_snapshot, local_files_only=True
        ).eval()
        print("  [2/3] Loading MobileSAM...")
        self.sam_model = SAM(str(self.model_path))
        print("  [3/3] Initializing optical-flow tracker...")

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.prev_gray: np.ndarray | None = None
        self.current_point = np.array([[init_point]], dtype=np.float32)
        self.previous_point = self.current_point.copy()

        self.prev_mask: np.ndarray | None = None
        self.prev_mask_point: np.ndarray | None = None
        self.focus_depth: float | None = None

        self.tracking_lost = False
        self.transition_frame = 0
        self.transition_frames = 1

        # Conservative temporal filters: current observations remain dominant
        # enough to follow a moving subject, while one-frame inference jitter is
        # suppressed.
        self.depth_ema = 0.72
        self.mask_current_weight = 0.76
        self.forward_backward_threshold = 1.5

    def _get_depth_map(self, rgb: np.ndarray, width: int, height: int) -> np.ndarray:
        inputs = self.depth_processor(images=Image.fromarray(rgb), return_tensors="pt")
        with torch.inference_mode():
            prediction = self.depth_model(**inputs).predicted_depth
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy().astype(np.float32)

        # Depth Anything produces relative depth.  Robust per-frame scaling
        # protects the blur ramp from a few extreme pixels while keeping the
        # existing relative-depth interpretation of the project.
        low, high = np.percentile(prediction, (2.0, 98.0))
        depth = (prediction - low) / max(high - low, 1e-6)
        depth = np.ascontiguousarray(depth, dtype=np.float32)

        # Keep RGB-aligned edges when the installed OpenCV build supports the
        # contrib ximgproc module.  The fallback still produces a valid render.
        if hasattr(cv2, "ximgproc"):
            guide = np.ascontiguousarray(rgb.astype(np.float32) / 255.0)
            depth = cv2.ximgproc.createGuidedFilter(guide, radius=8, eps=0.01).filter(depth)
        return np.clip(depth, 0.0, 1.0)

    def _track_point(self, gray: np.ndarray) -> bool:
        """Update the target point only when Lucas-Kanade passes an FB check."""
        if self.prev_gray is None:
            self.prev_gray = gray
            self.previous_point = self.current_point.copy()
            return True

        forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.current_point, None, **self.lk_params
        )
        if forward is None or status_forward is None or status_forward[0][0] != 1:
            self.prev_gray = gray
            return False

        backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.prev_gray, forward, None, **self.lk_params
        )
        self.prev_gray = gray
        if backward is None or status_backward is None or status_backward[0][0] != 1:
            return False

        fb_error = float(np.linalg.norm(backward - self.current_point))
        if fb_error > self.forward_backward_threshold:
            return False

        self.previous_point = self.current_point.copy()
        self.current_point = forward
        return True

    @staticmethod
    def _point_in_bounds(point: np.ndarray, width: int, height: int, margin: int = 15) -> bool:
        x, y = point.reshape(-1, 2)[0]
        return margin <= x < width - margin and margin <= y < height - margin

    def _get_sam_mask(self, frame: np.ndarray, point: np.ndarray) -> np.ndarray:
        x, y = (int(v) for v in point.reshape(-1, 2)[0])
        results = self.sam_model.predict(frame, points=[[x, y]], labels=[1], verbose=False)
        if not results or results[0].masks is None:
            return np.zeros(frame.shape[:2], dtype=np.float32)

        mask = results[0].masks.data[0].cpu().numpy().astype(np.float32)
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        return (mask > 0.5).astype(np.float32)

    def _stabilize_mask(self, current_mask: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Blend the new mask with a translation-warped prior mask."""
        if self.prev_mask is None or self.prev_mask_point is None:
            stable = current_mask
        else:
            dx, dy = (point.reshape(-1, 2)[0] - self.prev_mask_point.reshape(-1, 2)[0]).tolist()
            transform = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
            warped_prior = cv2.warpAffine(
                self.prev_mask,
                transform,
                (current_mask.shape[1], current_mask.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            stable = (
                current_mask * self.mask_current_weight
                + warped_prior * (1.0 - self.mask_current_weight)
            )

        stable = np.clip(stable, 0.0, 1.0)
        self.prev_mask = stable
        self.prev_mask_point = point.copy()
        return stable

    @staticmethod
    def _subject_depth(depth: np.ndarray, mask: np.ndarray) -> float | None:
        binary_mask = (mask >= 0.5).astype(np.uint8)
        if np.count_nonzero(binary_mask) < 64:
            return None

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        interior = cv2.erode(binary_mask, kernel, iterations=1)
        samples = depth[interior > 0]
        if samples.size < 64:
            samples = depth[binary_mask > 0]
        if samples.size < 64:
            return None
        return float(np.median(samples))

    @staticmethod
    def _smoothstep(value: np.ndarray) -> np.ndarray:
        value = np.clip(value, 0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    @staticmethod
    def _subject_protection(mask: np.ndarray, feather_px: float = 9.0) -> np.ndarray:
        binary_mask = (mask >= 0.5).astype(np.uint8)
        if not np.any(binary_mask):
            return np.zeros(mask.shape, dtype=np.float32)
        distance_inside = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        return np.clip(distance_inside / feather_px, 0.0, 1.0).astype(np.float32)

    def _compute_blur_map(
        self,
        depth: np.ndarray,
        focus_depth: float,
        subject_mask: np.ndarray | None,
    ) -> np.ndarray:
        """Map relative-depth separation to restrained, smooth blur radii."""
        depth_difference = np.abs(depth - focus_depth)

        # Relative monocular depth is not calibrated.  A dead zone keeps nearby
        # depth planes clear; a broad smoothstep ramp avoids the V6 cut-out look.
        dead_zone = 0.035
        full_blur_difference = 0.29
        normalized = (depth_difference - dead_zone) / (full_blur_difference - dead_zone)
        blur_map = self.max_blur * self._smoothstep(normalized)

        if subject_mask is not None:
            protection = self._subject_protection(subject_mask)
            blur_map *= 1.0 - protection

        # A light final smoothing removes quantization from the depth estimate
        # without bleeding blur deeply into the protected subject.
        blur_map = cv2.GaussianBlur(blur_map.astype(np.float32), (0, 0), 1.2)
        if subject_mask is not None:
            protection = self._subject_protection(subject_mask)
            blur_map[protection >= 0.98] = 0.0
        return np.clip(blur_map, 0.0, self.max_blur).astype(np.float32)

    @staticmethod
    def _render_variable_blur(image: np.ndarray, blur_map: np.ndarray) -> np.ndarray:
        max_radius = int(math.ceil(float(np.max(blur_map))))
        if max_radius <= 0:
            return image.copy()

        levels = [image.astype(np.float32)]
        radii = [0]
        radius = 1
        while radius <= max_radius + 1:
            kernel_size = radius * 2 + 1
            levels.append(cv2.GaussianBlur(image, (kernel_size, kernel_size), radius * 0.55).astype(np.float32))
            radii.append(radius)
            radius += 1 if radius < 5 else max(1, int(radius * 0.35))

        radii_array = np.asarray(radii, dtype=np.float32)
        indices = np.searchsorted(radii_array, blur_map)
        indices = np.clip(indices, 1, len(radii) - 1)
        result = np.empty_like(image, dtype=np.float32)

        for index in range(1, len(radii)):
            pixel_mask = indices == index
            if not np.any(pixel_mask):
                continue
            low_radius = radii_array[index - 1]
            high_radius = radii_array[index]
            high_weight = (blur_map[pixel_mask] - low_radius) / (high_radius - low_radius + 1e-6)
            high_weight = np.clip(high_weight, 0.0, 1.0)
            result[pixel_mask] = (
                levels[index - 1][pixel_mask] * (1.0 - high_weight[:, None])
                + levels[index][pixel_mask] * high_weight[:, None]
            )
        return np.clip(result, 0, 255).astype(np.uint8)

    def _begin_focus_pull(self, fps: float) -> None:
        if self.tracking_lost:
            return
        self.tracking_lost = True
        self.transition_frame = 0
        self.transition_frames = max(1, int(round(self.transition_sec * fps)))
        print("\n  ⚠️ Target tracking lost — starting natural deep-focus transition")

    def process_video(self, input_path: str, output_path: str) -> None:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open input video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
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

            rendered: np.ndarray
            if not self.tracking_lost:
                tracking_ok = self._track_point(gray)
                if not tracking_ok or not self._point_in_bounds(self.current_point, width, height):
                    self._begin_focus_pull(fps)
                else:
                    raw_mask = self._get_sam_mask(frame, self.current_point)
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
                        rendered = self._render_variable_blur(frame, blur_map)
                        output.write(rendered)
                        sys.stdout.write(
                            f"\r  🎥 [{frame_index}/{total_frames}] "
                            f"{frame_index * 100 // max(total_frames, 1)}% | Natural subject focus"
                        )
                        sys.stdout.flush()
                        continue

            # Target is gone.  Recompute the depth-dependent blur map for this
            # current frame, then ease its strength down to zero.
            if self.focus_depth is None:
                rendered = frame
            else:
                self.transition_frame += 1
                progress = min(1.0, self.transition_frame / self.transition_frames)
                strength = 0.5 * (1.0 + math.cos(math.pi * progress))
                current_blur_map = self._compute_blur_map(depth, self.focus_depth, None)
                rendered = self._render_variable_blur(frame, current_blur_map * strength)

            output.write(rendered)
            sys.stdout.write(
                f"\r  🎥 [{frame_index}/{total_frames}] "
                f"{frame_index * 100 // max(total_frames, 1)}% | Natural focus pull"
            )
            sys.stdout.flush()

        cap.release()
        output.release()
        elapsed = time.time() - started_at
        print(f"\n\n  ✅ {self.pipeline_name} rendering complete in {elapsed:.1f}s")
        print(f"  Output: {output_path}")


def parse_point(value: str) -> tuple[int, int]:
    try:
        x, y = (int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("--point must use the format x,y") from error
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart DoF v7 — Natural object- and depth-aware cinematic DoF"
    )
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--point", required=True, type=parse_point, help="Initial target point: x,y")
    parser.add_argument("--max-blur", type=float, default=12.0, help="Maximum blur radius in pixels")
    parser.add_argument(
        "--transition-sec", type=float, default=1.5, help="Deep-focus transition duration in seconds"
    )
    parser.add_argument("--sam-model", type=Path, default=None, help="Optional MobileSAM model path")
    args = parser.parse_args()

    pipeline = CinematicDoFv7(
        init_point=args.point,
        max_blur=args.max_blur,
        transition_sec=args.transition_sec,
        model_path=args.sam_model,
    )
    pipeline.process_video(args.input, args.output)


if __name__ == "__main__":
    main()
