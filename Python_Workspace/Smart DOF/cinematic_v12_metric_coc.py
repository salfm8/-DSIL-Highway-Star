"""Smart DoF V12 — metric linear depth and thin-lens Circle of Confusion.

V12 fixes the central limitation of V11: V11 subtracts normalized relative
inverse-depth values, which compresses distant geometry.  V12 instead uses one
of two explicitly separated depth paths:

1. ``metric`` (default): Depth Anything V2 Metric Outdoor predicts linear
   distance in metres directly.
2. ``inverse-calibrated`` (fallback): relative inverse depth is mapped to
   pseudo-metric linear depth with a two-anchor shifted-inverse transform,
   Z = 1 / (a*d + b).

Only the resulting linear depth Z is passed to the thin-lens CoC equation.
MobileSAM remains restricted to estimating the target focus distance.
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from cinematic_v7_natural import parse_point
from cinematic_v8_semantic_focus import parse_bbox, parse_points
from cinematic_v11_natural import CinematicDoFv11


class CinematicDoFv12(CinematicDoFv11):
    """Parallel-plane DoF rendered from linear metres and a virtual thin lens."""

    def __init__(
        self,
        *args,
        depth_mode: str = "metric",
        metric_model_path: Path | None = None,
        near_anchor_m: float = 1.5,
        far_anchor_m: float = 80.0,
        focal_length_mm: float = 135.0,
        f_number: float = 2.0,
        sensor_width_mm: float = 36.0,
        coc_gain: float = 1.0,
        depth_current_weight: float = 0.78,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            depth_current_weight=depth_current_weight,
            **kwargs,
        )
        self.pipeline_name = "Smart DoF v12.0 — Metric Linear Depth + Thin-Lens CoC"
        self.depth_mode = depth_mode
        self.near_anchor_m = float(near_anchor_m)
        self.far_anchor_m = float(far_anchor_m)
        self.focal_length_mm = float(focal_length_mm)
        self.f_number = float(f_number)
        self.sensor_width_mm = float(sensor_width_mm)
        self.coc_gain = float(coc_gain)

        if self.depth_mode not in {"metric", "inverse-calibrated"}:
            raise ValueError("--depth-mode must be metric or inverse-calibrated")
        if self.near_anchor_m <= 0.0:
            raise ValueError("--near-anchor-m must be positive")
        if self.far_anchor_m <= self.near_anchor_m:
            raise ValueError("--far-anchor-m must be greater than --near-anchor-m")
        if self.focal_length_mm <= 0.0:
            raise ValueError("--focal-length-mm must be positive")
        if self.f_number <= 0.0:
            raise ValueError("--f-number must be positive")
        if self.sensor_width_mm <= 0.0:
            raise ValueError("--sensor-width-mm must be positive")
        if self.coc_gain <= 0.0:
            raise ValueError("--coc-gain must be positive")

        self.inverse_low_ema: float | None = None
        self.inverse_high_ema: float | None = None
        self.inverse_range_ema = 0.96
        self.previous_linear_depth: np.ndarray | None = None
        self.previous_linear_gray: np.ndarray | None = None

        # Scene-anchored focus-plane tracking.  Metric monocular depth may keep
        # a roadside target at nearly the same predicted distance while the
        # camera approaches it.  The target's optical expansion supplies the
        # missing ego-motion scale: projected size is approximately inversely
        # proportional to camera-to-target distance.
        self.focus_feature_points: np.ndarray | None = None
        self.focus_feature_gray: np.ndarray | None = None
        self.dynamic_focus_mask: np.ndarray | None = None
        self.focus_scale_step_ema = 1.0
        self.scene_focus_distance_m: float | None = None
        self.focus_reference_affine = np.eye(3, dtype=np.float32)
        self.reference_scene_depth: np.ndarray | None = None
        self.reference_scene_focus_m: float | None = None
        self.previous_scene_focus_m: float | None = None
        self.last_camera_advance_m = 0.0
        self.focus_feature_lk = dict(
            winSize=(25, 25),
            maxLevel=4,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                40,
                0.01,
            ),
        )

        script_dir = Path(__file__).resolve().parent
        self.metric_model_path = metric_model_path or (
            script_dir
            / "models"
            / "depth-anything-v2-metric-outdoor-small"
        )

        if self.depth_mode == "metric":
            if not self.metric_model_path.is_dir():
                raise FileNotFoundError(
                    "Metric depth model not found: "
                    f"{self.metric_model_path}. Download "
                    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf."
                )

            # The V11 parent loads the V1 relative model. Replace it completely
            # so V12 cannot accidentally run the old percentile-normalized path.
            del self.depth_model
            del self.depth_processor
            gc.collect()
            print("  Loading Depth Anything V2 Metric Outdoor Small...")
            self.depth_processor = AutoImageProcessor.from_pretrained(
                self.metric_model_path,
                local_files_only=True,
            )
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(
                self.metric_model_path,
                local_files_only=True,
            ).eval()
            self.model_max_depth_m = float(
                getattr(self.depth_model.config, "max_depth", self.far_anchor_m)
            )
        else:
            # The inherited Depth Anything V1 relative model is retained only
            # for the explicitly selected calibrated-inverse fallback.
            self.model_max_depth_m = self.far_anchor_m

        print("  V12 linear-depth optics enabled")
        print(f"  Depth mode: {self.depth_mode}")
        if self.depth_mode == "metric":
            print(f"  Metric model range: up to {self.model_max_depth_m:.1f}m")
        else:
            print(
                "  Relative inverse-depth calibration: "
                f"{self.near_anchor_m:.2f}m .. {self.far_anchor_m:.1f}m"
            )
        print(
            "  Virtual lens: "
            f"{self.focal_length_mm:.1f}mm, f/{self.f_number:.1f}, "
            f"{self.sensor_width_mm:.1f}mm sensor"
        )
        print(f"  Art-directed CoC rendering gain: {self.coc_gain:.2f}x")
        print("  Scene-anchored moving focus plane: optical-expansion tracking")
        print("  SAM usage: target focus distance estimation only")

    def _predict_model_depth(self, rgb: np.ndarray, width: int, height: int) -> np.ndarray:
        inputs = self.depth_processor(
            images=Image.fromarray(rgb),
            return_tensors="pt",
        )
        with torch.inference_mode():
            prediction = self.depth_model(**inputs).predicted_depth
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        return prediction.cpu().numpy().astype(np.float32)

    @staticmethod
    def linearize_inverse_depth(
        inverse_depth: np.ndarray,
        near_m: float,
        far_m: float,
        inverse_near: float = 1.0,
        inverse_far: float = 0.0,
    ) -> np.ndarray:
        """Map affine inverse depth to linear metres using two distance anchors.

        Solve rho = a*d + b for:
            d=inverse_near -> rho=1/near_m
            d=inverse_far  -> rho=1/far_m
        and return Z=1/rho.

        This fallback is calibrated pseudo-metric depth, not sensor-ground-truth
        metric depth.  The default V12 path uses the metric outdoor model.
        """
        if near_m <= 0.0 or far_m <= near_m:
            raise ValueError("Linear-depth anchors require 0 < near_m < far_m")
        if abs(inverse_near - inverse_far) < 1e-8:
            raise ValueError("Inverse-depth anchors must be distinct")

        rho_near = 1.0 / near_m
        rho_far = 1.0 / far_m
        a = (rho_near - rho_far) / (inverse_near - inverse_far)
        b = rho_far - a * inverse_far
        reciprocal_depth = a * inverse_depth.astype(np.float32) + b
        reciprocal_depth = np.maximum(reciprocal_depth, 1.0 / far_m)
        return np.clip(1.0 / reciprocal_depth, near_m, far_m).astype(np.float32)

    def _normalize_relative_inverse_depth(self, raw: np.ndarray) -> np.ndarray:
        """Preserve more distant detail than V11's 2/98 percentile clipping."""
        low = float(np.percentile(raw, 0.1))
        high = float(np.percentile(raw, 99.9))
        if self.inverse_low_ema is None or self.inverse_high_ema is None:
            self.inverse_low_ema = low
            self.inverse_high_ema = high
        else:
            self.inverse_low_ema = (
                self.inverse_low_ema * self.inverse_range_ema
                + low * (1.0 - self.inverse_range_ema)
            )
            self.inverse_high_ema = (
                self.inverse_high_ema * self.inverse_range_ema
                + high * (1.0 - self.inverse_range_ema)
            )
        scale = max(self.inverse_high_ema - self.inverse_low_ema, 1e-6)
        return np.clip((raw - self.inverse_low_ema) / scale, 0.0, 1.0).astype(
            np.float32
        )

    def _stabilize_linear_depth(
        self,
        current_depth_m: np.ndarray,
        current_gray: np.ndarray,
    ) -> np.ndarray:
        """Flow-warp and confidence-blend linear metric depth, in metres."""
        if self.previous_linear_depth is None or self.previous_linear_gray is None:
            stable = current_depth_m.astype(np.float32)
        else:
            backward_flow = cv2.calcOpticalFlowFarneback(
                current_gray,
                self.previous_linear_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=21,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            height, width = current_depth_m.shape
            grid_x, grid_y = np.meshgrid(
                np.arange(width, dtype=np.float32),
                np.arange(height, dtype=np.float32),
            )
            map_x = grid_x + backward_flow[..., 0]
            map_y = grid_y + backward_flow[..., 1]
            warped_depth = cv2.remap(
                self.previous_linear_depth,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            warped_gray = cv2.remap(
                self.previous_linear_gray,
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
            stable = (
                current_depth_m * (1.0 - prior_weight)
                + warped_depth * prior_weight
            )

        stable = np.clip(
            stable,
            self.near_anchor_m if self.depth_mode == "inverse-calibrated" else 0.1,
            self.model_max_depth_m,
        ).astype(np.float32)
        self.previous_linear_depth = stable.copy()
        self.previous_linear_gray = current_gray.copy()
        return stable

    def _get_depth_map(self, rgb: np.ndarray, width: int, height: int) -> np.ndarray:
        model_output = self._predict_model_depth(rgb, width, height)
        if self.depth_mode == "metric":
            linear_depth_m = np.clip(
                model_output,
                0.1,
                self.model_max_depth_m,
            ).astype(np.float32)
        else:
            normalized_inverse = self._normalize_relative_inverse_depth(model_output)
            linear_depth_m = self.linearize_inverse_depth(
                normalized_inverse,
                near_m=self.near_anchor_m,
                far_m=self.far_anchor_m,
            )

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return self._stabilize_linear_depth(linear_depth_m, gray)

    def _compute_blur_map(
        self,
        depth_m: np.ndarray,
        focus_distance_m: float,
    ) -> np.ndarray:
        """Apply the thin-lens Circle-of-Confusion equation in metric space.

        With focal length f, f-number N, object distance Z and focus distance S:

            CoC_diameter_mm =
                (f^2 / N) * |Z - S| / (Z * (S - f))

        Z and S are converted from metres to millimetres.  The sensor-space CoC
        diameter is converted to an image-space radius using sensor width.
        """
        f_mm = self.focal_length_mm
        z_mm = np.maximum(depth_m.astype(np.float32) * 1000.0, f_mm + 1e-3)
        focus_mm = max(float(focus_distance_m) * 1000.0, f_mm + 1e-3)

        coc_diameter_mm = (
            (f_mm * f_mm / self.f_number)
            * np.abs(z_mm - focus_mm)
            / np.maximum(z_mm * (focus_mm - f_mm), 1e-6)
        )
        pixels_per_mm = depth_m.shape[1] / self.sensor_width_mm
        physical_coc_radius_px = 0.5 * coc_diameter_mm * pixels_per_mm

        # A distant focus plane produces sub-pixel CoC values at this video's
        # 854px width, even when metre ordering is correct.  coc_gain is an
        # explicit user-facing DoF-strength control: it preserves zero at the
        # focal plane and preserves the complete metric CoC ordering, while
        # scaling the radii into a visibly renderable range.  A value of 1.0 is
        # the unmodified thin-lens result.
        coc_radius_px = physical_coc_radius_px * self.coc_gain
        return np.clip(coc_radius_px, 0.0, self.max_blur).astype(np.float32)

    def _depth_visualization(self, depth_m: np.ndarray) -> np.ndarray:
        """Render a linear-metre map: warm is near, cool is far."""
        display_near = max(
            0.1,
            self.near_anchor_m if self.depth_mode == "inverse-calibrated" else 2.0,
        )
        display_far = self.model_max_depth_m
        normalized_z = np.clip(
            (depth_m - display_near) / max(display_far - display_near, 1e-6),
            0.0,
            1.0,
        )
        near_is_hot = ((1.0 - normalized_z) * 255.0).astype(np.uint8)
        return cv2.applyColorMap(near_is_hot, cv2.COLORMAP_TURBO)

    def _coc_visualization(self, coc_radius_px: np.ndarray) -> np.ndarray:
        """Render the effective CoC radius: cool is sharp, warm is blurred."""
        normalized = np.clip(
            coc_radius_px / max(self.max_blur, 1e-6),
            0.0,
            1.0,
        )
        coc_u8 = np.round(normalized * 255.0).astype(np.uint8)
        return cv2.applyColorMap(coc_u8, cv2.COLORMAP_TURBO)

    @staticmethod
    def _feature_mask(mask: np.ndarray) -> np.ndarray:
        binary = (mask >= 0.35).astype(np.uint8) * 255
        return cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )

    def _detect_focus_features(
        self,
        gray: np.ndarray,
        mask: np.ndarray,
        max_corners: int = 220,
    ) -> np.ndarray | None:
        return cv2.goodFeaturesToTrack(
            gray,
            mask=self._feature_mask(mask),
            maxCorners=max_corners,
            qualityLevel=0.01,
            minDistance=5,
            blockSize=7,
        )

    def _propagate_scene_focus_mask(
        self,
        gray: np.ndarray,
        sam_mask: np.ndarray,
    ) -> tuple[np.ndarray, float | None]:
        """Track target translation and scale, returning its current mask and scale step."""
        if (
            self.focus_feature_gray is None
            or self.focus_feature_points is None
            or self.dynamic_focus_mask is None
        ):
            self.dynamic_focus_mask = sam_mask.astype(np.float32).copy()
            self.focus_feature_points = self._detect_focus_features(
                gray,
                self.dynamic_focus_mask,
            )
            self.focus_feature_gray = gray.copy()
            return self.dynamic_focus_mask, None

        forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(
            self.focus_feature_gray,
            gray,
            self.focus_feature_points,
            None,
            **self.focus_feature_lk,
        )
        if forward is None or status_forward is None:
            self.focus_feature_gray = gray.copy()
            self.focus_feature_points = self._detect_focus_features(
                gray,
                sam_mask,
            )
            self.dynamic_focus_mask = sam_mask.astype(np.float32).copy()
            return self.dynamic_focus_mask, None

        backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            self.focus_feature_gray,
            forward,
            None,
            **self.focus_feature_lk,
        )
        if backward is None or status_backward is None:
            self.focus_feature_gray = gray.copy()
            self.focus_feature_points = self._detect_focus_features(
                gray,
                sam_mask,
            )
            self.dynamic_focus_mask = sam_mask.astype(np.float32).copy()
            return self.dynamic_focus_mask, None

        fb_error = np.linalg.norm(
            backward - self.focus_feature_points,
            axis=2,
        ).reshape(-1)
        valid = (
            (status_forward.reshape(-1) == 1)
            & (status_backward.reshape(-1) == 1)
            & (fb_error < 1.5)
        )
        old_points = self.focus_feature_points[valid].reshape(-1, 2)
        new_points = forward[valid].reshape(-1, 2)
        if old_points.shape[0] < 8:
            self.focus_feature_gray = gray.copy()
            self.focus_feature_points = self._detect_focus_features(
                gray,
                sam_mask,
            )
            self.dynamic_focus_mask = sam_mask.astype(np.float32).copy()
            return self.dynamic_focus_mask, None

        affine, inliers = cv2.estimateAffinePartial2D(
            old_points,
            new_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
            maxIters=2000,
            confidence=0.99,
        )
        if affine is None:
            self.focus_feature_gray = gray.copy()
            self.focus_feature_points = new_points.reshape(-1, 1, 2)
            return self.dynamic_focus_mask, None

        raw_scale_step = float(
            np.sqrt(abs(np.linalg.det(affine[:, :2])))
        )
        # This mode is used while the vehicle approaches the selected roadside
        # target.  Reject one-frame shrinkage jitter; the focus pull begins once
        # the target exits instead of modelling a receding phase.
        raw_scale_step = float(np.clip(raw_scale_step, 1.0, 1.06))
        self.focus_scale_step_ema = (
            self.focus_scale_step_ema * 0.20 + raw_scale_step * 0.80
        )

        # Accumulate only isotropic expansion and centroid translation. Local
        # perspective/rotation inside the building mask must not rotate the
        # entire parallel focal plane.
        old_centroid = np.median(old_points, axis=0)
        new_centroid = np.median(new_points, axis=0)
        affine_step = np.eye(3, dtype=np.float32)
        affine_step[0, 0] = self.focus_scale_step_ema
        affine_step[1, 1] = self.focus_scale_step_ema
        affine_step[0, 2] = (
            new_centroid[0] - self.focus_scale_step_ema * old_centroid[0]
        )
        affine_step[1, 2] = (
            new_centroid[1] - self.focus_scale_step_ema * old_centroid[1]
        )
        self.focus_reference_affine = (
            affine_step @ self.focus_reference_affine
        ).astype(np.float32)

        height, width = gray.shape
        self.dynamic_focus_mask = cv2.warpAffine(
            self.dynamic_focus_mask,
            affine,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        self.dynamic_focus_mask = np.clip(
            self.dynamic_focus_mask,
            0.0,
            1.0,
        ).astype(np.float32)

        if inliers is not None:
            tracked = new_points[inliers.reshape(-1) == 1]
        else:
            tracked = new_points
        self.focus_feature_points = tracked.reshape(-1, 1, 2)

        # Replenish features inside the flow-propagated target as older points
        # leave the frame.  The SAM mask is not used to force sharp output.
        if self.focus_feature_points.shape[0] < 80:
            replenish_mask = self._feature_mask(self.dynamic_focus_mask)
            for x, y in self.focus_feature_points.reshape(-1, 2):
                cv2.circle(
                    replenish_mask,
                    (int(round(x)), int(round(y))),
                    6,
                    0,
                    -1,
                )
            additions = cv2.goodFeaturesToTrack(
                gray,
                mask=replenish_mask,
                maxCorners=max(0, 180 - self.focus_feature_points.shape[0]),
                qualityLevel=0.01,
                minDistance=5,
                blockSize=7,
            )
            if additions is not None:
                self.focus_feature_points = np.concatenate(
                    (self.focus_feature_points, additions),
                    axis=0,
                )

        self.focus_feature_gray = gray.copy()
        return self.dynamic_focus_mask, self.focus_scale_step_ema

    def _update_scene_focus_distance(
        self,
        metric_observation_m: float,
        scale_step: float | None,
    ) -> float:
        if self.scene_focus_distance_m is None:
            self.scene_focus_distance_m = float(metric_observation_m)
        elif scale_step is not None:
            self.scene_focus_distance_m /= max(scale_step, 1e-6)
        else:
            # Only use metric depth as a slow fallback if geometric tracking
            # temporarily fails.
            self.scene_focus_distance_m = (
                self.scene_focus_distance_m * 0.95
                + float(metric_observation_m) * 0.05
            )
        self.scene_focus_distance_m = float(
            np.clip(
                self.scene_focus_distance_m,
                0.5,
                self.model_max_depth_m,
            )
        )
        return self.scene_focus_distance_m

    def _update_scene_depth(
        self,
        raw_depth_m: np.ndarray,
        gray: np.ndarray,
        focus_distance_m: float,
        focus_mask: np.ndarray | None,
    ) -> np.ndarray:
        """Propagate a camera-relative scene depth instead of rescaling each frame.

        The monocular metric model provides the initial spatial depth ordering,
        while optical expansion of the selected scene target estimates camera
        advance.  Reference-frame 3D points are reprojected through a pinhole
        camera after that forward translation.  This changes both their depth
        and screen position, so the equal-depth band on the road moves toward
        the camera instead of remaining fixed in image coordinates.

        The current monocular estimate fills newly revealed regions after a
        robust additive alignment.  A target-plane offset removes small drift,
        but target membership is never used to alter any pixel's CoC.
        """
        raw = raw_depth_m.astype(np.float32)
        if (
            self.reference_scene_depth is None
            or self.reference_scene_focus_m is None
        ):
            scene = raw.copy()
            if focus_mask is not None:
                valid_focus = focus_mask >= 0.45
                if np.count_nonzero(valid_focus) >= 32:
                    scene += focus_distance_m - float(
                        np.median(scene[valid_focus])
                    )
            scene = np.clip(
                scene,
                0.1,
                self.model_max_depth_m,
            ).astype(np.float32)
            self.reference_scene_depth = scene.copy()
            self.reference_scene_focus_m = float(focus_distance_m)
        else:
            frame_advance_m = float(
                np.clip(
                    (
                        self.previous_scene_focus_m - focus_distance_m
                        if self.previous_scene_focus_m is not None
                        else 0.0
                    ),
                    0.0,
                    1.5,
                )
            )
            if frame_advance_m > 1e-4:
                self.last_camera_advance_m = (
                    self.last_camera_advance_m * 0.75
                    + frame_advance_m * 0.25
                )

            cumulative_advance_m = float(
                np.clip(
                    self.reference_scene_focus_m - focus_distance_m,
                    0.0,
                    self.model_max_depth_m - 0.1,
                )
            )
            height, width = raw.shape
            grid_x, grid_y = np.meshgrid(
                np.arange(width, dtype=np.float32),
                np.arange(height, dtype=np.float32),
            )
            # The cumulative target affine is the observed reference-to-current
            # projection on the focal plane. Invert a depth-interpolated affine
            # at every current pixel to sample the smooth reference depth. This
            # avoids holes and stair-stepping from forward splatting.
            reference_z = self.reference_scene_depth
            target_expansion = (
                self.reference_scene_focus_m / max(focus_distance_m, 0.1)
            )
            parallax_denominator = max(target_expansion - 1.0, 1e-6)
            transform = self.focus_reference_affine
            target_scale = max(
                0.5 * (float(transform[0, 0]) + float(transform[1, 1])),
                1e-4,
            )
            target_tx = float(transform[0, 2])
            target_ty = float(transform[1, 2])

            relative_parallax = np.ones((height, width), dtype=np.float32)
            map_x = grid_x.copy()
            map_y = grid_y.copy()
            sampled_reference = reference_z
            for _ in range(6):
                interpolated_scale = (
                    1.0 + relative_parallax * (target_scale - 1.0)
                )
                map_x = (
                    grid_x - relative_parallax * target_tx
                ) / np.maximum(interpolated_scale, 1e-4)
                map_y = (
                    grid_y - relative_parallax * target_ty
                ) / np.maximum(interpolated_scale, 1e-4)
                sampled_reference = cv2.remap(
                    reference_z,
                    map_x,
                    map_y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                sampled_remaining = np.maximum(
                    sampled_reference - cumulative_advance_m,
                    0.1,
                )
                depth_expansion = (
                    sampled_reference / sampled_remaining
                )
                updated_parallax = np.clip(
                    (depth_expansion - 1.0) / parallax_denominator,
                    0.0,
                    4.0,
                )
                relative_parallax = (
                    relative_parallax * 0.40 + updated_parallax * 0.60
                )

            predicted_scene = sampled_reference - cumulative_advance_m
            valid_projection = (
                (map_x >= 0.0)
                & (map_x <= width - 1.0)
                & (map_y >= 0.0)
                & (map_y <= height - 1.0)
                & (sampled_reference > cumulative_advance_m + 0.1)
            )

            if focus_mask is not None:
                valid_focus = focus_mask >= 0.45
                if np.count_nonzero(valid_focus) >= 32:
                    raw_offset = focus_distance_m - float(
                        np.median(raw[valid_focus])
                    )
                else:
                    raw_offset = float(
                        np.median(predicted_scene[valid_projection] - raw[valid_projection])
                    )
            elif np.any(valid_projection):
                raw_offset = float(
                    np.median(predicted_scene[valid_projection] - raw[valid_projection])
                )
            else:
                raw_offset = 0.0
            aligned_raw = raw + raw_offset

            # Reference reprojection controls established scene geometry.
            # Current monocular depth supplies only genuinely new pixels;
            # blending it into valid projection would pull the focal locus back
            # toward the model's temporally compressed distance estimate.
            scene = np.where(
                valid_projection,
                predicted_scene,
                aligned_raw,
            )

        scene = np.clip(
            scene,
            0.1,
            self.model_max_depth_m,
        ).astype(np.float32)
        self.previous_scene_focus_m = float(focus_distance_m)
        return scene

    def process_video(
        self,
        input_path: str,
        output_path: str,
        depth_output_path: str | None = None,
        coc_output_path: str | None = None,
    ) -> None:
        """Render final DoF, linear depth, and the effective physical CoC map."""
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

        def optional_writer(path: str | None, label: str) -> cv2.VideoWriter | None:
            if not path:
                return None
            writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create {label} video: {path}")
            return writer

        try:
            depth_output = optional_writer(depth_output_path, "depth")
            coc_output = optional_writer(coc_output_path, "CoC")
        except Exception:
            cap.release()
            output.release()
            raise

        frame_index = 0
        started_at = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            raw_depth_m = self._get_depth_map(rgb, width, height)
            depth_m = raw_depth_m

            effective_coc = np.zeros((height, width), dtype=np.float32)
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

                    moving_mask, scale_step = self._propagate_scene_focus_mask(
                        gray,
                        stable_mask,
                    )

                    # The propagated mask is used only to observe target scale
                    # and initial metric distance. It never protects pixels in
                    # the CoC map or final renderer.
                    observed_focus_m = self._subject_depth(
                        raw_depth_m,
                        moving_mask,
                    )
                    if observed_focus_m is None:
                        self._begin_focus_pull(fps)
                    else:
                        self.focus_depth = self._update_scene_focus_distance(
                            observed_focus_m,
                            scale_step,
                        )
                        depth_m = self._update_scene_depth(
                            raw_depth_m,
                            gray,
                            self.focus_depth,
                            moving_mask,
                        )
                        effective_coc = self._compute_blur_map(
                            depth_m,
                            self.focus_depth,
                        )
                        rendered = self._render_variable_blur(frame, effective_coc)
                        if depth_output is not None:
                            depth_output.write(
                                self._depth_visualization(depth_m)
                            )
                        output.write(rendered)
                        if coc_output is not None:
                            coc_output.write(
                                self._coc_visualization(effective_coc)
                            )
                        sys.stdout.write(
                            f"\r  🎥 [{frame_index}/{total_frames}] "
                            f"{frame_index * 100 // max(total_frames, 1)}%"
                            f" | Scene focus {self.focus_depth:.1f}m"
                        )
                        sys.stdout.flush()
                        continue

            # Target loss: recompute CoC from every new metric-depth frame and
            # reduce its effective radius with the same cosine focus pull used
            # by the final renderer. The diagnostic CoC video therefore shows
            # exactly what was applied, including the transition to zero.
            if self.focus_depth is None:
                rendered = frame
            else:
                # Continue the last measured ego-motion during the short focus
                # pull so the abandoned focal plane remains scene-anchored
                # instead of freezing to the screen at the loss frame.
                self.focus_depth = max(
                    0.5,
                    self.focus_depth - self.last_camera_advance_m,
                )
                depth_m = self._update_scene_depth(
                    raw_depth_m,
                    gray,
                    self.focus_depth,
                    None,
                )
                self.transition_frame += 1
                progress = min(1.0, self.transition_frame / self.transition_frames)
                strength = 0.5 * (1.0 + math.cos(math.pi * progress))
                effective_coc = (
                    self._compute_blur_map(depth_m, self.focus_depth) * strength
                )
                rendered = self._render_variable_blur(frame, effective_coc)

            if depth_output is not None:
                depth_output.write(self._depth_visualization(depth_m))
            output.write(rendered)
            if coc_output is not None:
                coc_output.write(self._coc_visualization(effective_coc))
            sys.stdout.write(
                f"\r  🎥 [{frame_index}/{total_frames}] "
                f"{frame_index * 100 // max(total_frames, 1)}%"
                " | Metric deep-focus pull"
            )
            sys.stdout.flush()

        cap.release()
        output.release()
        if depth_output is not None:
            depth_output.release()
        if coc_output is not None:
            coc_output.release()

        elapsed = time.time() - started_at
        print(f"\n\n  ✅ {self.pipeline_name} rendering complete in {elapsed:.1f}s")
        print(f"  Output: {output_path}")
        if depth_output_path:
            print(f"  Linear metric depth: {depth_output_path}")
        if coc_output_path:
            print(f"  Effective CoC map: {coc_output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart DoF V12 — linear metric depth and thin-lens CoC"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--depth-output", default=None)
    parser.add_argument(
        "--coc-output",
        default=None,
        help="Optional effective thin-lens CoC radius visualization MP4",
    )
    parser.add_argument("--point", required=True, type=parse_point)
    parser.add_argument("--bbox", required=True, type=parse_bbox)
    parser.add_argument("--positive-points", default=None)
    parser.add_argument("--negative-points", default=None)
    parser.add_argument(
        "--depth-mode",
        choices=("metric", "inverse-calibrated"),
        default="metric",
        help="Use true metric-model output or calibrated relative inverse depth",
    )
    parser.add_argument(
        "--metric-model",
        type=Path,
        default=None,
        help="Local Depth Anything V2 Metric Outdoor model directory",
    )
    parser.add_argument(
        "--near-anchor-m",
        type=float,
        default=1.5,
        help="Near distance anchor for inverse-calibrated fallback",
    )
    parser.add_argument(
        "--far-anchor-m",
        type=float,
        default=80.0,
        help="Far distance anchor for inverse-calibrated fallback",
    )
    parser.add_argument("--focal-length-mm", type=float, default=135.0)
    parser.add_argument("--f-number", type=float, default=2.0)
    parser.add_argument("--sensor-width-mm", type=float, default=36.0)
    parser.add_argument(
        "--coc-gain",
        type=float,
        default=1.0,
        help="Art-directed multiplier for physical CoC radii (1.0 = unmodified)",
    )
    parser.add_argument("--max-blur", type=float, default=20.0)
    parser.add_argument("--depth-current-weight", type=float, default=0.78)
    parser.add_argument("--transition-sec", type=float, default=1.5)
    parser.add_argument("--sam-interval", type=int, default=2)
    parser.add_argument("--sam-model", type=Path, default=None)
    args = parser.parse_args()

    pipeline = CinematicDoFv12(
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
    )
    pipeline.process_video(
        args.input,
        args.output,
        args.depth_output,
        args.coc_output,
    )


if __name__ == "__main__":
    main()
