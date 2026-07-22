import cv2
import numpy as np
import time
import sys
import argparse
from PIL import Image
from transformers import pipeline as hf_pipeline
from ultralytics import SAM
import torch

class CinematicDoFv6:
    def __init__(self, init_point, max_blur=25):
        self.init_point = init_point
        self.max_blur = max_blur
        
        print("\n============================================================")
        print("  🎬 Cinematic DoF v6.0 — SAM-based Object Tracking")
        print("============================================================")
        print(f"  추적 포인트(Pixel): {init_point}")
        print(f"  최대 블러 반경: {max_blur}px")
        print("============================================================\n")
        
        # 1. Depth 모델
        print("  [1/3] 뎁스 추정 모델 로딩 (Depth Anything)...")
        self.depth_pipe = hf_pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf")
        
        # 2. SAM 모델
        print("  [2/3] SAM 모델 로딩 (MobileSAM)...")
        self.sam_model = SAM('../0720 Pitch/mobile_sam.pt')
        
        # 3. 추적기 초기화
        print("  [3/3] 광학 흐름(Optical Flow) 추적기 초기화...")
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        self.prev_gray = None
        self.current_point = np.array([[init_point]], dtype=np.float32)
        
        self.tracking_lost = False
        self.transition_speed = 0.05
        self.current_blur_strength = 1.0  # 1.0: 배경 흐림 유지, 0.0: Deep Focus 완료
        
        self.last_blur_map = None
        self.last_target_depth = None
        
        print("\n  ✅ 준비 완료!\n")

    def _get_depth_map(self, rgb, w, h):
        pil_img = Image.fromarray(rgb)
        result = self.depth_pipe(pil_img)
        depth = np.array(result["depth"]).astype(np.float32) / 255.0
        
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
            
        guide = rgb.astype(np.float32) / 255.0
        depth = cv2.ximgproc.createGuidedFilter(guide, radius=8, eps=0.01).filter(depth)
        depth = np.clip(depth, 0.0, 1.0)
        return depth

    def _get_sam_mask(self, frame, x, y):
        """SAM 모델을 사용해 현재 프레임의 (x, y) 좌표에 해당하는 객체 마스크를 추출"""
        # ultralytics SAM predict
        results = self.sam_model.predict(frame, points=[[x, y]], labels=[1], verbose=False)
        
        if results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy()
            mask = mask.astype(np.float32)
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            
            # 0.0 or 1.0
            return (mask > 0.5).astype(np.uint8) * 255
        else:
            # 마스크를 못 찾은 경우 빈 마스크 반환
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    def _compute_blur_map(self, depth_norm, object_mask, target_depth):
        """객체 마스크 밖은 뎁스 기반 블러, 안쪽은 블러 0"""
        depth_diff = np.abs(depth_norm - target_depth)
        
        blur_map = np.sqrt(depth_diff) * self.max_blur * 2.5
        blur_map = np.clip(blur_map, 0.0, self.max_blur)
        
        # 마스크 내부는 무조건 선명
        blur_map[object_mask > 0] = 0.0
        
        # 경계 스무딩
        blur_map = cv2.GaussianBlur(blur_map, (15, 15), 5)
        
        # 핵심 객체 내부 한 번 더 보호
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        safe_mask = cv2.erode(object_mask, kernel_small, iterations=1)
        blur_map[safe_mask > 0] = 0.0
        
        return blur_map.astype(np.float32)

    def _render_bokeh(self, image, blur_map):
        max_b = int(np.max(blur_map))
        if max_b == 0:
            return image
        
        levels = [image.astype(np.float32)]
        radii = [0]
        
        r = 1
        while r <= max_b + 2:
            k = r * 2 + 1
            sigma = r * 0.6
            blurred = cv2.GaussianBlur(image, (k, k), sigma)
            levels.append(blurred.astype(np.float32))
            radii.append(r)
            if r < 5:
                r += 1
            else:
                r = int(r * 1.4) + 1
        
        result = np.zeros_like(image, dtype=np.float32)
        radii_arr = np.array(radii, dtype=np.float32)
        
        idx = np.searchsorted(radii_arr, blur_map)
        idx = np.clip(idx, 1, len(radii_arr) - 1)
        
        for i in range(1, len(radii)):
            mask = (idx == i)
            if not np.any(mask):
                continue
            
            lo_r = radii_arr[i - 1]
            hi_r = radii_arr[i]
            w_hi = (blur_map[mask] - lo_r) / (hi_r - lo_r + 1e-6)
            w_hi = np.clip(w_hi, 0, 1)
            w_lo = 1.0 - w_hi
            
            result[mask] = levels[i-1][mask] * w_lo[:, None] + levels[i][mask] * w_hi[:, None]
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("  ❌ 영상을 열 수 없습니다.")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        frame_idx = 0
        t0 = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Step 1: 뎁스맵
            depth_norm = self._get_depth_map(rgb, w, h)
            
            # Step 2: 타겟 추적
            tracking_ok = False
            
            if self.prev_gray is None:
                self.prev_gray = gray
                tracking_ok = True
            elif not self.tracking_lost:
                new_pts, st, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.current_point, None, **self.lk_params
                )
                if st[0][0] == 1:
                    self.current_point = new_pts
                    tracking_ok = True
                self.prev_gray = gray
            
            # Step 3: 렌더링
            if tracking_ok and not self.tracking_lost:
                x = int(self.current_point[0][0][0])
                y = int(self.current_point[0][0][1])
                
                if x < 15 or x > w - 15 or y < 15 or y > h - 15:
                    self.tracking_lost = True
                    print(f"\n  ⚠️ 프레임 {frame_idx}: 타겟 이탈 → 포커스 풀링 시작")
                else:
                    target_depth = depth_norm[y, x]
                    self.last_target_depth = target_depth
                    
                    # SAM 객체 마스크 추출
                    object_mask = self._get_sam_mask(frame, x, y)
                    
                    # 블러 맵 계산
                    blur_map = self._compute_blur_map(depth_norm, object_mask, target_depth)
                    self.last_blur_map = blur_map
                    
                    rendered = self._render_bokeh(frame, blur_map)
                    
                    # 추적 위치 표시 (디버그 용이)
                    # cv2.circle(rendered, (x, y), 5, (0, 0, 255), -1)
                    
                    out.write(rendered)
                    sys.stdout.write(f"\r  🎥 [{frame_idx}/{total}] {frame_idx*100//total}% | Tracking SAM")
                    sys.stdout.flush()
                    continue
            
            # 타겟 이탈 후: 포커스 풀링 (점진적 배경 블러 해제)
            if self.last_blur_map is not None:
                self.current_blur_strength = max(0.0, self.current_blur_strength - self.transition_speed)
                
                if self.current_blur_strength > 0.01:
                    # 마지막으로 계산했던 블러 맵의 강도를 줄여가면서 적용
                    current_blur_map = self.last_blur_map * self.current_blur_strength
                    rendered = self._render_bokeh(frame, current_blur_map)
                else:
                    rendered = frame
            else:
                rendered = frame
            
            out.write(rendered)
            sys.stdout.write(f"\r  🎥 [{frame_idx}/{total}] {frame_idx*100//total}% | Focus Pulling {self.current_blur_strength:.0%}")
            sys.stdout.flush()
        
        cap.release()
        out.release()
        
        elapsed = time.time() - t0
        print(f"\n\n  ✅ 렌더링 완료! ({elapsed:.1f}초)")
        print(f"  출력: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--point", type=str, required=True, help="x,y")
    parser.add_argument("--max_blur", type=int, default=25)
    
    args = parser.parse_args()
    px, py = map(int, args.point.split(','))
    
    pipe = CinematicDoFv6(init_point=(px, py), max_blur=args.max_blur)
    pipe.process_video(args.input, args.output)
