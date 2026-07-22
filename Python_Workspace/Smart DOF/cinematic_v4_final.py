"""
[DSIL] Highway Star — Cinematic DoF v4.0 (Final)
================================================================
근본적 리팩토링: 모든 물리 오류 및 렌더링 결함을 완전히 제거.

핵심 변경사항:
1. 타겟이 선명할 때: 오직 타겟 깊이의 물체만 선명, 나머지는 거리에 비례한 자연스러운 블러
2. 타겟이 사라졌을 때: 즉시 블러 맵 = 0 (전체 선명, Deep Focus)
3. 블러 강도를 자연스러운 수준으로 조절 (과도한 multiplier 제거)
"""

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline as hf_pipeline
import argparse
import time
import sys


class CinematicDoFv4:
    def __init__(self, init_point, lens_focal_mm=85.0, f_number=1.4):
        self.init_point = init_point
        self.focal_mm = lens_focal_mm
        self.f_number = f_number
        
        print(f"\n{'='*60}")
        print(f"  🎬 Cinematic DoF v4.0 — Final Edition")
        print(f"{'='*60}")
        print(f"  렌즈: {lens_focal_mm}mm f/{f_number}")
        print(f"  추적 포인트(Pixel): {init_point}")
        print(f"{'='*60}\n")
        
        print("  [1/2] 뎁스 추정 모델 로딩...")
        self.depth_pipe = hf_pipeline(
            task="depth-estimation",
            model="LiheYoung/depth-anything-base-hf"
        )
        
        print("  [2/2] 추적기 초기화...")
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        self.prev_gray = None
        self.current_point = np.array([[init_point]], dtype=np.float32)
        self.tracking_lost = False
        self.smoothed_focus = None
        
        # 블러 강도 설정 — 자연스러운 수준
        self.max_blur_radius = 30  # 최대 블러 반경 (px) — 30이 자연스러운 한계
        
        print("\n  ✅ 준비 완료!\n")

    def _get_depth_map(self, frame_rgb, w, h):
        """뎁스맵 추출 + Guided Filter로 경계선 보존"""
        pil_img = Image.fromarray(frame_rgb)
        result = self.depth_pipe(pil_img)
        depth = np.array(result["depth"]).astype(np.float32) / 255.0
        
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Guided Filter: 원본 RGB를 가이드로 경계선 보존
        guide = frame_rgb.astype(np.float32) / 255.0
        depth = cv2.ximgproc.createGuidedFilter(guide, radius=8, eps=0.01).filter(depth)
        return np.clip(depth, 0.0, 1.0)

    def _compute_blur_map(self, depth_norm, focus_depth_value):
        """
        핵심 렌더링 로직: 뎁스맵에서 직접 블러 맵을 생성.
        
        물리 공식을 거치지 않고, 뎁스 값의 차이를 직접 블러 강도로 변환합니다.
        이유: 모노큘러 뎁스맵은 상대적 깊이(relative depth)이므로,
              물리적 절대 거리로 변환하면 오차가 누적되어 결과가 엉뚱해집니다.
              대신 뎁스 값 차이를 직접 사용하면 훨씬 안정적이고 자연스럽습니다.
        
        blur(pixel) = |depth(pixel) - depth(target)| * blur_strength
        """
        # 타겟 뎁스와의 차이 = 블러 강도의 근거
        depth_diff = np.abs(depth_norm - focus_depth_value)
        
        # 비선형 매핑: 차이가 작을 때는 거의 0, 차이가 커지면 급격히 증가
        # 이렇게 하면 타겟 주변은 선명하게 유지되고, 먼 곳은 확 날아감
        blur_map = np.power(depth_diff * 3.0, 1.5) * self.max_blur_radius
        
        # 클리핑
        blur_map = np.clip(blur_map, 0.0, self.max_blur_radius)
        
        return blur_map.astype(np.float32)

    def _render_bokeh(self, image, blur_map):
        """
        블러 맵에 따라 각 픽셀에 적절한 블러를 적용.
        선형 보간 방식으로 부드러운 그라데이션을 보장합니다.
        """
        max_blur = int(np.max(blur_map))
        if max_blur == 0:
            return image
        
        # 여러 단계의 블러 이미지를 미리 생성
        levels = [image.astype(np.float32)]  # 레벨 0: 원본
        radii = [0]
        
        r = 1
        while r <= max_blur + 2:
            k = r * 2 + 1
            sigma = r * 0.6  # 가우시안 시그마 — 0.6이 자연스러운 보케 느낌
            blurred = cv2.GaussianBlur(image, (k, k), sigma)
            levels.append(blurred.astype(np.float32))
            radii.append(r)
            if r < 5:
                r += 1  # 작은 블러는 촘촘하게
            else:
                r = int(r * 1.4) + 1  # 큰 블러는 점프
        
        result = np.zeros_like(image, dtype=np.float32)
        radii_arr = np.array(radii, dtype=np.float32)
        
        # 각 픽셀에 대해 상/하한 블러 레벨을 찾아 선형 보간
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
            
            # Step 1: 뎁스맵 추출
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
            
            # Step 3: 블러 맵 결정
            if tracking_ok and not self.tracking_lost:
                x = int(self.current_point[0][0][0])
                y = int(self.current_point[0][0][1])
                
                # 화면 밖으로 나갔는지 확인
                if x < 15 or x > w - 15 or y < 15 or y > h - 15:
                    self.tracking_lost = True
                    print(f"\n  ⚠️ 프레임 {frame_idx}: 타겟 이탈 → Deep Focus (전체 선명)")
                else:
                    # 타겟 픽셀 주변 5x5 영역의 중앙 뎁스값 추출
                    roi = depth_norm[max(0,y-2):min(h,y+3), max(0,x-2):min(w,x+3)]
                    target_depth = np.median(roi) if roi.size > 0 else 0.5
                    
                    # 타겟 뎁스를 기준으로 블러 맵 생성
                    blur_map = self._compute_blur_map(depth_norm, target_depth)
                    
                    # 빨간 점 표시
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                    
                    # 보케 렌더링
                    rendered = self._render_bokeh(frame, blur_map)
                    
                    # 상태 표시
                    cv2.putText(rendered, f"Target Depth: {target_depth:.2f}", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    
                    out.write(rendered)
                    sys.stdout.write(f"\r  🎥 [{frame_idx}/{total}] {frame_idx*100//total}% | Tracking | Depth: {target_depth:.2f}")
                    sys.stdout.flush()
                    continue  # 이 프레임 처리 완료
            
            # 타겟을 잃었거나 추적 실패 → Deep Focus (원본 그대로 = 전체 선명)
            cv2.putText(frame, "DEEP FOCUS (All Sharp)", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            out.write(frame)
            sys.stdout.write(f"\r  🎥 [{frame_idx}/{total}] {frame_idx*100//total}% | Deep Focus")
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
    parser.add_argument("--focal", type=float, default=85.0)
    parser.add_argument("--fnum", type=float, default=1.4)
    
    args = parser.parse_args()
    px, py = map(int, args.point.split(','))
    
    pipe = CinematicDoFv4(
        init_point=(px, py),
        lens_focal_mm=args.focal,
        f_number=args.fnum
    )
    pipe.process_video(args.input, args.output)
