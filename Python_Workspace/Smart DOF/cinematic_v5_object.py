"""
[DSIL] Highway Star — Cinematic DoF v5.0 (Object-Aware)
================================================================
핵심 혁신: 뎁스 기반이 아닌 "객체 인식 기반" 선택적 포커스.

이전 버전의 문제: 같은 깊이의 모든 물체가 선명해짐 (육교, 전봇대 등)
해결: 타겟 포인트에서 flood fill로 "객체 마스크"를 생성하여
      오직 해당 객체의 픽셀만 선명하게 유지.

동작 방식:
1. 타겟 포인트의 뎁스 값을 읽고, 그 값과 비슷한 뎁스를 가진
   연결된(connected) 픽셀 영역만을 "객체"로 인식
2. 객체 마스크 안쪽 = 원본 선명도 유지
3. 객체 마스크 바깥 = 뎁스 차이에 비례한 자연스러운 블러
4. 타겟 이탈 시 = 부드러운 포커스 풀링으로 Deep Focus 전환
"""

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline as hf_pipeline
import argparse
import time
import sys


class CinematicDoFv5:
    def __init__(self, init_point, max_blur=25):
        self.init_point = init_point
        self.max_blur = max_blur
        
        print(f"\n{'='*60}")
        print(f"  🎬 Cinematic DoF v5.0 — Object-Aware Focus")
        print(f"{'='*60}")
        print(f"  추적 포인트(Pixel): {init_point}")
        print(f"  최대 블러 반경: {max_blur}px")
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
        
        # 포커스 풀링용 상태
        self.current_blur_strength = 1.0  # 1.0 = 풀 블러, 0.0 = 전체 선명
        self.transition_speed = 0.05      # 프레임당 블러 감소량 (약 20프레임에 걸쳐 전환)
        
        print("\n  ✅ 준비 완료!\n")

    def _get_depth_map(self, frame_rgb, w, h):
        """뎁스맵 추출 + Guided Filter로 경계선 보존"""
        pil_img = Image.fromarray(frame_rgb)
        result = self.depth_pipe(pil_img)
        depth = np.array(result["depth"]).astype(np.float32) / 255.0
        
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
        
        guide = frame_rgb.astype(np.float32) / 255.0
        depth = cv2.ximgproc.createGuidedFilter(guide, radius=8, eps=0.01).filter(depth)
        return np.clip(depth, 0.0, 1.0)

    def _create_object_mask(self, depth_norm, target_x, target_y, w, h):
        """
        타겟 포인트에서 시작하여 비슷한 뎁스를 가진 연결 영역만 찾아서
        "객체 마스크"를 생성합니다.
        
        원리: 
        - 타겟 포인트의 뎁스값을 기준으로, 일정 범위(threshold) 안에 있는
          인접 픽셀들만 flood fill로 찾음
        - 공간적으로 연결되지 않은 같은 깊이의 물체(육교 등)는 포함되지 않음
        """
        target_depth = depth_norm[target_y, target_x]
        
        # 뎁스를 8비트로 변환 (floodFill 요구사항)
        depth_8bit = (depth_norm * 255).astype(np.uint8)
        
        # flood fill 허용 범위: 뎁스 차이 ±15 (0~255 기준)
        # 이 값이 작을수록 더 정밀하게 객체만 잡음
        lo_diff = 15
        hi_diff = 15
        
        # flood fill 실행 (positional args for OpenCV 5.0 compatibility)
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        flood_flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
        
        cv2.floodFill(
            depth_8bit, mask, 
            (target_x, target_y), 
            255,
            (lo_diff,), (hi_diff,),
            flood_flags
        )
        
        # mask는 (h+2, w+2) 크기이므로 원래 크기로 자름
        object_mask = mask[1:-1, 1:-1]
        
        # 마스크 정리: 모폴로지 연산으로 노이즈 제거 및 빈틈 메우기
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
        
        return object_mask  # 255 = 객체, 0 = 배경

    def _compute_blur_map(self, depth_norm, object_mask, target_depth):
        """
        객체 마스크 바깥의 픽셀에 대해 뎁스 차이 기반 블러 맵 생성.
        
        - 객체 마스크 안 = 블러 0 (완벽히 선명)
        - 객체 마스크 밖 = 뎁스 차이에 비례한 자연스러운 블러
        """
        # 뎁스 차이 기반 블러
        depth_diff = np.abs(depth_norm - target_depth)
        
        # 비선형 매핑: sqrt를 써서 가까운 물체는 약간만 블러, 먼 물체는 확 날림
        blur_map = np.sqrt(depth_diff) * self.max_blur * 2.5
        blur_map = np.clip(blur_map, 0.0, self.max_blur)
        
        # 객체 마스크 안쪽은 블러 0으로 강제 (핵심!)
        blur_map[object_mask > 0] = 0.0
        
        # 마스크 경계를 부드럽게 (갑자기 선명↔블러가 바뀌면 부자연스러움)
        blur_map = cv2.GaussianBlur(blur_map, (15, 15), 5)
        
        # 다시 한번 마스크 안쪽 보호 (블러 번짐 방지)
        # 마스크를 약간 축소(erode)한 영역은 절대 블러 0 유지
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        safe_mask = cv2.erode(object_mask, kernel_small, iterations=1)
        blur_map[safe_mask > 0] = 0.0
        
        return blur_map.astype(np.float32)

    def _render_bokeh(self, image, blur_map):
        """블러 맵에 따라 가변 블러 적용 (선형 보간 방식)"""
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
                    # 타겟 추적 중: 객체 마스크 생성 → 선택적 블러
                    target_depth = depth_norm[y, x]
                    object_mask = self._create_object_mask(depth_norm, x, y, w, h)
                    blur_map = self._compute_blur_map(depth_norm, object_mask, target_depth)
                    
                    rendered = self._render_bokeh(frame, blur_map)
                    
                    # 빨간 점 (타겟 위치)
                    cv2.circle(rendered, (x, y), 5, (0, 0, 255), -1)
                    
                    out.write(rendered)
                    sys.stdout.write(f"\r  🎥 [{frame_idx}/{total}] {frame_idx*100//total}% | Tracking | Depth: {target_depth:.2f}")
                    sys.stdout.flush()
                    continue
            
            # 타겟 이탈 후: 부드러운 포커스 풀링으로 Deep Focus 전환
            self.current_blur_strength = max(0.0, self.current_blur_strength - self.transition_speed)
            
            if self.current_blur_strength > 0.01:
                # 아직 전환 중: 전체에 약한 균일 블러를 점점 줄이면서 적용
                blur_radius = int(self.max_blur * self.current_blur_strength)
                if blur_radius > 0:
                    k = blur_radius * 2 + 1
                    blurred_full = cv2.GaussianBlur(frame, (k, k), blur_radius * 0.5)
                    # 원본과 블러의 블렌딩 (strength가 줄수록 원본 비중 증가)
                    alpha = self.current_blur_strength
                    rendered = cv2.addWeighted(blurred_full, alpha, frame, 1.0 - alpha, 0)
                else:
                    rendered = frame
                
                cv2.putText(rendered, f"Focus Pulling... ({self.current_blur_strength:.0%})", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            else:
                # 전환 완료: 원본 그대로 (Deep Focus)
                rendered = frame
                cv2.putText(rendered, "Deep Focus", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            out.write(rendered)
            sys.stdout.write(f"\r  🎥 [{frame_idx}/{total}] {frame_idx*100//total}% | Transition {self.current_blur_strength:.0%}")
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
    
    pipe = CinematicDoFv5(init_point=(px, py), max_blur=args.max_blur)
    pipe.process_video(args.input, args.output)
