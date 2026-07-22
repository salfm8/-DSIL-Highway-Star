"""
[DSIL] Highway Star — High-End Cinematic DoF v3.0
================================================================
Phase 4: 초고화질 시네마틱 렌더링 파이프라인 (레퍼런스 영상 퀄리티 달성)

문제점 해결:
1. 뎁스맵 품질 저하(경계선 번짐) -> Base 모델 사용 및 Guided Filter로 엣지 보존 (칼같은 경계)
2. 박스 추적의 한계(배경 뎁스 섞임) -> Lucas-Kanade Optical Flow 기반 단일 픽셀 추적으로 완벽한 객체 거리 측정
3. 부자연스러운 포커싱 -> 강화된 시간축 스무딩 적용
"""

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline as hf_pipeline
import argparse
import time
import sys
import torch

from optical_dof_engine import CircleOfConfusion, OpticalBokehRenderer, LENS_PRESETS

class HighEndCinematicDoFPipeline:
    def __init__(self, init_point, lens_preset="85mm_f1.4", num_layers=24, r_max=60):
        self.init_point = init_point  # (x, y) 튜플
        
        # 렌즈 설정 (더 극적인 아웃포커싱을 위해 r_max 증가, 레이어 수 증가)
        preset = LENS_PRESETS.get(lens_preset, LENS_PRESETS["85mm_f1.4"])
        self.focal_length = preset["focal_length"]
        self.f_number = preset["f_number"]
        self.lens_name = preset["name"]
        self.num_layers = num_layers
        self.r_max = r_max
        
        print(f"\n{'='*60}")
        print(f"  🎬 High-End Cinematic DoF v3.0")
        print(f"{'='*60}")
        print(f"  렌즈: {self.lens_name}")
        print(f"  추적 포인트(Pixel): {self.init_point}")
        print(f"{'='*60}\n")
        
        print("  [1/3] 뎁스 추정 모델 로딩 (Base 모델로 업그레이드)...")
        # Base 모델을 사용하여 더 디테일한 뎁스맵 추출
        self.depth_pipe = hf_pipeline(
            task="depth-estimation", 
            model="LiheYoung/depth-anything-base-hf"
        )
        
        print("  [2/3] 광학 흐름(Optical Flow) 추적기 초기화...")
        # Lucas-Kanade 파라미터 설정
        self.lk_params = dict(winSize=(21, 21), 
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        print("  [3/3] 고해상도 보케 렌더러 초기화...")
        self.bokeh_renderer = OpticalBokehRenderer(num_layers=num_layers)
        
        # 추적 변수
        self.prev_gray = None
        self.current_point = np.array([[self.init_point]], dtype=np.float32)
        
        self.smoothed_focus = None
        self.deep_focus_distance = 100.0
        self.tracking_lost = False
        
        print("\n  ✅ 모듈 로드 완료!\n")

    def _extract_depth_with_guided_filter(self, frame_rgb, width, height):
        """
        뎁스를 추출하고 원본 RGB 이미지를 가이드로 삼아 경계선(Edge)을 칼같이 살려냅니다.
        이 과정이 없으면 객체 경계가 뭉개져 렌즈 특유의 분리감이 나지 않습니다.
        """
        pil_img = Image.fromarray(frame_rgb)
        result = self.depth_pipe(pil_img)
        depth_raw = np.array(result["depth"]).astype(np.float32) / 255.0
        
        if depth_raw.shape[:2] != (height, width):
            depth_raw = cv2.resize(depth_raw, (width, height), interpolation=cv2.INTER_CUBIC)
            
        # Guided Filter 적용 (RGB 이미지를 가이드로 사용)
        # 반지름(radius)=8, 정규화(eps)=0.01
        guide = frame_rgb.astype(np.float32) / 255.0
        depth_filtered = cv2.ximgproc.createGuidedFilter(guide, radius=8, eps=0.01).filter(depth_raw)
        
        # 필터링 과정에서 벗어난 값(0~1) 클리핑
        return np.clip(depth_filtered, 0.0, 1.0)

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. 뎁스 추출 (Guided Filter로 엣지 샤프닝)
            depth_norm = self._extract_depth_with_guided_filter(rgb, width, height)
                
            # 2. 픽셀 단위 타겟 추적 (Lucas-Kanade)
            success = False
            if self.prev_gray is None:
                self.prev_gray = gray
                success = True
            else:
                if not self.tracking_lost:
                    new_points, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.current_point, None, **self.lk_params)
                    if st[0][0] == 1:
                        self.current_point = new_points
                        success = True
                    self.prev_gray = gray
            
            target_distance = self.deep_focus_distance
            
            if success and not self.tracking_lost:
                x, y = int(self.current_point[0][0][0]), int(self.current_point[0][0][1])
                
                # 스마트 이탈 감지
                if x < 15 or x > width - 15 or y < 15 or y > height - 15:
                    success = False
                    print(f"\n  ⚠️ 프레임 {frame_idx}: 타겟 이탈. Deep Focus 전환.")
                    self.tracking_lost = True
                else:
                    # 완벽한 타겟팅: 정확히 그 픽셀 주변(5x5)의 뎁스만 추출
                    # 배경이 섞일 확률 원천 차단
                    roi_depth = depth_norm[max(0, y-2):min(height, y+3), max(0, x-2):min(width, x+3)]
                    if roi_depth.size > 0:
                        median_depth = np.median(roi_depth)
                        # 거리 매핑 공식 (수정된 올바른 공식)
                        depth_near_m = 0.5
                        depth_far_m = 80.0
                        inv_d = (1.0 / depth_far_m) + ((1.0 / depth_near_m) - (1.0 / depth_far_m)) * median_depth
                        target_distance = 1.0 / max(inv_d, 1e-6)
                        
                        # 시각적 피드백 (빨간 점)
                        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            
            # 3. 초점 시간적 스무딩
            if self.smoothed_focus is None:
                self.smoothed_focus = target_distance
            else:
                if self.tracking_lost:
                    smoothing_factor = 0.98  # 타겟 이탈 시 아주 천천히 스무딩 (Deep Focus 연출)
                else:
                    smoothing_factor = 0.3   # 타겟 추적 중일 때는 빠르게 쫓아감 (항상 쨍하게 유지)
                self.smoothed_focus = (self.smoothed_focus * smoothing_factor) + (target_distance * (1.0 - smoothing_factor))
                
            # 4. 순수 물리 기반 CoC 반경 계산
            coc = CircleOfConfusion(
                focal_length_mm=self.focal_length,
                f_number=self.f_number,
                image_width_px=width,
                focus_distance_m=self.smoothed_focus,
                d_zero_m=0.1,  # 극강의 심도를 위해 초점 허용 구간 최소화
                r_max_px=self.r_max
            )
            blur_map = coc.compute_blur_radius_map(depth_norm)
            
            # 5. 고해상도 보케 렌더링
            rendered = self.bokeh_renderer.render(frame, blur_map)
            
            # 상태 표시
            focus_str = f"Focus: {self.smoothed_focus:.1f}m"
            cv2.putText(rendered, focus_str, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if self.tracking_lost:
                cv2.putText(rendered, "DEEP FOCUS (INFINITY)", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            out.write(rendered)
            
            # 진행률
            sys.stdout.write(f"\r  🎥 [{frame_idx}/{total_frames}] {frame_idx/total_frames*100:.0f}% | {focus_str}")
            sys.stdout.flush()
            
        cap.release()
        out.release()
        
        elapsed = time.time() - start_time
        print(f"\n\n  ✅ 렌더링 완료! ({elapsed:.1f}초)")
        print(f"  출력: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--point", type=str, required=True, help="x,y (예: 240,220)")
    parser.add_argument("--lens", default="85mm_f1.4")
    
    args = parser.parse_args()
    px, py = map(int, args.point.split(','))
    
    pipeline = HighEndCinematicDoFPipeline(init_point=(px, py), lens_preset=args.lens)
    pipeline.process_video(args.input, args.output)
