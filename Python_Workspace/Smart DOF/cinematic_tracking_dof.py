"""
[DSIL] Highway Star — True Optical Tracking DoF v2.0
================================================================
Phase 3: 객체 추적 기반의 순수 광학 렌더링 파이프라인

기존의 마스크 기반 강제 블러(가짜 DoF)를 제거하고, 
1. 안정적인 객체 추적(OpenCV CSRT)으로 타겟의 거리를 계산
2. 시간에 따른 부드러운 초점 이동(Focus Pulling) 적용
3. 타겟이 사라지면 자연스럽게 딥포커스(Deep Focus) 전환
4. 오직 픽셀의 깊이(Depth)와 초점 거리(Focus Distance)에 의해서만 CoC 물리 블러 적용

이 방식을 통해 실제 아이폰 시네마틱 모드나 렌즈의 광학적 특성을 정확히 모방합니다.
"""

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline as hf_pipeline
import argparse
import time
import sys

# Phase 1의 순수 CoC 엔진 재사용
from optical_dof_engine import CircleOfConfusion, OpticalBokehRenderer, LENS_PRESETS

class TrueCinematicDoFPipeline:
    def __init__(self, init_bbox, lens_preset="85mm_f1.4", num_layers=16, r_max=50):
        self.init_bbox = init_bbox  # (x, y, w, h)
        
        # 렌즈 설정
        preset = LENS_PRESETS.get(lens_preset, LENS_PRESETS["85mm_f1.4"])
        self.focal_length = preset["focal_length"]
        self.f_number = preset["f_number"]
        self.lens_name = preset["name"]
        self.num_layers = num_layers
        self.r_max = r_max
        
        print(f"\n{'='*60}")
        print(f"  🎬 True Optical Tracking DoF v2.0")
        print(f"{'='*60}")
        print(f"  렌즈: {self.lens_name}")
        print(f"  추적 타겟 BBox: {self.init_bbox}")
        print(f"{'='*60}\n")
        
        print("  [1/3] 뎁스 추정 모델 로딩...")
        self.depth_pipe = hf_pipeline(
            task="depth-estimation", 
            model="LiheYoung/depth-anything-small-hf"
        )
        
        print("  [2/3] 객체 추적기(CSRT) 초기화...")
        # CSRT: 속도는 조금 느리지만 정확도가 높음
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker_initialized = False
        
        print("  [3/3] 보케 렌더러 초기화...")
        self.bokeh_renderer = OpticalBokehRenderer(num_layers=num_layers)
        
        # 초점 추적용 변수
        self.smoothed_focus = None
        self.deep_focus_distance = 100.0  # 타겟 상실 시 딥포커스 기준 (m)
        self.tracking_lost = False
        
        print("\n  ✅ 모듈 로드 완료!\n")

    def _extract_depth(self, frame_rgb):
        pil_img = Image.fromarray(frame_rgb)
        result = self.depth_pipe(pil_img)
        return np.array(result["depth"]).astype(np.float32) / 255.0

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
            
            # 1. 뎁스 추출
            depth_norm = self._extract_depth(rgb)
            if depth_norm.shape[:2] != (height, width):
                depth_norm = cv2.resize(depth_norm, (width, height))
                
            # 2. 타겟 추적
            if not self.tracker_initialized:
                self.tracker.init(frame, self.init_bbox)
                self.tracker_initialized = True
                success = True
                bbox = self.init_bbox
            else:
                success, bbox = self.tracker.update(frame)
            
            # 3. 초점 거리 계산 (Focus Pulling)
            target_distance = self.deep_focus_distance
            
            if success:
                x, y, w, h = [int(v) for v in bbox]
                
                # 스마트 이탈 감지 (타겟이 화면 가장자리에 닿으면 추적 종료로 간주)
                if x < 15 or (x + w) > width - 15 or y < 15 or (y + h) > height - 15 or w < 10 or h < 10:
                    success = False
                    if not self.tracking_lost:
                        print(f"\n  ⚠️ 프레임 {frame_idx}: 타겟이 화면 밖으로 이탈. 딥포커스로 전환합니다.")
                        self.tracking_lost = True
                else:
                    # 정상 추적 중
                    roi_depth = depth_norm[y:y+h, x:x+w]
                    if roi_depth.size > 0:
                        median_depth = np.median(roi_depth)
                        # 뎁스 0~1을 실제 거리(m)로 변환 (CoC 공식과 정확히 일치)
                        # 0 (검은색) = Far (80m), 1 (흰색) = Near (0.5m)
                        depth_near_m = 0.5
                        depth_far_m = 80.0
                        inv_d = (1.0 / depth_far_m) + ((1.0 / depth_near_m) - (1.0 / depth_far_m)) * median_depth
                        target_distance = 1.0 / max(inv_d, 1e-6)
                        
                        # 시각적 피드백을 위해 BBox 그리기 (초록색)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if not success and not self.tracking_lost:
                print(f"\n  ⚠️ 프레임 {frame_idx}: 추적 알고리즘 자체 실패. 딥포커스로 전환합니다.")
                self.tracking_lost = True
            
            # 초점 시간적 스무딩 (시네마틱 포커스 풀링)
            if self.smoothed_focus is None:
                self.smoothed_focus = target_distance
            else:
                # 딥포커스로 갈 때(거리가 멀어질 때)와 가까워질 때의 스피드 조절
                # 부드러운 포커스 전환을 위해 smoothing_factor 0.9 적용
                smoothing_factor = 0.9
                self.smoothed_focus = (self.smoothed_focus * smoothing_factor) + (target_distance * (1 - smoothing_factor))
                
            # 4. 순수 물리 기반 CoC 반경 계산
            # 인위적 마스크 조작 제거! 순수 거리에 의존!
            coc = CircleOfConfusion(
                focal_length_mm=self.focal_length,
                f_number=self.f_number,
                image_width_px=width,
                focus_distance_m=self.smoothed_focus,
                d_zero_m=0.1,  # 선명한 구간을 0.1m로 좁게 설정하여 영화 렌즈의 얕은 심도 극대화
                r_max_px=self.r_max
            )
            blur_map = coc.compute_blur_radius_map(depth_norm)
            
            # 5. 보케 렌더링
            rendered = self.bokeh_renderer.render(frame, blur_map)
            
            # 상태 표시
            focus_str = f"Focus: {self.smoothed_focus:.1f}m"
            cv2.putText(rendered, focus_str, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if not success:
                cv2.putText(rendered, "TRACKING LOST - DEEP FOCUS", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            out.write(rendered)
            
            # 콘솔 진행률 표시
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
    parser.add_argument("--roi", type=str, required=True, help="x,y,w,h (예: 190,160,90,120)")
    parser.add_argument("--lens", default="85mm_f1.4")
    
    args = parser.parse_args()
    bbox = tuple(map(int, args.roi.split(',')))
    
    pipeline = TrueCinematicDoFPipeline(init_bbox=bbox, lens_preset=args.lens)
    pipeline.process_video(args.input, args.output)
