"""
[DSIL] Highway Star — Cinematic DoF + SAM 3 통합 파이프라인 v1.0
================================================================
Phase 2: SAM 3의 텍스트 기반 세그먼테이션 + CoC 광학 보케의 융합

핵심 기능:
1. 텍스트 프롬프트로 초점 대상 지정 (예: "car", "person")
2. 텍스트 프롬프트로 방해 요소 지정 → 추가 블러 (예: "sign", "pole")
3. SAM 세그먼트 + 뎁스맵 융합으로 경계 아티팩트 제거
4. 프레임별 자동 초점 추적 (피사체 뎁스 → focus_distance 동적 조절)
5. CoC 물리 보케 렌더링

Usage:
    python cinematic_dof_sam3.py --input video.mp4 --focus-on "car" --blur-extra "sign,pole"
    python cinematic_dof_sam3.py --input video.mp4 --focus-on "car" --lens 85mm_f1.4
"""

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline as hf_pipeline
import argparse
import time
import sys
import os

# Phase 1 엔진 재사용
from optical_dof_engine import (
    CircleOfConfusion, 
    OpticalBokehRenderer, 
    LENS_PRESETS
)


# ============================================================
#  SAM 모델 래퍼 (SAM 3 / SAM 2 / MobileSAM 자동 감지)
# ============================================================
class SAMSegmenter:
    """
    SAM 모델을 자동으로 감지하고 로드합니다.
    SAM 3 → SAM 2 → MobileSAM 순서로 폴백합니다.
    
    Mac 환경에서는 CPU 모드로 자동 전환됩니다.
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Args:
            model_path: 모델 가중치 경로 (None이면 자동 감지)
            device: 'cpu', 'cuda', 'mps' (None이면 자동 감지)
        """
        self.model = None
        self.model_type = None
        self.device = device
        
        # 디바이스 자동 감지
        if self.device is None:
            import torch
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # SAM 3는 MPS에서 triton 의존성 문제로 불안정
                # CPU로 폴백하되 사용자에게 알림
                self.device = 'cpu'
                print("  ⚠️  Mac 환경 감지 — CPU 모드로 실행합니다 (MPS는 SAM 3과 호환 이슈)")
            else:
                self.device = 'cpu'
        
        self._load_model(model_path)
    
    def _load_model(self, model_path):
        """모델을 우선순위에 따라 로드"""
        from ultralytics import SAM
        
        # 모델 경로 후보 (우선순위 순서)
        candidates = []
        if model_path:
            candidates.append((model_path, "Custom"))
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        
        candidates.extend([
            (os.path.join(script_dir, "sam3.pt"), "SAM 3"),
            (os.path.join(script_dir, "sam3.1.pt"), "SAM 3.1"),
            ("sam3.pt", "SAM 3 (auto-download)"),
            (os.path.join(parent_dir, "0720 Pitch", "mobile_sam.pt"), "MobileSAM"),
            (os.path.join(script_dir, "mobile_sam.pt"), "MobileSAM"),
            ("mobile_sam.pt", "MobileSAM (auto-download)"),
        ])
        
        for path, name in candidates:
            try:
                print(f"  [SAM] {name} 모델 로드 시도: {path}")
                self.model = SAM(path)
                self.model_type = name
                print(f"  [SAM] ✅ {name} 모델 로드 성공!")
                return
            except Exception as e:
                print(f"  [SAM] ❌ {name} 실패: {str(e)[:80]}")
                continue
        
        raise RuntimeError("사용 가능한 SAM 모델을 찾을 수 없습니다. sam3.pt 또는 mobile_sam.pt를 다운로드해 주세요.")
    
    def segment_by_points(self, frame, points, labels=None):
        """
        클릭 좌표 기반 세그먼테이션 (SAM 1/2/3 공통)
        
        Args:
            frame: BGR 이미지 (H, W, 3)
            points: [[x, y], ...] 좌표 리스트
            labels: [1, 1, ...] 전경=1, 배경=0
            
        Returns:
            mask: 이진 마스크 (H, W), float32
        """
        if labels is None:
            labels = [1] * len(points)
        
        results = self.model.predict(
            frame, points=points, labels=labels, 
            device=self.device, verbose=False
        )
        
        if results and len(results) > 0 and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy().astype(np.float32)
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            return mask
        
        return np.zeros(frame.shape[:2], dtype=np.float32)
    
    def segment_by_text(self, frame, text_prompts, depth_map=None):
        """
        텍스트 기반 세그먼테이션 (SAM 3 전용 기능)
        
        SAM 3가 아닌 경우 → 뎁스 기반 스마트 포인트 프롬프트로 폴백
        
        Args:
            frame: BGR 이미지 (H, W, 3)
            text_prompts: ["car", "person"] 등 텍스트 리스트
            depth_map: (선택) 정규화 뎁스맵 — 폴백 시 활용
            
        Returns:
            masks: {텍스트: mask} 딕셔너리
        """
        masks = {}
        
        for idx, text in enumerate(text_prompts):
            try:
                # SAM 3 텍스트 프롬프트 시도
                results = self.model.predict(
                    frame, texts=[text],
                    device=self.device, verbose=False
                )
                
                if results and len(results) > 0 and results[0].masks is not None:
                    combined = np.zeros(frame.shape[:2], dtype=np.float32)
                    for i in range(len(results[0].masks.data)):
                        m = results[0].masks.data[i].cpu().numpy().astype(np.float32)
                        if m.shape[:2] != frame.shape[:2]:
                            m = cv2.resize(m, (frame.shape[1], frame.shape[0]))
                        combined = np.maximum(combined, m)
                    masks[text] = combined
                else:
                    masks[text] = np.zeros(frame.shape[:2], dtype=np.float32)
                    
            except (TypeError, AttributeError, RuntimeError, SyntaxError, Exception) as e:
                # 텍스트 프롬프트 미지원 모델 (MobileSAM 등)
                if not hasattr(self, '_text_warning_shown'):
                    print(f"\n  ⚠️  텍스트 프롬프트 미지원 (모델: {self.model_type})")
                    print(f"       → 뎁스 기반 스마트 포인트 프롬프트로 폴백합니다\n")
                    self._text_warning_shown = True
                
                h, w = frame.shape[:2]
                
                if idx == 0:
                    # 첫 번째 프롬프트(초점 대상): 화면 중앙~하단의 주요 피사체
                    # 주행 영상에서는 중앙-하단에 차량이 위치하는 경향
                    sample_points = [
                        [w // 2, int(h * 0.55)],      # 중앙
                        [w // 2, int(h * 0.65)],      # 중앙 하단
                        [int(w * 0.4), int(h * 0.6)], # 좌측 중앙
                        [int(w * 0.6), int(h * 0.6)], # 우측 중앙
                    ]
                    
                    best_mask = np.zeros((h, w), dtype=np.float32)
                    best_area = 0
                    
                    for pt in sample_points:
                        try:
                            m = self.segment_by_points(frame, [pt])
                            area = m.sum()
                            if area > best_area and area < h * w * 0.7:
                                best_mask = m
                                best_area = area
                        except Exception:
                            continue
                    
                    masks[text] = best_mask
                else:
                    # 추가 프롬프트(방해 요소): 상단 영역에서 샘플링
                    # 간판, 건물 등은 대체로 화면 상단에 위치
                    top_points = [
                        [int(w * 0.3), int(h * 0.2)],
                        [int(w * 0.7), int(h * 0.2)],
                        [int(w * 0.5), int(h * 0.15)],
                    ]
                    
                    combined = np.zeros((h, w), dtype=np.float32)
                    for pt in top_points:
                        try:
                            m = self.segment_by_points(frame, [pt])
                            combined = np.maximum(combined, m)
                        except Exception:
                            continue
                    
                    masks[text] = combined
        
        return masks


# ============================================================
#  스마트 초점 추적기
# ============================================================
class SmartFocusTracker:
    """
    프레임마다 피사체의 뎁스를 읽어 초점 거리를 자동 추적합니다.
    
    동작 방식:
    1. SAM 마스크 내부의 뎁스 값 중앙값 계산
    2. 피사체가 사라지면(마스크 비율 < 임계값) → 전체 선명(deep focus)으로 전환
    3. 급격한 초점 변화 방지를 위한 시간적 스무딩 적용
    """
    
    def __init__(self, depth_near=0.5, depth_far=80.0, 
                 smoothing=0.3, min_mask_ratio=0.005,
                 deep_focus_distance=50.0):
        """
        Args:
            depth_near/far: 뎁스맵 매핑 범위 (m)
            smoothing: 초점 전환 부드러움 (0=즉시, 1=매우 느리게)
            min_mask_ratio: 마스크가 이 비율 이하면 피사체 사라진 것으로 판단
            deep_focus_distance: 피사체 없을 때의 기본 초점 거리 (m)
        """
        self.depth_near = depth_near
        self.depth_far = depth_far
        self.smoothing = smoothing
        self.min_mask_ratio = min_mask_ratio
        self.deep_focus_distance = deep_focus_distance
        self.prev_focus = None
    
    def compute_focus_distance(self, depth_normalized, subject_mask):
        """
        피사체 마스크 영역의 뎁스 중앙값으로 초점 거리를 계산합니다.
        
        Args:
            depth_normalized: 정규화 뎁스맵 (H, W), 0~1
            subject_mask: 피사체 마스크 (H, W), 0~1
            
        Returns:
            focus_distance: 초점 거리 (m)
            subject_present: 피사체 존재 여부
        """
        # 마스크 비율 확인
        mask_ratio = subject_mask.mean()
        subject_present = mask_ratio > self.min_mask_ratio
        
        if subject_present:
            # 마스크 내부 픽셀의 뎁스 중앙값
            masked_depth = depth_normalized[subject_mask > 0.5]
            if len(masked_depth) > 0:
                median_depth = np.median(masked_depth)
                # 정규화 뎁스 → 실제 거리 (역수 보간)
                inv_near = 1.0 / self.depth_near
                inv_far = 1.0 / self.depth_far
                inv_d = inv_near + (inv_far - inv_near) * median_depth
                raw_focus = 1.0 / max(inv_d, 1e-6)
            else:
                raw_focus = self.deep_focus_distance
                subject_present = False
        else:
            raw_focus = self.deep_focus_distance
        
        # 시간적 스무딩 (급격한 초점 변화 방지)
        if self.prev_focus is None:
            smoothed_focus = raw_focus
        else:
            smoothed_focus = self.prev_focus * self.smoothing + raw_focus * (1 - self.smoothing)
        
        self.prev_focus = smoothed_focus
        return smoothed_focus, subject_present


# ============================================================
#  통합 시네마틱 DOF + SAM 3 파이프라인
# ============================================================
class CinematicDoFSAM3Pipeline:
    """
    SAM 3 세그먼테이션 + CoC 물리 보케의 완전 통합 파이프라인.
    
    워크플로우:
    1. 프레임 입력
    2. 뎁스맵 추출 (Depth Anything)
    3. SAM 3 텍스트 프롬프트 → 피사체 마스크 + 방해요소 마스크
    4. 피사체 마스크의 뎁스 → 자동 초점 거리 계산
    5. CoC 블러 반경 계산 + 방해요소 추가 블러
    6. 세그먼트 경계 정밀화 (마스크 + 뎁스 융합)
    7. 광학 보케 렌더링
    8. 출력
    """
    
    def __init__(self, focus_on="car", blur_extra=None,
                 lens_preset="85mm_f1.4", 
                 num_layers=16, r_max=40, d_zero=0.8,
                 extra_blur_strength=1.5,
                 sam_model_path=None,
                 segment_interval=3):
        """
        Args:
            focus_on: 초점 맞출 대상 텍스트 (예: "car")
            blur_extra: 추가 블러 대상 텍스트 (예: "sign,pole")
            lens_preset: 렌즈 프리셋
            extra_blur_strength: 방해 요소에 대한 추가 블러 배율
            sam_model_path: SAM 모델 경로
            segment_interval: N 프레임마다 SAM 재실행 (성능 최적화)
        """
        self.focus_on = focus_on
        self.blur_extra = blur_extra.split(",") if blur_extra else []
        self.extra_blur_strength = extra_blur_strength
        self.segment_interval = segment_interval
        
        # 렌즈 설정
        preset = LENS_PRESETS.get(lens_preset, LENS_PRESETS["85mm_f1.4"])
        self.focal_length = preset["focal_length"]
        self.f_number = preset["f_number"]
        self.lens_name = preset["name"]
        self.num_layers = num_layers
        self.r_max = r_max
        self.d_zero = d_zero
        
        # 모듈 초기화
        print(f"\n{'='*60}")
        print(f"  🎬 Cinematic DoF + SAM 3 통합 파이프라인 v1.0")
        print(f"{'='*60}")
        print(f"  렌즈: {self.lens_name}")
        print(f"  초점 대상: \"{focus_on}\"")
        if self.blur_extra:
            print(f"  추가 블러 대상: {self.blur_extra}")
        print(f"  세그먼테이션 간격: {segment_interval}프레임마다")
        print(f"{'='*60}\n")
        
        print("  [1/3] 뎁스 추정 모델 로딩...")
        self.depth_pipe = hf_pipeline(
            task="depth-estimation", 
            model="LiheYoung/depth-anything-small-hf"
        )
        
        print("  [2/3] SAM 모델 로딩...")
        self.sam = SAMSegmenter(model_path=sam_model_path)
        
        print("  [3/3] 보케 렌더러 초기화...")
        self.bokeh_renderer = OpticalBokehRenderer(num_layers=num_layers)
        self.focus_tracker = SmartFocusTracker()
        
        print("\n  ✅ 모든 모듈 로드 완료!\n")
    
    def _extract_depth(self, frame_rgb):
        """뎁스맵 추출"""
        pil_img = Image.fromarray(frame_rgb)
        result = self.depth_pipe(pil_img)
        return np.array(result["depth"]).astype(np.float32) / 255.0
    
    def _refine_mask_with_depth(self, mask, depth_map, subject_depth, threshold=0.15):
        """
        세그먼트 경계를 뎁스맵으로 정밀화합니다.
        
        SAM 마스크 경계에서 뎁스가 급변하는 부분(전경-배경 경계)을
        깔끔하게 분리하여 색번짐 아티팩트를 방지합니다.
        """
        # 마스크 경계 영역 추출 (erosion - original = 경계)
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode((mask > 0.5).astype(np.uint8), kernel)
        dilated = cv2.dilate((mask > 0.5).astype(np.uint8), kernel)
        border = dilated - eroded  # 경계 영역
        
        # 경계 영역에서 뎁스 차이가 큰 픽셀 제거 (배경 유출 방지)
        refined = mask.copy()
        if subject_depth > 0:
            depth_diff = np.abs(depth_map - subject_depth)
            # 경계에서 뎁스가 많이 다른 픽셀은 마스크에서 제외
            background_leak = (border > 0) & (depth_diff > threshold)
            refined[background_leak] = 0.0
        
        # 마스크 경계 부드럽게 (anti-aliasing)
        refined = cv2.GaussianBlur(refined, (5, 5), 1.0)
        
        return refined
    
    def process_video(self, input_path, output_path):
        """영상 전체를 SAM 3 + CoC로 처리합니다."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"❌ 영상을 열 수 없습니다: {input_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"  📐 해상도: {width}x{height} @ {fps:.1f}fps")
        print(f"  📊 총 프레임: {total_frames}")
        print(f"  🎯 초점 대상: \"{self.focus_on}\"\n")
        
        # 캐시 변수 (segment_interval 프레임마다 갱신)
        cached_subject_mask = None
        cached_extra_masks = {}
        
        frame_idx = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # ── Step 1: 뎁스맵 추출 ──
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth_norm = self._extract_depth(rgb)
            if depth_norm.shape[:2] != (height, width):
                depth_norm = cv2.resize(depth_norm, (width, height))
            
            # ── Step 2: SAM 세그먼테이션 (N프레임 간격) ──
            if frame_idx == 1 or frame_idx % self.segment_interval == 0:
                # 초점 대상 세그먼테이션
                all_prompts = [self.focus_on] + self.blur_extra
                masks = self.sam.segment_by_text(frame, all_prompts)
                
                cached_subject_mask = masks.get(self.focus_on, 
                                                np.zeros((height, width), dtype=np.float32))
                cached_extra_masks = {k: masks[k] for k in self.blur_extra if k in masks}
            
            subject_mask = cached_subject_mask
            
            # ── Step 3: 자동 초점 추적 ──
            focus_dist, subject_present = self.focus_tracker.compute_focus_distance(
                depth_norm, subject_mask
            )
            
            # ── Step 4: CoC 블러 반경 계산 ──
            coc = CircleOfConfusion(
                focal_length_mm=self.focal_length,
                f_number=self.f_number,
                image_width_px=width,
                focus_distance_m=focus_dist,
                d_zero_m=self.d_zero if subject_present else 5.0,  # 피사체 없으면 deep focus
                r_max_px=self.r_max
            )
            blur_map = coc.compute_blur_radius_map(depth_norm)
            
            # ── Step 5: 피사체 마스크로 경계 정밀화 ──
            if subject_present and subject_mask.max() > 0:
                # 피사체의 뎁스 중앙값
                masked_depths = depth_norm[subject_mask > 0.5]
                subj_depth = np.median(masked_depths) if len(masked_depths) > 0 else 0.5
                
                # 마스크 경계를 뎁스로 정밀화
                refined_mask = self._refine_mask_with_depth(
                    subject_mask, depth_norm, subj_depth
                )
                
                # 피사체 영역은 블러 0으로 강제 (초점 보호)
                blur_map = blur_map * (1.0 - refined_mask)
            
            # ── Step 6: 방해 요소에 추가 블러 적용 ──
            for extra_text, extra_mask in cached_extra_masks.items():
                if extra_mask.max() > 0:
                    # 방해 요소 영역의 블러를 증폭
                    boost = extra_mask * self.r_max * self.extra_blur_strength
                    blur_map = np.maximum(blur_map, boost)
            
            # ── Step 7: 광학 보케 렌더링 ──
            rendered = self.bokeh_renderer.render(frame, blur_map)
            
            out.write(rendered)
            
            # 진행 표시
            elapsed = time.time() - start_time
            avg_fps_val = frame_idx / elapsed
            remaining = (total_frames - frame_idx) / max(avg_fps_val, 0.01)
            
            focus_status = f"초점={focus_dist:.1f}m" if subject_present else "DeepFocus"
            sys.stdout.write(
                f"\r  🎥 [{frame_idx}/{total_frames}] "
                f"{frame_idx/total_frames*100:.0f}% | "
                f"{focus_status} | "
                f"남은: {remaining:.0f}s"
            )
            sys.stdout.flush()
        
        cap.release()
        out.release()
        
        total_time = time.time() - start_time
        print(f"\n\n  ✅ 렌더링 완료!")
        print(f"  총 소요: {total_time:.1f}초 ({total_time/60:.1f}분)")
        print(f"  출력: {output_path}\n")


# ============================================================
#  CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="🎬 Highway Star — Cinematic DoF + SAM 3 통합 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 자동차에 초점, 간판은 추가 블러
  python cinematic_dof_sam3.py -i video.mp4 --focus-on "car" --blur-extra "sign,pole"

  # 사람에 초점, 85mm 포트레이트 렌즈
  python cinematic_dof_sam3.py -i video.mp4 --focus-on "person" --lens 85mm_f1.4

  # 건물에 초점 (건물 지나가면 자동 deep focus 전환)
  python cinematic_dof_sam3.py -i video.mp4 --focus-on "building" --lens 50mm_f1.4
        """
    )
    
    parser.add_argument("--input", "-i", required=True, help="입력 영상")
    parser.add_argument("--output", "-o", default=None, help="출력 경로")
    parser.add_argument("--focus-on", "-f", default="car", 
                        help="초점 맞출 대상 (텍스트, 기본: car)")
    parser.add_argument("--blur-extra", "-b", default=None,
                        help="추가 블러 대상 (콤마 구분, 예: sign,pole)")
    parser.add_argument("--lens", "-l", default="85mm_f1.4",
                        choices=list(LENS_PRESETS.keys()),
                        help="렌즈 프리셋 (기본: 85mm_f1.4)")
    parser.add_argument("--r-max", type=int, default=40, help="최대 블러 반경")
    parser.add_argument("--d-zero", type=float, default=0.8, help="초점 선명 구간 (m)")
    parser.add_argument("--extra-strength", type=float, default=1.5,
                        help="방해요소 추가 블러 강도 배율")
    parser.add_argument("--segment-interval", type=int, default=3,
                        help="SAM 실행 간격 (N프레임마다, 기본: 3)")
    parser.add_argument("--sam-model", default=None, help="SAM 모델 경로")
    parser.add_argument("--layers", type=int, default=16, help="뎁스 레이어 수")
    
    args = parser.parse_args()
    
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}_cinematic_sam3.mp4"
    
    pipeline = CinematicDoFSAM3Pipeline(
        focus_on=args.focus_on,
        blur_extra=args.blur_extra,
        lens_preset=args.lens,
        num_layers=args.layers,
        r_max=args.r_max,
        d_zero=args.d_zero,
        extra_blur_strength=args.extra_strength,
        sam_model_path=args.sam_model,
        segment_interval=args.segment_interval,
    )
    
    pipeline.process_video(args.input, args.output)


if __name__ == "__main__":
    main()
