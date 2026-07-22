"""
[DSIL] Highway Star — Optical DoF Engine v1.0
==============================================
시네마틱 트랜스포메이션을 위한 물리 기반 피사계 심도(DoF) 렌더링 엔진.

기존 균일 가우시안 블러 방식의 한계를 극복하고,
실제 시네 렌즈의 광학 공식(Circle of Confusion)을 코드로 구현하여
자연스럽고 아름다운 보케(Bokeh) 효과를 생성합니다.

핵심 물리 모델 (Thin Lens Model):
    CoC(d) = |A * f / (d - f)| * |1/d_focus - 1/d| * sensor_scale
    
    여기서:
    - A: 조리개 직경 = focal_length / f_number
    - f: 렌즈 초점 거리 (mm)
    - d: 해당 픽셀의 피사체 거리 (m)
    - d_focus: 초점이 맞춰진 거리 (m)

참고 논문: "Synthetic Depth-of-Field with a Single-Camera Mobile Phone" (Wadhwa et al., Google)

Usage:
    python optical_dof_engine.py --input sample_38_short.mp4 --lens 50 --aperture 1.4
"""

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import argparse
import time
import sys
import os


# ============================================================
#  시네 렌즈 프리셋 (Cinematic Lens Presets)
# ============================================================
LENS_PRESETS = {
    "35mm_f1.4": {"focal_length": 35.0, "f_number": 1.4, "name": "35mm f/1.4 (Wide Cinematic)"},
    "50mm_f1.2": {"focal_length": 50.0, "f_number": 1.2, "name": "50mm f/1.2 (Standard Cinematic)"},
    "50mm_f1.4": {"focal_length": 50.0, "f_number": 1.4, "name": "50mm f/1.4 (Standard)"},
    "85mm_f1.4": {"focal_length": 85.0, "f_number": 1.4, "name": "85mm f/1.4 (Portrait Cinematic)"},
    "85mm_f1.8": {"focal_length": 85.0, "f_number": 1.8, "name": "85mm f/1.8 (Portrait)"},
    "135mm_f2.0": {"focal_length": 135.0, "f_number": 2.0, "name": "135mm f/2.0 (Telephoto Cinematic)"},
}


# ============================================================
#  CoC 물리 계산 엔진
# ============================================================
class CircleOfConfusion:
    """
    실제 시네 렌즈의 광학 공식으로 각 픽셀의 블러 반경을 계산합니다.
    
    Thin Lens Model에 기반하되, Google 논문의 실용적 개선 사항을 적용:
    1. 초점면 주변 sharp zone (d_zero) 도입 — 초점 주변의 자연스러운 선명 영역
    2. 최대 블러 반경 제한 (r_max) — 과도한 블러 방지
    3. 거리 기반 점진적 전환 — 급격한 경계 없는 부드러운 DOF 전이
    """
    
    def __init__(self, focal_length_mm=50.0, f_number=1.4, 
                 sensor_width_mm=36.0, image_width_px=1920,
                 focus_distance_m=5.0, d_zero_m=1.0, r_max_px=40):
        """
        Args:
            focal_length_mm: 렌즈 초점거리 (mm). 클수록 보케가 극적.
            f_number: 조리개 값. 작을수록(개방) 보케가 강해짐.
            sensor_width_mm: 센서 폭. Full Frame = 36mm.
            image_width_px: 출력 이미지 폭 (px).
            focus_distance_m: 초점 맞출 피사체 거리 (m).
            d_zero_m: 초점면 전후로 선명하게 유지할 구간 폭 (m).
            r_max_px: 최대 보케 블러 반경 (px). 성능과 시각적 극단 제어.
        """
        self.focal_length_mm = focal_length_mm
        self.f_number = f_number
        self.sensor_width_mm = sensor_width_mm
        self.image_width_px = image_width_px
        self.focus_distance_m = focus_distance_m
        self.d_zero_m = d_zero_m
        self.r_max_px = r_max_px
        
        # 파생 상수 계산
        self.focal_length_m = focal_length_mm / 1000.0
        self.aperture_diameter_m = self.focal_length_m / f_number
        self.px_per_mm = image_width_px / sensor_width_mm
        
    def compute_blur_radius_map(self, depth_map_normalized, 
                                 depth_near_m=0.5, depth_far_m=80.0):
        """
        정규화된 뎁스맵(0~1)을 실제 물리 거리(m)로 변환한 뒤,
        각 픽셀의 CoC 기반 블러 반경을 계산합니다.
        
        Args:
            depth_map_normalized: 0.0(가까움)~1.0(멀리) 정규화 뎁스맵
            depth_near_m: 뎁스맵 0.0에 대응하는 최소 거리 (m)
            depth_far_m: 뎁스맵 1.0에 대응하는 최대 거리 (m)
            
        Returns:
            blur_radius_map: 각 픽셀의 블러 반경 (float, px 단위)
        """
        # Step 1: 정규화 뎁스 → 실제 거리(m) 매핑
        # 0.0(검은색) = Far, 1.0(흰색) = Near
        inv_near = 1.0 / depth_near_m
        inv_far = 1.0 / depth_far_m
        inv_depth = inv_far + (inv_near - inv_far) * depth_map_normalized
        depth_m = 1.0 / np.maximum(inv_depth, 1e-6)
        
        # Step 2: CoC 물리 공식 적용
        f = self.focal_length_m
        A = self.aperture_diameter_m
        d_focus = self.focus_distance_m
        
        # 정확한 물리 방정식: c = A * f * |depth_m - d_focus| / (depth_m * (d_focus - f))
        # (이는 c = (f^2 / N) * |S2 - S1| / (S2 * (S1 - f)) 와 수학적으로 완벽히 동일함)
        coc_m = A * f * np.abs(depth_m - d_focus) / (depth_m * (d_focus - f))
        
        # 얕은 심도를 극대화하여 1인 미디어 시네마틱 룩 완성 (배경 완벽 아웃포커싱)
        cinematic_multiplier = 2.5
        coc_m *= cinematic_multiplier
        
        # Step 3: 센서 좌표 → 픽셀 좌표 변환
        coc_px = coc_m * self.px_per_mm * 1000.0  # m → mm → px
        
        # Step 4: 초점면 주변 sharp zone 적용 (Google 논문 Eq.5 참조)
        # |d - d_focus| < d_zero 범위는 블러 0으로 강제
        distance_from_focus = np.abs(depth_m - d_focus)
        sharp_mask = np.maximum(0.0, distance_from_focus - self.d_zero_m) / \
                     np.maximum(distance_from_focus, 1e-6)
        coc_px = coc_px * sharp_mask
        
        # Step 5: 최대 반경 제한
        blur_radius = np.clip(coc_px, 0.0, self.r_max_px)
        
        return blur_radius


# ============================================================
#  광학 보케 렌더러
# ============================================================
class OpticalBokehRenderer:
    """
    CoC 반경 맵을 기반으로 각 픽셀에 가변 크기의 가우시안 블러를 적용하여
    실제 렌즈의 보케를 시뮬레이션합니다.
    """
    
    def __init__(self, num_layers=None):
        pass

    def render(self, image, blur_radius_map):
        """
        100% 원본 해상도를 유지하며, 각 픽셀별 정확한 블러 레벨을 선형 보간하여 렌더링합니다.
        
        Args:
            image: BGR 원본 프레임 (H, W, 3)
            blur_radius_map: 각 픽셀의 블러 반경 (H, W), float
            
        Returns:
            result: 보케가 적용된 프레임 (H, W, 3)
        """
        max_blur = int(np.max(blur_radius_map))
        if max_blur == 0:
            return image
            
        # 블러 레벨 생성 (1, 2, 4, 8, 16, 32, 64)
        levels = []
        valid_radii = []
        
        # 레벨 0: 원본(가장 선명함)
        levels.append(image.astype(np.float32))
        valid_radii.append(0)
        
        r = 1
        while r <= max_blur + 2:
            k = int(r) * 2 + 1
            blurred = cv2.GaussianBlur(image, (k, k), r / 2.0)
            levels.append(blurred.astype(np.float32))
            valid_radii.append(r)
            r = int(r * 1.5) + 1
            
        rendered = np.zeros_like(image, dtype=np.float32)
        radii_arr = np.array(valid_radii, dtype=np.float32)
        
        # 각 픽셀별로 적용할 상/하한 블러 레벨 인덱스 탐색
        idx = np.searchsorted(radii_arr, blur_radius_map)
        idx = np.clip(idx, 1, len(radii_arr) - 1)
        
        # 벡터 연산으로 모든 픽셀 보간
        for i in range(1, len(valid_radii)):
            mask = (idx == i)
            if not np.any(mask):
                continue
                
            lower_r = radii_arr[i - 1]
            upper_r = radii_arr[i]
            
            # 가중치 (0~1)
            w_up = (blur_radius_map[mask] - lower_r) / (upper_r - lower_r + 1e-6)
            w_up = np.clip(w_up, 0, 1)
            w_low = 1.0 - w_up
            
            w_up_3 = w_up[:, None]
            w_low_3 = w_low[:, None]
            
            rendered[mask] = levels[i - 1][mask] * w_low_3 + levels[i][mask] * w_up_3
            
        return np.clip(rendered, 0, 255).astype(np.uint8)


# ============================================================
#  통합 파이프라인
# ============================================================
class CinematicDoFPipeline:
    """
    전체 시네마틱 DOF 파이프라인을 관리합니다.
    
    흐름: 입력 영상 → 뎁스 추출 → CoC 계산 → 광학 보케 렌더링 → 출력 영상
    """
    
    def __init__(self, lens_preset="50mm_f1.4", focus_distance=5.0, 
                 num_layers=16, r_max=40, d_zero=1.0,
                 custom_focal=None, custom_aperture=None):
        """
        Args:
            lens_preset: LENS_PRESETS 중 하나, 또는 "custom"
            focus_distance: 초점 거리 (m)
            num_layers: 뎁스 레이어 수
            r_max: 최대 블러 반경 (px)
            d_zero: 초점 전후 선명 구간 (m)
            custom_focal: 커스텀 초점거리 (mm)
            custom_aperture: 커스텀 조리개 값
        """
        # 렌즈 파라미터 결정
        if custom_focal and custom_aperture:
            self.focal_length = float(custom_focal)
            self.f_number = float(custom_aperture)
            self.lens_name = f"{self.focal_length:.0f}mm f/{self.f_number}"
        elif lens_preset in LENS_PRESETS:
            preset = LENS_PRESETS[lens_preset]
            self.focal_length = preset["focal_length"]
            self.f_number = preset["f_number"]
            self.lens_name = preset["name"]
        else:
            self.focal_length = 50.0
            self.f_number = 1.4
            self.lens_name = "50mm f/1.4 (Default)"
        
        self.focus_distance = focus_distance
        self.num_layers = num_layers
        self.r_max = r_max
        self.d_zero = d_zero
        
        # 뎁스 추정 모델 로드
        print(f"  [모델 로딩] Depth Anything 모델을 로드합니다...")
        self.depth_pipe = pipeline(
            task="depth-estimation", 
            model="LiheYoung/depth-anything-small-hf"
        )
        print(f"  [모델 로딩] 완료!")
        
        # 렌더러 초기화 (CoC 엔진과 보케 렌더러는 프레임 처리 시 생성)
        self.bokeh_renderer = OpticalBokehRenderer(num_layers=num_layers)
        self.coc_engine = None  # 영상 해상도 확인 후 초기화
        
    def _extract_depth(self, frame_rgb):
        """단일 프레임의 뎁스맵 추출"""
        pil_img = Image.fromarray(frame_rgb)
        result = self.depth_pipe(pil_img)
        depth_array = np.array(result["depth"]).astype(np.float32) / 255.0
        return depth_array
    
    def process_video(self, input_path, output_path):
        """
        영상 전체를 처리합니다.
        
        Args:
            input_path: 입력 영상 경로
            output_path: 출력 영상 경로
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"❌ 영상을 열 수 없습니다: {input_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # CoC 엔진 초기화 (이미지 해상도에 맞춰)
        self.coc_engine = CircleOfConfusion(
            focal_length_mm=self.focal_length,
            f_number=self.f_number,
            sensor_width_mm=36.0,  # Full Frame 기준
            image_width_px=width,
            focus_distance_m=self.focus_distance,
            d_zero_m=self.d_zero,
            r_max_px=self.r_max
        )
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"\n{'='*60}")
        print(f"  🎬 Cinematic DoF Rendering Engine v1.0")
        print(f"{'='*60}")
        print(f"  렌즈: {self.lens_name}")
        print(f"  초점 거리: {self.focus_distance}m")
        print(f"  초점 선명 구간: ±{self.d_zero}m")
        print(f"  최대 보케 반경: {self.r_max}px")
        print(f"  뎁스 레이어: {self.num_layers}개")
        print(f"  입력: {input_path}")
        print(f"  출력: {output_path}")
        print(f"  해상도: {width}x{height} @ {fps:.1f}fps")
        print(f"  총 프레임: {total_frames}")
        print(f"{'='*60}\n")
        
        frame_idx = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            frame_start = time.time()
            
            # 1. BGR → RGB 변환 후 뎁스맵 추출
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth_normalized = self._extract_depth(rgb_frame)
            
            # 뎁스맵을 원본 해상도로 리사이즈
            if depth_normalized.shape[:2] != (height, width):
                depth_normalized = cv2.resize(depth_normalized, (width, height))
            
            # 2. CoC 블러 반경 계산
            blur_radius_map = self.coc_engine.compute_blur_radius_map(depth_normalized)
            
            # 3. 광학 보케 렌더링
            rendered = self.bokeh_renderer.render(frame, blur_radius_map)
            
            out.write(rendered)
            
            # 진행 상황 표시
            elapsed = time.time() - frame_start
            total_elapsed = time.time() - start_time
            avg_fps = frame_idx / total_elapsed
            remaining = (total_frames - frame_idx) / max(avg_fps, 0.01)
            
            sys.stdout.write(
                f"\r  🎥 프레임 {frame_idx}/{total_frames} "
                f"({frame_idx/total_frames*100:.1f}%) | "
                f"{elapsed:.2f}s/frame | "
                f"남은 시간: {remaining:.0f}s"
            )
            sys.stdout.flush()
        
        cap.release()
        out.release()
        
        total_time = time.time() - start_time
        print(f"\n\n  ✅ 렌더링 완료!")
        print(f"  총 소요 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
        print(f"  출력 파일: {output_path}\n")


# ============================================================
#  비교 모드: 기존 가우시안 vs CoC 보케
# ============================================================
def render_comparison(input_path, output_path, lens_preset="50mm_f1.4",
                      focus_distance=5.0):
    """
    기존 균일 가우시안 블러와 CoC 광학 보케를 나란히 비교하는 영상을 생성합니다.
    교수님 피칭용 데모 자료로 활용할 수 있습니다.
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 비교 영상은 원본 폭의 3배 (원본 | 가우시안 | CoC)
    out_width = width * 3
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, height))
    
    # 뎁스 모델 로드
    print(f"  [모델 로딩] Depth Anything 모델 로드...")
    depth_pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
    
    # CoC 엔진 & 보케 렌더러 초기화
    preset = LENS_PRESETS.get(lens_preset, LENS_PRESETS["50mm_f1.4"])
    coc = CircleOfConfusion(
        focal_length_mm=preset["focal_length"],
        f_number=preset["f_number"],
        image_width_px=width,
        focus_distance_m=focus_distance,
    )
    bokeh = OpticalBokehRenderer(num_layers=16)
    
    print(f"\n  🔬 비교 렌더링 시작: 원본 | 가우시안(기존) | CoC 보케(신규)")
    print(f"  렌즈: {preset['name']}\n")
    
    frame_idx = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # 뎁스맵 추출
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        depth_result = depth_pipe(pil_img)
        depth_norm = np.array(depth_result["depth"]).astype(np.float32) / 255.0
        if depth_norm.shape[:2] != (height, width):
            depth_norm = cv2.resize(depth_norm, (width, height))
        
        # --- 방법 1: 기존 균일 가우시안 (apply_dof.py 방식) ---
        mask = 1.0 - depth_norm  # 먼 곳 = 블러
        mask_3ch = np.dstack([mask] * 3)
        gaussian_blurred = cv2.GaussianBlur(frame, (45, 45), 0)
        gaussian_result = (frame * mask_3ch + gaussian_blurred * (1.0 - mask_3ch))
        gaussian_result = np.clip(gaussian_result, 0, 255).astype(np.uint8)
        
        # --- 방법 2: CoC 광학 보케 ---
        blur_map = coc.compute_blur_radius_map(depth_norm)
        coc_result = bokeh.render(frame, blur_map)
        
        # 라벨 추가
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_h = 40
        
        orig_labeled = frame.copy()
        cv2.putText(orig_labeled, "ORIGINAL", (10, label_h), font, 1.0, (255, 255, 255), 2)
        
        gauss_labeled = gaussian_result.copy()
        cv2.putText(gauss_labeled, "GAUSSIAN (Before)", (10, label_h), font, 1.0, (100, 100, 255), 2)
        
        coc_labeled = coc_result.copy()
        cv2.putText(coc_labeled, "CoC BOKEH (After)", (10, label_h), font, 1.0, (100, 255, 100), 2)
        
        # 3개를 가로로 합치기
        comparison = np.hstack([orig_labeled, gauss_labeled, coc_labeled])
        out.write(comparison)
        
        sys.stdout.write(f"\r  🎥 프레임 {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")
        sys.stdout.flush()
    
    cap.release()
    out.release()
    
    total_time = time.time() - start_time
    print(f"\n\n  ✅ 비교 렌더링 완료! ({total_time:.1f}초)")
    print(f"  출력 파일: {output_path}\n")


# ============================================================
#  CLI 인터페이스
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="🎬 Highway Star — Cinematic DoF Rendering Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
렌즈 프리셋:
  35mm_f1.4   - 와이드 시네마틱 (넓은 화각, 환경 포함)
  50mm_f1.4   - 표준 시네마틱 (자연스러운 원근감)
  50mm_f1.2   - 표준 극한 보케 (꿈결 같은 DOF)
  85mm_f1.4   - 인물 시네마틱 (극적인 배경 분리)
  85mm_f1.8   - 인물 표준 (자연스러운 보케)
  135mm_f2.0  - 텔레포토 시네마틱 (압축된 원근, 강한 보케)

사용 예시:
  python optical_dof_engine.py --input video.mp4 --lens 85mm_f1.4
  python optical_dof_engine.py --input video.mp4 --focal 50 --aperture 1.2
  python optical_dof_engine.py --input video.mp4 --compare --lens 50mm_f1.4
        """
    )
    
    parser.add_argument("--input", "-i", required=True, help="입력 영상 경로")
    parser.add_argument("--output", "-o", default=None, help="출력 영상 경로 (기본: 자동 생성)")
    parser.add_argument("--lens", "-l", default="50mm_f1.4", 
                        choices=list(LENS_PRESETS.keys()),
                        help="시네 렌즈 프리셋 선택 (기본: 50mm_f1.4)")
    parser.add_argument("--focal", type=float, default=None, help="커스텀 초점거리 (mm)")
    parser.add_argument("--aperture", type=float, default=None, help="커스텀 조리개 값 (f/N)")
    parser.add_argument("--focus-distance", "-fd", type=float, default=5.0, 
                        help="초점 거리 (m, 기본: 5.0)")
    parser.add_argument("--layers", type=int, default=16, help="뎁스 레이어 수 (기본: 16)")
    parser.add_argument("--r-max", type=int, default=40, help="최대 블러 반경 px (기본: 40)")
    parser.add_argument("--d-zero", type=float, default=1.0, 
                        help="초점 주변 선명 구간 m (기본: 1.0)")
    parser.add_argument("--compare", "-c", action="store_true",
                        help="기존 가우시안 vs CoC 비교 모드")
    
    args = parser.parse_args()
    
    # 출력 경로 자동 생성
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        if args.compare:
            args.output = f"{base}_comparison.mp4"
        else:
            args.output = f"{base}_cinematic_dof.mp4"
    
    if args.compare:
        # 비교 모드
        render_comparison(
            args.input, args.output,
            lens_preset=args.lens,
            focus_distance=args.focus_distance,
        )
    else:
        # 일반 렌더링 모드
        pipeline_obj = CinematicDoFPipeline(
            lens_preset=args.lens,
            focus_distance=args.focus_distance,
            num_layers=args.layers,
            r_max=args.r_max,
            d_zero=args.d_zero,
            custom_focal=args.focal,
            custom_aperture=args.aperture,
        )
        pipeline_obj.process_video(args.input, args.output)


if __name__ == "__main__":
    main()
