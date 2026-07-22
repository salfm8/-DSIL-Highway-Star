"""
기존 가우시안 DOF 영상을 별도 파일로 생성 (비교용)
+ 0720 Pitch의 apply_dof.py 방식과 동일한 로직
"""
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import sys

input_path = "../0720 Pitch/sample_38_short.mp4"
output_path = "step0_gaussian_dof.mp4"

# 뎁스 모델 로드
print("  [모델 로딩] Depth Anything 모델 로드...")
depth_pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"  🎥 기존 가우시안 DOF 렌더링 시작 ({total}프레임)")

idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    idx += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    depth_result = depth_pipe(pil_img)
    depth_norm = np.array(depth_result["depth"]).astype(np.float32) / 255.0
    if depth_norm.shape[:2] != (height, width):
        depth_norm = cv2.resize(depth_norm, (width, height))

    # 기존 방식: 가까운 곳 선명, 먼 곳 블러 (apply_dof.py와 동일)
    mask = depth_norm  # 가까운=밝=선명, 먼=어두움=블러
    # Depth Anything은 가까운=큰값이므로 그대로 사용
    mask_3ch = np.dstack([mask] * 3)
    blurred = cv2.GaussianBlur(frame, (45, 45), 0)
    result = (frame * mask_3ch + blurred * (1.0 - mask_3ch))
    result = np.clip(result, 0, 255).astype(np.uint8)

    out.write(result)
    sys.stdout.write(f"\r  프레임 {idx}/{total} ({idx/total*100:.0f}%)")
    sys.stdout.flush()

cap.release()
out.release()
print(f"\n  ✅ 완료: {output_path}")
