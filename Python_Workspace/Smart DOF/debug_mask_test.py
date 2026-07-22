"""
디버그: 첫 프레임에서 뎁스맵, 객체 마스크, 블러 맵을 저장하여
실제로 무엇이 잡히고 있는지 눈으로 확인하기 위한 스크립트.
"""
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline as hf_pipeline

# 영상 첫 프레임 읽기
cap = cv2.VideoCapture("../0720 Pitch/sample_38_short.mp4")
ret, frame = cap.read()
cap.release()

h, w = frame.shape[:2]
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# 뎁스 추출
print("뎁스 추정 중...")
depth_pipe = hf_pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf")
pil_img = Image.fromarray(rgb)
result = depth_pipe(pil_img)
depth = np.array(result["depth"]).astype(np.float32) / 255.0
if depth.shape[:2] != (h, w):
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)

# Guided Filter
guide = rgb.astype(np.float32) / 255.0
depth = cv2.ximgproc.createGuidedFilter(guide, radius=8, eps=0.01).filter(depth)
depth = np.clip(depth, 0.0, 1.0)

# 뎁스맵 저장
depth_vis = (depth * 255).astype(np.uint8)
depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
cv2.imwrite("debug_01_depth_map.png", depth_color)
print(f"뎁스맵 저장 완료. 타겟(225,125)의 뎁스값: {depth[125, 225]:.4f}")
print(f"뎁스 범위: min={depth.min():.4f}, max={depth.max():.4f}")

# 타겟 포인트 주변 뎁스 분포 확인
tx, ty = 225, 125
roi = depth[max(0,ty-20):min(h,ty+20), max(0,tx-20):min(w,tx+20)]
print(f"타겟 주변 40x40 뎁스 평균: {roi.mean():.4f}, std: {roi.std():.4f}")

# Flood Fill 테스트 (여러 threshold로)
for threshold in [5, 10, 15, 20, 30]:
    depth_8bit = (depth * 255).astype(np.uint8)
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood_flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
    
    cv2.floodFill(depth_8bit, mask, (tx, ty), 255, (threshold,), (threshold,), flood_flags)
    
    obj_mask = mask[1:-1, 1:-1]
    
    # 모폴로지
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, kernel)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, kernel)
    
    # 마스크를 원본 위에 오버레이
    overlay = frame.copy()
    overlay[obj_mask > 0] = overlay[obj_mask > 0] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
    cv2.circle(overlay, (tx, ty), 8, (0, 0, 255), -1)
    
    pixel_count = np.sum(obj_mask > 0)
    percent = pixel_count / (w * h) * 100
    cv2.putText(overlay, f"Threshold={threshold}, Pixels={pixel_count} ({percent:.1f}%)", 
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imwrite(f"debug_02_mask_threshold_{threshold}.png", overlay)
    print(f"  Threshold {threshold}: {pixel_count} pixels ({percent:.1f}%)")

# 대안: Connected Component 방식
print("\n--- Connected Component 방식 테스트 ---")
target_depth = depth[ty, tx]
for thresh_val in [0.02, 0.04, 0.06, 0.08, 0.10]:
    binary = (np.abs(depth - target_depth) < thresh_val).astype(np.uint8) * 255
    
    # Connected components
    num_labels, labels = cv2.connectedComponents(binary)
    target_label = labels[ty, tx]
    
    if target_label > 0:
        obj_mask = (labels == target_label).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, kernel)
        
        overlay = frame.copy()
        overlay[obj_mask > 0] = overlay[obj_mask > 0] * 0.5 + np.array([0, 200, 255], dtype=np.uint8) * 0.5
        cv2.circle(overlay, (tx, ty), 8, (0, 0, 255), -1)
        
        pixel_count = np.sum(obj_mask > 0)
        percent = pixel_count / (w * h) * 100
        cv2.putText(overlay, f"DepthThresh={thresh_val:.2f}, Pixels={pixel_count} ({percent:.1f}%)", 
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imwrite(f"debug_03_cc_thresh_{thresh_val:.2f}.png", overlay)
        print(f"  DepthThresh {thresh_val:.2f}: {pixel_count} pixels ({percent:.1f}%), labels={num_labels}")
    else:
        print(f"  DepthThresh {thresh_val:.2f}: 타겟이 배경(0)에 속함 - 마스크 생성 불가")

print("\n디버그 이미지 저장 완료!")
