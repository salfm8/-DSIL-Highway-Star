import cv2
import torch
import numpy as np
from PIL import Image
from transformers import pipeline

# 뎁스 추정 모델 로드 (가볍고 빠른 모델 사용)
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

input_video_path = "sample_38_short.mp4" # 준비한 원본 영상 이름으로 변경
output_video_path = "depth_output.mp4"

cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print("뎁스 맵 추출을 시작합니다. 잠시만 기다려주세요...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # BGR 이미지를 RGB로 변환 후 PIL 객체로 생성
    color_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(color_img)
    
    # AI 모델을 통해 뎁스 맵 추론
    depth_result = pipe(pil_img)
    depth_img = depth_result["depth"]
    
    # 흑백 뎁스 맵을 다시 OpenCV가 읽을 수 있는 배열로 변환
    depth_array = np.array(depth_img)
    depth_array = cv2.cvtColor(depth_array, cv2.COLOR_GRAY2BGR)
    
    out.write(depth_array)

cap.release()
out.release()
print("뎁스 맵 영상 추출 완료: depth_output.mp4")