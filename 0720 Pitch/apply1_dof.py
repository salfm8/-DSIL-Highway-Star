import cv2
import numpy as np

# 경로 설정 (터미널에서 확인했던 절대 경로 적용)
original_video_path = '/Users/shinmireu/Desktop/CAU/Project/[DSIL]Highway Star/Python_Workspace/0720 Pitch/sample_38_short.mp4'
depth_video_path = '/Users/shinmireu/Desktop/CAU/Project/[DSIL]Highway Star/Python_Workspace/0720 Pitch/depth_output.mp4'
output_path = '/Users/shinmireu/Desktop/CAU/Project/[DSIL]Highway Star/Python_Workspace/0720 Pitch/output1_dof.mp4'

cap_orig = cv2.VideoCapture(original_video_path)
cap_depth = cv2.VideoCapture(depth_video_path)

width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap_orig.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("원거리(배경) 초점 데이터 렌더링을 시작합니다...")

while cap_orig.isOpened() and cap_depth.isOpened():
    ret_orig, frame_orig = cap_orig.read()
    ret_depth, frame_depth = cap_depth.read()

    if not ret_orig or not ret_depth:
        break

    # 1. 뎁스맵 흑백 변환
    gray_depth = cv2.cvtColor(frame_depth, cv2.COLOR_BGR2GRAY)

    # 2. 마스크 반전 (핵심 변경 포인트)
    # 1.0에서 뎁스 값을 빼주어, 원래 밝았던 곳(가까운 곳)을 어둡게 만들고 블러 처리함
    mask = 1.0 - (gray_depth.astype(float) / 255.0)
    mask = np.dstack([mask] * 3)

    # 3. 가우시안 블러 적용
    blurred_frame = cv2.GaussianBlur(frame_orig, (45, 45), 0)

    # 4. 원본과 블러 합성
    blended = (frame_orig * mask) + (blurred_frame * (1.0 - mask))
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    out.write(blended)

cap_orig.release()
cap_depth.release()
out.release()
cv2.destroyAllWindows()

print("렌더링 완료! output1_dof.mp4 파일을 확인하세요.")