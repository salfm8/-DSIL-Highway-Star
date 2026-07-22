import cv2
import numpy as np

# 영상 경로 설정
original_video_path = 'sample_38_short.mp4'
depth_video_path = 'depth_output.mp4'
output_path = 'output_dof.mp4'

# 비디오 캡처 객체 생성
cap_orig = cv2.VideoCapture(original_video_path)
cap_depth = cv2.VideoCapture(depth_video_path)

# 비디오 속성 가져오기
width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap_orig.get(cv2.CAP_PROP_FPS)

# Mac 환경에 호환되는 코덱 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("심도 효과 데이터 렌더링을 시작합니다...")

while cap_orig.isOpened() and cap_depth.isOpened():
    ret_orig, frame_orig = cap_orig.read()
    ret_depth, frame_depth = cap_depth.read()

    if not ret_orig or not ret_depth:
        break

    # 1. 뎁스맵을 1채널 흑백으로 변환
    gray_depth = cv2.cvtColor(frame_depth, cv2.COLOR_BGR2GRAY)

    # 2. 마스크 정규화 (0.0 ~ 1.0)
    # 뎁스맵이 '가까운 곳=흰색(255), 먼 곳=검은색(0)'이라고 가정
    mask = gray_depth.astype(float) / 255.0
    
    # 초점을 먼 곳에 맞추고 싶다면 아래 주석을 해제해서 마스크를 반전시키면 됨
    # mask = 1.0 - mask 
    
    mask = np.dstack([mask] * 3) # 컬러 영상과 합성하기 위해 3채널로 확장

    # 3. 전체 화면에 강한 가우시안 블러 적용 (홀수만 입력 가능, 수치가 클수록 흐려짐)
    blurred_frame = cv2.GaussianBlur(frame_orig, (45, 45), 0)

    # 4. 원본과 블러 합성
    # 마스크 값이 1(흰색)에 가까울수록 원본, 0(검은색)에 가까울수록 블러 영상을 렌더링
    blended = (frame_orig * mask) + (blurred_frame * (1.0 - mask))
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    out.write(blended)

cap_orig.release()
cap_depth.release()
out.release()
cv2.destroyAllWindows()

print("렌더링 완료! output_dof.mp4 파일을 확인하세요.")