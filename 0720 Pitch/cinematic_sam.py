import cv2
import numpy as np
from ultralytics import SAM

# 절대 경로 설정
video_path = '/Users/shinmireu/Desktop/CAU/Project/[DSIL]Highway Star/Python_Workspace/0720 Pitch/sample_38_short.mp4'

# 가볍고 빠른 MobileSAM 모델 로드
model = SAM('mobile_sam.pt')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

clicked_points = []

# 마우스 클릭 이벤트를 처리하는 함수
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        print(f"좌표 선택됨: X={x}, Y={y}")

# OpenCV 창 생성 및 마우스 이벤트 연결
cv2.namedWindow('Cinematic Mode - Click Object')
cv2.setMouseCallback('Cinematic Mode - Click Object', mouse_callback)

print("화면에서 초점을 맞출 객체를 마우스로 클릭하고 스페이스바를 누르세요.")

while True:
    temp_frame = frame.copy()
    
    # 클릭한 위치에 빨간 점 표시
    for p in clicked_points:
        cv2.circle(temp_frame, (p[0], p[1]), 5, (0, 0, 255), -1)
        
    cv2.imshow('Cinematic Mode - Click Object', temp_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 32 and len(clicked_points) > 0: # 스페이스바(32)를 누르면 추출 시작
        break
    elif key == 27: # ESC(27) 누르면 종료
        cap.release()
        cv2.destroyAllWindows()
        exit()

print("SAM 모델이 선택한 객체의 영역을 추출하고 있습니다...")

# 선택한 좌표를 바탕으로 SAM 모델에 입력하여 객체 마스크(외곽선) 추출
results = model.predict(frame, points=clicked_points, labels=[1]*len(clicked_points))

# 마스크 데이터 정규화 및 리사이즈 (에러 해결을 위해 float32 변환 코드 추가)
mask = results[0].masks.data[0].cpu().numpy()
mask = mask.astype(np.float32) # OpenCV가 읽을 수 있도록 bool을 숫자로 변환
mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
mask_3ch = np.dstack([mask] * 3)

# 전체 화면에 강한 가우시안 블러 적용 (아웃포커싱 효과)
blurred_frame = cv2.GaussianBlur(frame, (55, 55), 0)

# 선택한 객체는 선명한 원본을, 나머지는 흐린 배경을 합성
cinematic_frame = (frame * mask_3ch) + (blurred_frame * (1.0 - mask_3ch))
cinematic_frame = np.clip(cinematic_frame, 0, 255).astype(np.uint8)

cv2.destroyWindow('Cinematic Mode - Click Object')
cv2.imshow('Result - Cinematic DoF', cinematic_frame)
print("시네마틱 효과 적용 완료! 결과창에서 아무 키나 누르면 종료됩니다.")

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()