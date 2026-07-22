import cv2
import librosa
import numpy as np

# 1. 파일 경로 설정
audio_path = 'Deep Purple - Highway Star.mp3'
video_path = 'sample_38.mp4'
output_path = 'final_processed_video.mp4'

print("⏳ 1단계: 오디오 타임라인 데이터베이스 로드 중...")
y, sr = librosa.load(audio_path)
rms = librosa.feature.rms(y=y)[0]
rms_timestamps = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=512)
normalized_rms = rms / (np.max(rms) if np.max(rms) > 0 else 1)

onset_env = librosa.onset.onset_strength(y=y, sr=sr)
onset_timestamps = librosa.frames_to_time(range(len(onset_env)), sr=sr, hop_length=512)
normalized_onset = onset_env / (np.max(onset_env) if np.max(onset_env) > 0 else 1)

print("⏳ 2단계: 비디오 연출 엔진 가동...")
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

current_frame = 0
accumulated_time = 0.0  # 정밀 싱크 보정용 타임스탬프 누적기

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 결과물 영상의 실제 시간 흐름에 맞춰 오디오 데이터 매핑
    rms_idx = np.argmin(np.abs(rms_timestamps - accumulated_time))
    onset_idx = np.argmin(np.abs(onset_timestamps - accumulated_time))
    
    tension = normalized_rms[rms_idx]
    onset_strength = normalized_onset[onset_idx]
    
    # [설계 규칙 1 & 3] 거시적 속도 조절 및 가상 YOLO 클라이맥스 구간 매칭
    # 22.5초 터널 탈출(Climax) 부근인 21.0초~24.0초 사이에는 극적인 0.4배속 슬로우 모션 연출
    if 21.0 <= accumulated_time <= 24.0:
        speed_factor = 0.4
    else:
        # 일반 구간: 기타 멜로디 텐션에 맞춰 최고 2.0배속까지 가속 주행
        speed_factor = 1.0 + (tension * 1.0)
    
    # [설계 규칙 2] 미시적 타격점 제어 (Beat Flash)
    # 우리가 수정한 20.8초의 수학적 오차(92.1% 튀는 현상)는 강제로 걸러내어 엇박자 전면 차단!
    if abs(accumulated_time - 20.8) < 0.2:
        onset_strength = 0.2  # 일반적인 주변부 세기로 강제 평탄화 보정
        
    # 진짜 정박 드럼 타격 강도가 0.45 이상으로 시원하게 터질 때만 화면 플래시 라이팅 연출
    if onset_strength > 0.45:
        flash_val = int(onset_strength * 35)
        frame = cv2.add(frame, np.full(frame.shape, flash_val, dtype=np.uint8))
    
    # 가변 속도(Time-Warping) 공식을 프레임 중복/스킵 방식으로 구현
    if speed_factor < 1.0:
        # 슬로우 모션: 동일 프레임을 반복하여 저장
        repeat_count = int(1.0 / speed_factor)
        for _ in range(repeat_count):
            out.write(frame)
            accumulated_time += (1.0 / fps)
    else:
        # 가속 주행: 배속에 맞춰 프레임을 스킵하며 저장
        if current_frame % int(speed_factor) == 0:
            out.write(frame)
            accumulated_time += (1.0 / fps)
            
    current_frame += 1

cap.release()
out.release()
print(f"\n🎉 멀티모달 싱크 비디오 빌드 완료: {output_path}")
print("💡 퀵타임이나 프리미어에서 Highway Star.mp3를 0초 시작점에 딱 맞춰서 얹어보세요!")