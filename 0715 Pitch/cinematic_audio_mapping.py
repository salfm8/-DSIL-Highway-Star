import cv2
import librosa
import numpy as np

# 1. 파일 경로 설정
audio_path = 'Deep Purple - Highway Star.mp3'
video_path = 'sample_38.mp4'
output_path = 'output_cinematic.mp4'

print("⏳ 1단계: 오디오 멀티 레이어 분석 중 (텐션 곡선 & 비트 추출)...")
y, sr = librosa.load(audio_path)

# 거시적 레이어: 오디오 음량 에너지(RMS) 추출 및 규격화 (Tension Curve)
rms = librosa.feature.rms(y=y)[0]
rms_timestamps = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=512)
max_rms = np.max(rms) if np.max(rms) > 0 else 1
normalized_rms = rms / max_rms  # 0 ~ 1 사이 값으로 규격화

# 미시적 레이어: 타격점(Beat Tracking) 추출
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

print("⏳ 2단계: 비디오 사양 확인 및 렌더링 준비...")
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("⏳ 3단계: 멀티모달 시네마틱 매핑 렌더링 시작...")
current_frame = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    current_time = current_frame / fps
    
    # [기능 1] 오디오 텐션 곡선 매핑 -> 텐션에 비례하여 부드러운 가속 (1.0배속 ~ 2.5배속)
    # 현재 시간과 가장 가까운 RMS 인덱스 찾기
    rms_idx = np.argmin(np.abs(rms_timestamps - current_time))
    current_tension = normalized_rms[rms_idx]
    
    # [기능 3] 구간 매칭 시뮬레이션 -> 곡의 특정 빌드업/클라이맥스 구간(예: 15초~25초 사이)에는 웅장한 슬로우 모션
    if 15.0 <= current_time <= 25.0:
        speed_multiplier = 0.5  # 클라이맥스 구간 슬로우
    else:
        speed_multiplier = 1.0 + (current_tension * 1.5)  # 잔잔하면 1배속, 음악 텐션 높으면 최대 2.5배속
    
    # 배속에 맞춰 프레임을 선택적으로 저장 (가변 속도 제어 로직)
    # 재생 속도가 빨라질 때 프레임을 스킵하여 배속을 구현함
    skip_rate = max(1, int(speed_multiplier))
    if current_frame % skip_rate != 0:
        current_frame += 1
        continue

    # [기능 2] 미시적 타격점 매핑 -> 비트 타격 순간 화면 플래시 조명 효과
    closest_beat_idx = np.argmin(np.abs(beat_times - current_time))
    time_to_beat = np.abs(beat_times[closest_beat_idx] - current_time)
    
    # 비트가 탁 치는 시점(0.06초 이내)에는 화면에 화사한 플래시 라이팅 연출
    if time_to_beat < 0.06:
        # NumPy 브로드캐스팅을 활용해 화면 전체 밝기를 증가시킴 (최대 40픽셀)
        flash_intensity = int((0.06 - time_to_beat) / 0.06 * 40)
        frame = cv2.add(frame, np.full(frame.shape, flash_intensity, dtype=np.uint8))
        
    out.write(frame)
    current_frame += 1

cap.release()
out.release()
print(f"\n🎉 멀티모달 시네마틱 주행 영상 생성 완료: {output_path}")