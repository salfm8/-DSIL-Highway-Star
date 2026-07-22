import cv2
import librosa
import numpy as np

# 1. 소스 파일 경로 설정
audio_path = 'Deep Purple - Highway Star.mp3'
video_path = 'sample_38.mp4'
output_path = 'output_warped.mp4'

print("⏳ 1단계: 음악 비트 분석 중...")
y, sr = librosa.load(audio_path)
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

print("⏳ 2단계: 비디오 소스 로드 및 사양 확인 중...")
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 결과물 비디오를 저장할 설정 (H.264 코덱 사용)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"📹 원본 영상 FPS: {fps:.2f} | 총 프레임 수: {frame_count}")
print("⏳ 3단계: 음악 비트 연동 가변 속도(Time-Warping) 렌더링 시작...")

current_frame = 0

# 영상의 모든 프레임을 순회하면서 속도 왜곡(Warping) 알고리즘 적용
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 현재 프레임의 시간(초) 계산
    current_time = current_frame / fps
    
    # 현재 시간과 가장 가까운 오디오 비트 사이의 거리 계산
    closest_beat_idx = np.argmin(np.abs(beat_times - current_time))
    time_to_beat = np.abs(beat_times[closest_beat_idx] - current_time)
    
    # [핵심 매핑 논리] 비트가 터지는 타이밍(거리 0.1초 이내)에는 프레임을 중복 추가하여 슬로우 효과, 
    # 비트 사이 구간에는 프레임을 건너뛰어 가속 효과 연출
    if time_to_beat < 0.08:
        # 비트 타격 시점: 화면을 극적으로 붙잡아두기 위해 같은 프레임을 2번 써서 0.5배속 슬로우 모션 연출
        out.write(frame)
        out.write(frame)
    else:
        # 일반 구간: 비트가 오기 전까지는 영상을 1.5배속으로 빠르게 전진시킴 (2프레임당 1프레임만 저장)
        if current_frame % 2 == 0:
            out.write(frame)
            
    current_frame += 1

cap.release()
out.release()
print(f"\n🎉 렌더링 완료! 새로운 결과물 파일이 생성되었습니다: {output_path}")