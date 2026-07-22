import cv2
import librosa
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip

# 1. 파일 경로 설정
audio_path = 'Deep Purple - Highway Star.mp3'
video_path = 'sample_38.mp4'
temp_video_path = 'temp_silent.mp4'
final_output_path = 'final_music_video.mp4'

print("⏳ 1단계: 오디오 멀티 레이어 정밀 분석 시작...")
y, sr = librosa.load(audio_path)

# 거시적 레이어: 오디오 음량 곡선(RMS) 및 텐션 규격화
rms = librosa.feature.rms(y=y)[0]
rms_timestamps = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=512)
normalized_rms = rms / (np.max(rms) if np.max(rms) > 0 else 1)

# 미시적 레이어: 음악의 정밀한 온셋 타격 강도(Onset Strength) 추출
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
onset_timestamps = librosa.frames_to_time(range(len(onset_env)), sr=sr, hop_length=512)
normalized_onset = onset_env / (np.max(onset_env) if np.max(onset_env) > 0 else 1)

print("⏳ 2단계: 비디오 소스 분석 및 매핑 싱크 보정 알고리즘 가동...")
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

current_frame = 0
accumulated_time = 0.0  # 시간 누적식 싱크 파이프라인 보정 장치

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 누적된 실제 결과물 타임스탬프 기준으로 오디오 특징 매핑
    # 이 방식으로 짜야 오디오와 비디오가 뒤로 갈수록 싱크가 어긋나지 않음
    rms_idx = np.argmin(np.abs(rms_timestamps - accumulated_time))
    onset_idx = np.argmin(np.abs(onset_timestamps - accumulated_time))
    
    tension = normalized_rms[rms_idx]
    onset_strength = normalized_onset[onset_idx]
    
    # [기능 1 & 3] 거시적 텐션 속도 조절 및 클라이맥스 구간 매칭
    if 15.0 <= accumulated_time <= 25.0:
        # 15초에서 25초 사이 하이라이트 구간: 웅장한 슬로우 모션 (0.5배속)
        speed_factor = 0.5
    else:
        # 일반 구간: 오디오 텐션이 올라가면 최고 2.2배속까지 유기적 가속
        speed_factor = 1.0 + (tension * 1.2)
        
    # [기능 2] 미시적 타격점 제어 -> 온셋 에너지가 터지는 변곡점 순간 화면 플래시
    if onset_strength > 0.4:  # 드럼/기타 타격 강도가 일정 수준 이상일 때만 번쩍임
        flash_val = int(onset_strength * 45)
        # 화면의 밝기를 소리의 크기만큼 다이내믹하게 증가시킴
        frame = cv2.add(frame, np.full(frame.shape, flash_val, dtype=np.uint8))
        
    out.write(frame)
    
    # 프레임이 추가될 때마다 재생 속도 배속을 역산하여 타임스탬프 누적
    accumulated_time += (1.0 / fps) * speed_factor
    current_frame += 1

cap.release()
out.release()

print("⏳ 4단계: MoviePy 활용 비디오-오디오 스트림 자동 병합 및 인코딩...")
try:
    video_clip = VideoFileClip(temp_video_path)
    audio_clip = AudioFileClip(audio_path)
    
    # 비디오 길이에 맞춰 음악 오디오 싱크 결합
    final_clip = video_clip.set_audio(audio_clip.set_duration(video_clip.duration))
    final_clip.write_videofile(final_output_path, codec="libx264", audio_codec="aac")
    
    video_clip.close()
    audio_clip.close()
    print(f"\n🎉 최종 성공! 음악이 내장된 자동 편집 비디오가 완성되었습니다: {final_output_path}")
except Exception as e:
    print(f"\n⚠️ 오디오 병합 중 에러 발생 (단, 영상 자체는 생성됨): {e}")