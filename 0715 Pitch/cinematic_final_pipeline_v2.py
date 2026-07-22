import cv2
import librosa
import numpy as np
import os
import subprocess

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
accumulated_time = 0.0  # 시간 누적식 싱크 파이프라인 보정

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 누적된 실제 결과물 타임스탬프 기준으로 오디오 특징 매핑 (싱크 엇박자 방지)
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
        frame = cv2.add(frame, np.full(frame.shape, flash_val, dtype=np.uint8))
        
    out.write(frame)
    
    # 재생 속도 배속을 역산하여 타임스탬프 누적
    accumulated_time += (1.0 / fps) * speed_factor
    current_frame += 1

cap.release()
out.release()

print("⏳ 3단계: 시스템 명령어를 이용한 오디오-비디오 병합 작업 시작...")

# 기존 파일이 있다면 충돌 방지를 위해 삭제
if os.path.exists(final_output_path):
    os.remove(final_output_path)

# ffmpeg를 직접 호출하여 임시 영상에 원본 mp3 음원을 싱크에 맞춰 강제 인코딩 병합
cmd = f'ffmpeg -i "{temp_video_path}" -i "{audio_path}" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest "{final_output_path}" -y'

try:
    # 파이썬에서 시스템 셸 명령어로 ffmpeg 실행
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"\n🎉 최종 성공! 음악이 내장된 자동 편집 비디오가 완성되었습니다: {final_output_path}")
    
    # 임시 무음 영상 파일 삭제 정리
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
except Exception as e:
    print(f"\n⚠️ 오디오 합성 단계에서 시스템 에러가 발생했습니다: {e}")
    print("단, 음악이 없는 임시 영상('temp_silent.mp4')은 생성되었으니 개별적으로 확인 가능합니다.")