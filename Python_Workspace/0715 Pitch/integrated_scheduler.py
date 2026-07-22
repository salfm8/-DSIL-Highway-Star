import librosa
import numpy as np
import os

# 1. 분리된 악기 파일 경로
drum_path = 'separated_tracks/htdemucs/Deep Purple - Highway Star/drums.wav'
guitar_path = 'separated_tracks/htdemucs/Deep Purple - Highway Star/other.wav'

print("⏳ 1단계: 확보된 드럼 및 기타 트랙 데이터 로드 중...")
y_drum, sr_drum = librosa.load(drum_path)
y_guitar, sr_guitar = librosa.load(guitar_path)

# 2. 정밀 피처 추출 (드럼 타격점 & 기타 텐션 곡선)
onset_env = librosa.onset.onset_strength(y=y_drum, sr=sr_drum)
rms_guitar = librosa.feature.rms(y=y_guitar)[0]

# 시간 축 생성
timestamps_drum = librosa.frames_to_time(range(len(onset_env)), sr=sr_drum)
timestamps_guitar = librosa.frames_to_time(range(len(rms_guitar)), sr=sr_guitar)

# 0 ~ 1 사이 규격화
norm_onset = onset_env / (np.max(onset_env) if np.max(onset_env) > 0 else 1)
norm_guitar = rms_guitar / (np.max(rms_guitar) if np.max(rms_guitar) > 0 else 1)

# [가상 데이터] 내일 회의용 YOLOv8 도로 지형 변곡점 레이어 생성
# 실제 연구에서는 YOLO 모델이 뱉은 터널/교량 진입 초(seconds) 데이터가 들어오게 됨
yolo_landmarks = {
    15.0: "터널 진입",
    22.5: "터널 탈출 (시각적 변곡점 Climax)"
}

print("⏳ 2단계: 추출된 멀티모달 데이터를 'audio_features.txt' 파일로 시각화 저장 중...")
with open("audio_features.txt", "w", encoding="utf-8") as f:
    f.write("=== 멀티모달 자동 편집용 통합 데이터 타임라인 ===\n\n")
    
    # 0초부터 30초까지 0.1초 간격으로 스케줄러가 읽을 마스터 테이블 매핑
    for t in np.arange(0.0, 30.0, 0.1):
        # 가장 가까운 시간대의 오디오 인덱스 매칭
        d_idx = np.argmin(np.abs(timestamps_drum - t))
        g_idx = np.argmin(np.abs(timestamps_guitar - t))
        
        drum_power = norm_onset[d_idx] * 100
        guitar_tension = norm_guitar[g_idx] * 100
        
        # 해당 시간에 지형 변곡점이 있는지 체크
        landmark_info = ""
        for landmark_time, name in yolo_landmarks.items():
            if abs(landmark_time - t) < 0.15:
                landmark_info = f" 📍 [YOLO 감지: {name}]"
        
        # 텍스트 파일에 타임라인 데이터 한 줄씩 기록
        log_line = f"⏱️ [{t:4.1f}초] | 🥁 드럼 타격: {drum_power:5.1f}% | 🎸 기타 텐션: {guitar_tension:5.1f}%{landmark_info}\n"
        f.write(log_line)

print("\n🎉 3단계: 통합 매핑 스케줄러 가상 시뮬레이션 작동 원리 검증:")

# 스케줄러가 타임라인을 훑으며 명령을 내리는 매핑 엔진 예시 (0.5초 간격 샘플링 출력)
for t in np.arange(0.0, 25.0, 0.5):
    d_idx = np.argmin(np.abs(timestamps_drum - t))
    g_idx = np.argmin(np.abs(timestamps_guitar - t))
    drum_power = norm_onset[d_idx] * 100
    guitar_tension = norm_guitar[g_idx] * 100
    
    # 스케줄러의 핵심 조건문(Rule) 논리 구조
    if abs(22.5 - t) < 0.3:
        print(f"[{t:4.1f}초] 🚨 [스케줄러 명령] YOLO 터널 탈출 감지! 음악 Climax와 일치화 ➔ 0.5배속 시네마틱 슬로우 발동!")
    elif drum_power > 60.0:
        print(f"[{t:4.1f}초] ⚡ [스케줄러 명령] 강한 드럼 타격 ({drum_power:.1f}%) 감지 ➔ 비디오 프레임에 강한 Beat Flash 레이어 가산!")
    elif guitar_tension > 55.0:
        print(f"[{t:4.1f}초] 🚀 [스케줄러 명령] 기타 루프 고조 ({guitar_tension:.1f}%) 감지 ➔ 주행 배속 2.0배속으로 부드러운 가속!")