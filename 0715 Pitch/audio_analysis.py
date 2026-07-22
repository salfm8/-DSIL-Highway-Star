import librosa
import numpy as np

# 1. 오디오 파일 로드
audio_path = 'Deep Purple - Highway Star.mp3'

print("⏳ 음악 파일을 분석 중입니다. 잠시만 기다려주세요...")
y, sr = librosa.load(audio_path)

# 2. 템포(BPM)와 비트의 프레임 위치 추출 (최신 버전 대응)
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

# 최신 librosa 버전에서는 tempo가 배열로 나오므로 첫 번째 요소 추출
if isinstance(tempo, np.ndarray):
    tempo_val = tempo[0]
else:
    tempo_val = tempo

print(f"\n🎵 분석된 음악 템포 (BPM): {tempo_val:.2f}")

# 3. 프레임 단위를 실제 영상 편집에 쓸 수 있는 '초(seconds)' 단위 타임스탬프로 변환
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# 4. 음악 시작 후 초반 15개의 비트가 터지는 정확한 타이밍(초) 출력
print("\n⏱️ 비트가 터지는 하이라이트 타임스탬프 (상위 15개):")
for i, time in enumerate(beat_times[:15]):
    print(f"[{i+1}번째 비트] {time:.2f}초")