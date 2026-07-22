import librosa
import numpy as np
import os

# 1. AI로 분리된 드럼 파일 경로 지정 (Demucs가 생성한 경로에 맞춰줘)
# 보통 separated_tracks/htdemucs/Deep_Purple_-_Highway_Star/drums.wav 구조야.
# 만약 폴더 이름에 공백이 있다면 실제 폴더명을 확인하고 아래 경로를 수정해줘!
drum_audio_path = 'separated_tracks/htdemucs/Deep Purple - Highway Star/drums.wav'

if not os.path.exists(drum_audio_path):
    print(f"⚠️ 드럼 파일 경로를 찾을 수 없습니다: {drum_audio_path}")
    print("💡 폴더 안으로 들어가서 drums.wav 파일이 있는 정확한 경로로 오타를 수정해 주세요.")
else:
    print("⏳ 드럼 트랙에서 정밀 타격점(Onset)을 분석하는 중입니다...")
    y, sr = librosa.load(drum_audio_path)
    
    # 드럼 주파수의 에너지 급증 구간(Onset Strength) 계산
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    frames = range(len(onset_env))
    timestamps = librosa.frames_to_time(frames, sr=sr)
    
    # 텐션 강도를 0 ~ 1 사이로 정규화
    max_onset = np.max(onset_env) if np.max(onset_env) > 0 else 1
    normalized_onset = onset_env / max_onset
    
    print("\n🥁 [드럼 타격점 분석 결과] 임팩트가 강한 상위 15개 타이밍:")
    
    # 에너지 강도가 0.5 이상인 진짜 강한 드럼 타격 타이밍만 골라내기
    strong_beats = []
    for time, strength in zip(timestamps, normalized_onset):
        if strength > 0.5:
            strong_beats.append((time, strength))
            
    # 초반 15개만 터미널에 출력해서 확인
    for i, (time, strength) in enumerate(strong_beats[:15]):
        print(f"[{i+1}번째 타격] 시간: {time:.2f}초 | 타격 강도: {strength*100:.1f}%")
        
    print(f"\n📁 총 {len(strong_beats)}개의 드럼 하이라이트 타격 포인트를 확보했습니다.")