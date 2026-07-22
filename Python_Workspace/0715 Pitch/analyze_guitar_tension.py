import librosa
import numpy as np
import os

# 1. Demucs가 분리해 준기타/기타 악기 트랙(other.wav) 경로 지정
guitar_audio_path = 'separated_tracks/htdemucs/Deep Purple - Highway Star/other.wav'

if not os.path.exists(guitar_audio_path):
    print(f"⚠️ 기타/기타 트랙 파일을 찾을 수 없습니다: {guitar_audio_path}")
else:
    print("⏳ 기타 트랙에서 거시적 텐션 곡선(RMS 에너지)을 정밀 분석 중입니다...")
    y, sr = librosa.load(guitar_audio_path)
    
    # 멜로디의 거시적인 흐름(음량 에너지)을 계산
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    timestamps = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
    
    # 텐션 값을 0 ~ 100% 사이로 규격화
    max_rms = np.max(rms) if np.max(rms) > 0 else 1
    normalized_tension = (rms / max_rms) * 100
    
    print("\n🎸 [기타 트랙 분석 완료] 주요 타임라인별 에너지 텐션 변화 (5초 간격):")
    
    # 전체 시간대 중에서 5초 간격으로 기타 에너지가 어떻게 변하는지 샘플링 출력
    for time, tension in zip(timestamps, normalized_tension):
        if int(time * 100) % 500 == 0 and time < 30:  # 초반 30초까지만 5초 간격 출력
            print(f"⏱️ [{time:4.1f}초 시점] 기타/멜로디 텐션 강도: {tension:5.1f}%")
            
    # 전체 텐션의 평균을 구해, 평균보다 에너지가 강한 '기타 메인 하이라이트 구간' 식별
    avg_tension = np.mean(normalized_tension)
    print(f"\n📈 기타 트랙 평균 텐션: {avg_tension:.1f}%")
    print("💡 이 데이터를 기반으로 평균 텐션 이상으로 휘몰아치는 구간에서 영상 앵글 변곡점을 트리거합니다.")