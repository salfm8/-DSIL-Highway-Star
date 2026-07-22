import os
import subprocess

# 1. 분리할 오디오 파일 경로 지정
audio_path = 'Deep Purple - Highway Star.mp3'
output_dir = 'separated_tracks'

print("⏳ AI 음원 분리 모델(Demucs)을 구동 중입니다...")
print("🎵 드럼, 베이스, 보컬, 기타 트랙을 정밀하게 분리하고 있습니다. 잠시만 기다려주세요...")

# 2. 시스템 명령어로 demucs 가동
# htdemucs 모델을 사용하여 지정된 출력 폴더로 4개 트랙 분리 실행
cmd = f'./.venv/bin/demucs -n htdemucs -o "{output_dir}" "{audio_path}"'

try:
    # 파이썬 내부에서 AI 음원 분리 프로세스 실행
    subprocess.run(cmd, shell=True, check=True)
    print(f"\n🎉 음원 분리가 완벽하게 완료되었습니다!")
    print(f"📁 결과물 저장 위치: {output_dir}/htdemucs/Deep Purple - Highway Star/")
    print("💡 폴더 안에 drums.wav, 기타.wav 등의 파일이 생성되었는지 확인해 주세요.")
except Exception as e:
    print(f"\n⚠️ 음원 분리 중 에러가 발생했습니다: {e}")