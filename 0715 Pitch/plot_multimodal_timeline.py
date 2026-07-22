import matplotlib.pyplot as plt
import numpy as np

# 1. 한글 폰트 설정 (맥북 시스템 폰트 활용)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 2. 가상의 시간 축 (0초부터 25초까지 0.1초 간격)
time_axis = np.arange(0.0, 25.0, 0.1)

# 3. 실제 음악 특징 데이터를 모사한 시각화용 베이스 데이터 생성
# (우리가 찾은 20.8초 윈도잉 오류 피크와 15초 이후 흐름을 정밀 반영)
guitar_tension = 40 + 20 * np.sin(time_axis * 0.5) + np.random.normal(0, 3, len(time_axis))
drum_onset = 15 + 15 * np.cos(time_axis * 1.2) + np.random.normal(0, 5, len(time_axis))
drum_onset = np.clip(drum_onset, 0, 100)
guitar_tension = np.clip(guitar_tension, 0, 100)

# 우리가 발견한 '문제의 20.8초 윈도잉 오류 피크' 강제 매핑 (데이터 한계 증명용)
idx_208 = np.argmin(np.abs(time_axis - 20.8))
drum_onset[idx_208] = 92.1

# 4. 그래프 그리기 시작
plt.figure(figsize=(12, 6))

# 기타 텐션 곡선 (거시 레이어)
plt.plot(time_axis, guitar_tension, label='🎸 기타/멜로디 텐션 (%)', color='#ff7f0e', alpha=0.8, linewidth=2)

# 드럼 타격점 (미시 레이어)
plt.stem(time_axis, drum_onset, linefmt='g-', markerfmt='go', label='🥁 드럼 타격 강도 (%)', basefmt=" ")

# 5. 핵심 연구 변곡점 레이어 표기 (YOLO 및 데이터 오류점)
# YOLO 터널 진입/탈출 시점 세팅
plt.axvline(x=15.0, color='blue', linestyle='--', linewidth=1.5)
plt.text(15.2, 85, '📍 YOLO: 터널 진입', color='blue', fontsize=10, fontweight='bold')

plt.axvline(x=22.5, color='purple', linestyle='--', linewidth=1.5)
plt.text(22.7, 85, '📍 YOLO: 터널 탈출\n(Climax 매핑 지점)', color='purple', fontsize=10, fontweight='bold')

# 우리가 찾아낸 20.8초 수학적 오차 지점 화살표 표기! (★이게 내일 발표 핵심 무기)
plt.annotate('⚠️ 수학적 연산 오차 발생\n(주파수 윈도잉 튀어오름 현상)', 
             xy=(20.8, 92.1), xytext=(16.0, 5.0),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6))

# 그래프 디테일 설정
plt.title('음악 리듬 - 도로 지형 데이터 통합 타임라인 (1차 프로토타입 분석 장표)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('시간 (초)', fontsize=11)
plt.ylabel('데이터 강도 (%)', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', fontsize=10)
plt.xlim(-0.5, 25.5)
plt.ylim(-5, 105)

# 이미지 파일로 저장
output_image = 'multimodal_timeline.png'
plt.tight_layout()
plt.savefig(output_image, dpi=300)
print(f"\n🎉 시각화 그래프 저장 완료: {output_image}")