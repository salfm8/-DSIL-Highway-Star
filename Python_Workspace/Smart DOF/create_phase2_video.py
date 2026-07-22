import cv2
import numpy as np
import sys

orig_path = "../0720 Pitch/sample_38_short.mp4"
coc_path = "step1_coc_bokeh_85mm.mp4"
sam_path = "step2_cinematic_sam3.mp4"
out_path = "phase2_video_comparison.mp4"

cap1 = cv2.VideoCapture(orig_path)
cap2 = cv2.VideoCapture(coc_path)
cap3 = cv2.VideoCapture(sam_path)

fps = cap1.get(cv2.CAP_PROP_FPS)
w = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, fps, (w * 3, h))

print(f"비교 영상 렌더링 시작... (총 {total} 프레임)")

font = cv2.FONT_HERSHEY_SIMPLEX
idx = 0
while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    ret3, f3 = cap3.read()
    
    if not (ret1 and ret2 and ret3):
        break
        
    idx += 1
    
    cv2.putText(f1, 'ORIGINAL', (10, 40), font, 1.0, (255,255,255), 2)
    cv2.putText(f2, 'CoC ONLY (Phase 1)', (10, 40), font, 0.8, (100,255,100), 2)
    cv2.putText(f3, 'SAM + CoC (Phase 2)', (10, 40), font, 0.8, (100,200,255), 2)
    
    combined = np.hstack([f1, f2, f3])
    out.write(combined)
    sys.stdout.write(f"\r  Frame {idx}/{total} ({idx/total*100:.0f}%)")
    sys.stdout.flush()

cap1.release()
cap2.release()
cap3.release()
out.release()
print(f"\n✅ 완료! 출력 파일: {out_path}")
