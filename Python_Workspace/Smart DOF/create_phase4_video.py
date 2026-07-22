import cv2
import numpy as np
import sys

orig_path = "../0720 Pitch/sample_38_short.mp4"
step4_path = "step10_v6_sam.mp4"
out_path = "phase10_video_comparison.mp4"

cap1 = cv2.VideoCapture(orig_path)
cap2 = cv2.VideoCapture(step4_path)

fps = cap1.get(cv2.CAP_PROP_FPS)
w = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, fps, (w * 2, h))

font = cv2.FONT_HERSHEY_SIMPLEX
idx = 0
while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    
    if not (ret1 and ret2):
        break
        
    idx += 1
    
    cv2.putText(f1, 'ORIGINAL', (10, 40), font, 1.0, (255,255,255), 2)
    cv2.putText(f2, 'Phase 4 High-End Cinematic', (10, 40), font, 1.0, (100,255,100), 2)
    
    combined = np.hstack([f1, f2])
    out.write(combined)
    sys.stdout.write(f"\r  Frame {idx}/{total}")
    sys.stdout.flush()

cap1.release()
cap2.release()
out.release()
print("\nDone")
