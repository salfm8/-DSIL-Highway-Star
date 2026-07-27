"""Create a synchronized V9 / V10 progressive-focus comparison video."""

from __future__ import annotations

import argparse

import cv2
import numpy as np


def add_label(frame: np.ndarray, text: str) -> np.ndarray:
    labeled = frame.copy()
    cv2.rectangle(labeled, (14, 14), (365, 58), (0, 0, 0), -1)
    cv2.putText(labeled, text, (27, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v9", required=True)
    parser.add_argument("--v10", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    v9, v10 = cv2.VideoCapture(args.v9), cv2.VideoCapture(args.v10)
    if not v9.isOpened() or not v10.isOpened():
        raise RuntimeError("Could not open both comparison inputs")
    fps = v9.get(cv2.CAP_PROP_FPS)
    width, height = int(v9.get(cv2.CAP_PROP_FRAME_WIDTH)), int(v9.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height))
    frames = 0
    while True:
        ok_v9, frame_v9 = v9.read()
        ok_v10, frame_v10 = v10.read()
        if not ok_v9 or not ok_v10:
            break
        if frame_v10.shape[:2] != (height, width):
            frame_v10 = cv2.resize(frame_v10, (width, height), interpolation=cv2.INTER_LINEAR)
        output.write(np.hstack((add_label(frame_v9, "V9 CONTINUOUS FOCUS"), add_label(frame_v10, "V10 PROGRESSIVE FOCUS"))))
        frames += 1
    v9.release(); v10.release(); output.release()
    print(f"Wrote {frames} frames to {args.output}")


if __name__ == "__main__":
    main()
