"""Create a labeled V6-baseline versus V8-semantic-focus comparison video."""

from __future__ import annotations

import argparse

import cv2
import numpy as np


def label(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    result = frame.copy()
    cv2.putText(result, text, (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create V6/V8 semantic-focus comparison")
    parser.add_argument("--v6", required=True)
    parser.add_argument("--v8", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = cv2.VideoCapture(args.v6)
    v8 = cv2.VideoCapture(args.v8)
    if not baseline.isOpened() or not v8.isOpened():
        raise RuntimeError("Both input videos must be readable")
    fps = v8.get(cv2.CAP_PROP_FPS)
    width = int(v8.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(v8.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height))
    if not output.isOpened():
        raise RuntimeError(f"Cannot create output video: {args.output}")

    frames = 0
    while True:
        ok_v6, v6_frame = baseline.read()
        ok_v8, v8_frame = v8.read()
        if not ok_v6 or not ok_v8:
            break
        if v6_frame.shape[:2] != (height, width):
            v6_frame = cv2.resize(v6_frame, (width, height), interpolation=cv2.INTER_AREA)
        output.write(np.hstack([label(v6_frame, "V6 BASELINE", (80, 80, 255)), label(v8_frame, "V8 SEMANTIC FOCUS", (80, 255, 120))]))
        frames += 1

    baseline.release()
    v8.release()
    output.release()
    print(f"Created {frames}-frame comparison: {args.output}")


if __name__ == "__main__":
    main()
