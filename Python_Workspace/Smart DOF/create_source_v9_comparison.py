"""Create a frame-synchronised original-versus-V9 comparison video."""

from __future__ import annotations

import argparse

import cv2
import numpy as np


def label(frame: np.ndarray, text: str) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (14, 14), (300, 58), (0, 0, 0), -1)
    cv2.putText(
        result, text, (26, 45), cv2.FONT_HERSHEY_SIMPLEX,
        0.82, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create original / V9 side-by-side comparison")
    parser.add_argument("--source", required=True)
    parser.add_argument("--v9", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = cv2.VideoCapture(args.source)
    v9 = cv2.VideoCapture(args.v9)
    if not source.isOpened() or not v9.isOpened():
        raise RuntimeError("Could not open source or V9 video")

    fps = source.get(cv2.CAP_PROP_FPS)
    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height)
    )
    if not output.isOpened():
        raise RuntimeError(f"Could not create comparison video: {args.output}")

    frames = 0
    while True:
        source_ok, source_frame = source.read()
        v9_ok, v9_frame = v9.read()
        if not source_ok or not v9_ok:
            break
        if v9_frame.shape[:2] != (height, width):
            v9_frame = cv2.resize(v9_frame, (width, height), interpolation=cv2.INTER_LINEAR)
        combined = np.hstack((label(source_frame, "ORIGINAL"), label(v9_frame, "V9 CINEMATIC FOCUS")))
        output.write(combined)
        frames += 1

    source.release()
    v9.release()
    output.release()
    print(f"Wrote {frames} comparison frames to {args.output}")


if __name__ == "__main__":
    main()
