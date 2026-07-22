"""Create a labeled V6-versus-V7 side-by-side comparison video."""

from __future__ import annotations

import argparse

import cv2
import numpy as np


def add_label(frame: np.ndarray, label: str, color: tuple[int, int, int]) -> np.ndarray:
    labeled = frame.copy()
    cv2.putText(labeled, label, (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(description="Create V6 and V7 Natural comparison video")
    parser.add_argument("--v6", required=True, help="Baseline V6 video")
    parser.add_argument("--v7", required=True, help="V7 Natural video")
    parser.add_argument("--output", required=True, help="Output comparison MP4")
    args = parser.parse_args()

    v6_capture = cv2.VideoCapture(args.v6)
    v7_capture = cv2.VideoCapture(args.v7)
    if not v6_capture.isOpened() or not v7_capture.isOpened():
        raise RuntimeError("Both V6 and V7 videos must be readable")

    fps = v7_capture.get(cv2.CAP_PROP_FPS)
    width = int(v7_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(v7_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create comparison video: {args.output}")

    frames = 0
    while True:
        ok_v6, frame_v6 = v6_capture.read()
        ok_v7, frame_v7 = v7_capture.read()
        if not ok_v6 or not ok_v7:
            break
        if frame_v6.shape[:2] != (height, width):
            frame_v6 = cv2.resize(frame_v6, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(
            np.hstack(
                [
                    add_label(frame_v6, "V6 BASELINE", (80, 80, 255)),
                    add_label(frame_v7, "V7 NATURAL", (80, 255, 120)),
                ]
            )
        )
        frames += 1

    v6_capture.release()
    v7_capture.release()
    writer.release()
    print(f"Created {frames}-frame comparison: {args.output}")


if __name__ == "__main__":
    main()
