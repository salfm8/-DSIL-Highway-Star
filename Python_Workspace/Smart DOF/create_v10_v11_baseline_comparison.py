"""Create a synchronized 2x2 Original / Depth / V10 / V11 comparison."""

from __future__ import annotations

import argparse

import cv2
import numpy as np


def add_label(frame: np.ndarray, text: str) -> np.ndarray:
    labeled = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.62, frame.shape[1] / 1050.0)
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    x, y = 20, 20
    cv2.rectangle(
        labeled,
        (x, y),
        (x + text_width + 24, y + text_height + baseline + 20),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        labeled,
        text,
        (x + 12, y + text_height + 8),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create 2x2 Original / Depth / V10 / V11 comparison"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--v10", required=True)
    parser.add_argument("--v11", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = [args.source, args.depth, args.v10, args.v11]
    captures = [cv2.VideoCapture(path) for path in paths]
    if not all(capture.isOpened() for capture in captures):
        for capture in captures:
            capture.release()
        raise RuntimeError("Could not open every comparison input")

    source = captures[0]
    fps = source.get(cv2.CAP_PROP_FPS)
    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Keep a presentation-friendly 16:9 frame while preserving all source pixels.
    output = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height * 2),
    )
    if not output.isOpened():
        for capture in captures:
            capture.release()
        raise RuntimeError(f"Could not create comparison video: {args.output}")

    labels = [
        "ORIGINAL",
        "TEMPORALLY STABILIZED DEPTH",
        "V10 OBJECT-CENTRED FOCUS",
        "V11 PARALLEL-PLANE NATURAL DoF",
    ]
    frames_written = 0
    while True:
        frames: list[np.ndarray] = []
        for index, capture in enumerate(captures):
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(
                    frame, (width, height), interpolation=cv2.INTER_LINEAR
                )
            frames.append(add_label(frame, labels[index]))
        if len(frames) != 4:
            break

        output.write(
            np.vstack(
                (
                    np.hstack((frames[0], frames[1])),
                    np.hstack((frames[2], frames[3])),
                )
            )
        )
        frames_written += 1

    for capture in captures:
        capture.release()
    output.release()
    print(f"Wrote {frames_written} comparison frames to {args.output}")


if __name__ == "__main__":
    main()
