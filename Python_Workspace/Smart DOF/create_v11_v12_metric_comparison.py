"""Create Original / Linear Metric Depth / V11 / V12 2x2 comparison."""

from __future__ import annotations

import argparse

import cv2
import numpy as np


def add_label(frame: np.ndarray, text: str) -> np.ndarray:
    output = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.62, frame.shape[1] / 1050.0)
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    cv2.rectangle(
        output,
        (18, 18),
        (42 + text_width, 38 + text_height + baseline),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        output,
        text,
        (30, 30 + text_height),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--v11", required=True)
    parser.add_argument("--v12", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    captures = [
        cv2.VideoCapture(args.source),
        cv2.VideoCapture(args.depth),
        cv2.VideoCapture(args.v11),
        cv2.VideoCapture(args.v12),
    ]
    if not all(capture.isOpened() for capture in captures):
        for capture in captures:
            capture.release()
        raise RuntimeError("Could not open all comparison inputs")

    fps = captures[0].get(cv2.CAP_PROP_FPS)
    width = int(captures[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(captures[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height * 2),
    )
    if not writer.isOpened():
        for capture in captures:
            capture.release()
        raise RuntimeError(f"Could not create comparison video: {args.output}")

    labels = [
        "ORIGINAL",
        "V12 LINEAR METRIC DEPTH (2-80m)",
        "V11 RELATIVE INVERSE-DEPTH DoF",
        "V12 METRIC THIN-LENS DoF",
    ]
    count = 0
    while True:
        frames = []
        for capture, label in zip(captures, labels):
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            frames.append(add_label(frame, label))
        if len(frames) != 4:
            break
        writer.write(
            np.vstack(
                (
                    np.hstack((frames[0], frames[1])),
                    np.hstack((frames[2], frames[3])),
                )
            )
        )
        count += 1

    for capture in captures:
        capture.release()
    writer.release()
    print(f"Wrote {count} comparison frames to {args.output}")


if __name__ == "__main__":
    main()
