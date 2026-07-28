"""Create a 2x2 V12 pipeline comparison.

Quadrants:
    Original | Final V12 Metric Natural DoF
    Scene-reconstructed metric depth | Moving thin-lens CoC radius
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np


def add_label(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    output = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = max(0.60, frame.shape[1] / 1100.0)
    subtitle_scale = max(0.43, frame.shape[1] / 1550.0)
    title_thickness = 2
    subtitle_thickness = 1

    (title_width, title_height), _ = cv2.getTextSize(
        title,
        font,
        title_scale,
        title_thickness,
    )
    (subtitle_width, subtitle_height), subtitle_baseline = cv2.getTextSize(
        subtitle,
        font,
        subtitle_scale,
        subtitle_thickness,
    )
    box_width = max(title_width, subtitle_width) + 28
    box_height = title_height + subtitle_height + subtitle_baseline + 32
    cv2.rectangle(output, (16, 16), (16 + box_width, 16 + box_height), (0, 0, 0), -1)
    cv2.putText(
        output,
        title,
        (30, 29 + title_height),
        font,
        title_scale,
        (255, 255, 255),
        title_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        subtitle,
        (30, 36 + title_height + subtitle_height),
        font,
        subtitle_scale,
        (210, 210, 210),
        subtitle_thickness,
        cv2.LINE_AA,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Create V12 four-stage pipeline comparison")
    parser.add_argument("--source", required=True)
    parser.add_argument("--final", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--coc", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--final-title", default="2. V12 FINAL")
    parser.add_argument(
        "--final-subtitle",
        default="Scene-anchored moving focal plane",
    )
    parser.add_argument("--depth-title", default="3. SCENE DEPTH")
    parser.add_argument(
        "--depth-subtitle",
        default="Target-affine inverse reprojection",
    )
    parser.add_argument("--coc-title", default="4. EFFECTIVE CoC")
    parser.add_argument(
        "--coc-subtitle",
        default="Moving focal plane: blue is sharp",
    )
    args = parser.parse_args()

    captures = [
        cv2.VideoCapture(args.source),
        cv2.VideoCapture(args.final),
        cv2.VideoCapture(args.depth),
        cv2.VideoCapture(args.coc),
    ]
    if not all(capture.isOpened() for capture in captures):
        for capture in captures:
            capture.release()
        raise RuntimeError("Could not open all V12 pipeline inputs")

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
        ("1. ORIGINAL", "Input video"),
        (args.final_title, args.final_subtitle),
        (args.depth_title, args.depth_subtitle),
        (args.coc_title, args.coc_subtitle),
    ]
    frames_written = 0
    while True:
        labeled_frames = []
        for capture, (title, subtitle) in zip(captures, labels):
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            labeled_frames.append(add_label(frame, title, subtitle))

        if len(labeled_frames) != 4:
            break
        writer.write(
            np.vstack(
                (
                    np.hstack((labeled_frames[0], labeled_frames[1])),
                    np.hstack((labeled_frames[2], labeled_frames[3])),
                )
            )
        )
        frames_written += 1

    for capture in captures:
        capture.release()
    writer.release()
    print(f"Wrote {frames_written} V12 pipeline comparison frames to {args.output}")


if __name__ == "__main__":
    main()
