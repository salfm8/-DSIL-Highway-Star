"""Create a playable V9 delivery render from the verified V8 semantic render.

This delivery fallback preserves V8's selected-building focus and replaces the
abrupt post-target handoff with a cosine-eased pull back to the source image.
"""
from __future__ import annotations

import argparse
import math
import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--v8", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pull-start", type=int, default=120)
    parser.add_argument("--pull-frames", type=int, default=44)
    args = parser.parse_args()

    source = cv2.VideoCapture(args.source)
    v8 = cv2.VideoCapture(args.v8)
    fps = source.get(cv2.CAP_PROP_FPS)
    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    index = 0
    while True:
        ok_source, source_frame = source.read()
        ok_v8, v8_frame = v8.read()
        if not ok_source or not ok_v8:
            break
        if index < args.pull_start:
            frame = v8_frame
        else:
            progress = min(1.0, (index - args.pull_start + 1) / max(args.pull_frames, 1))
            v8_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
            frame = cv2.addWeighted(v8_frame, v8_weight, source_frame, 1.0 - v8_weight, 0.0)
        output.write(frame)
        index += 1
    source.release(); v8.release(); output.release()
    print(f"Wrote {index} frames to {args.output}")


if __name__ == "__main__":
    main()
