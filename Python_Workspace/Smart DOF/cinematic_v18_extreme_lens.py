"""Smart DoF V18 — target-kinematic focus with an extreme virtual lens.

V18 preserves V17's target-kinematic scale anchoring exactly.  The only
rendering change is a deliberately shallow virtual-lens preset:

    focal length:       220 mm
    f-number:           f/1.0
    digital CoC gain:   5x
    maximum blur radius: 60 px

As the forced target distance moves from about 38 m to 15 m, the thin-lens
Circle of Confusion grows strongly for geometry behind the target.  SAM is
still used only to measure the target depth; it never decides which pixels
remain sharp.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cinematic_v7_natural import parse_point
from cinematic_v8_semantic_focus import parse_bbox, parse_points
from cinematic_v17_target_kinematic_anchor import CinematicDoFv17


class CinematicDoFv18(CinematicDoFv17):
    """V17 kinematic focus rendered with a very shallow virtual DoF."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pipeline_name = (
            "Smart DoF v18.0 — Extreme-Lens Kinematic Focus Pull"
        )
        print("  V18 extreme virtual-lens rendering enabled")
        print(
            "  Intended effect: background/overpass separation grows "
            "aggressively as S(t) approaches 15m"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Smart DoF V18 — V17 target-kinematic focus with an extreme "
            "220mm f/1.0 virtual lens"
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--depth-output", default=None)
    parser.add_argument("--coc-output", default=None)
    parser.add_argument("--focus-log", default=None)
    parser.add_argument("--point", required=True, type=parse_point)
    parser.add_argument("--bbox", required=True, type=parse_bbox)
    parser.add_argument("--positive-points", default=None)
    parser.add_argument("--negative-points", default=None)
    parser.add_argument("--metric-model", type=Path, default=None)
    parser.add_argument("--focal-length-mm", type=float, default=220.0)
    parser.add_argument("--f-number", type=float, default=1.0)
    parser.add_argument("--sensor-width-mm", type=float, default=36.0)
    parser.add_argument(
        "--coc-gain",
        type=float,
        default=5.0,
        help="Presentation-scale multiplier applied after physical thin-lens CoC.",
    )
    parser.add_argument("--max-blur", type=float, default=60.0)
    parser.add_argument("--transition-sec", type=float, default=1.5)
    parser.add_argument(
        "--kinematic-start-distance-m",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--kinematic-end-distance-m",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--kinematic-end-frame",
        type=int,
        default=None,
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--sam-model", type=Path, default=None)
    args = parser.parse_args()

    pipeline = CinematicDoFv18(
        init_point=args.point,
        focus_bbox=args.bbox,
        positive_points=parse_points(args.positive_points),
        negative_points=parse_points(args.negative_points),
        max_blur=args.max_blur,
        transition_sec=args.transition_sec,
        sam_interval=1,
        model_path=args.sam_model,
        depth_mode="metric",
        metric_model_path=args.metric_model,
        focal_length_mm=args.focal_length_mm,
        f_number=args.f_number,
        sensor_width_mm=args.sensor_width_mm,
        coc_gain=args.coc_gain,
        kinematic_start_distance_m=args.kinematic_start_distance_m,
        kinematic_end_distance_m=args.kinematic_end_distance_m,
        kinematic_end_frame=args.kinematic_end_frame,
    )
    pipeline.process_video(
        args.input,
        args.output,
        args.depth_output,
        args.coc_output,
        args.focus_log,
        args.max_frames,
    )


if __name__ == "__main__":
    main()
