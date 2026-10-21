"""Generate reusable free-wave training presets for PsiAsInput.py."""

import argparse
from pathlib import Path

import torch

from PsiAsInput import (
    DATASET_FRAMES,
    DT,
    GRID_POINTS,
    STEPS_PER_FRAME,
    X_MAX,
    X_MIN,
    make_trajectory,
)


DEFAULT_TIME_MULTIPLIER = 7
DEFAULT_PRESETS = 128
DEFAULT_OUTPUT = Path(__file__).with_name("psi_presets.pt")


def generate_presets(preset_count: int, time_multiplier: int) -> dict[str, object]:
    """Build slightly varied normalized packet trajectories on the CPU."""
    device = torch.device("cpu")
    x_grid = torch.linspace(X_MIN, X_MAX, GRID_POINTS, device=device)
    frame_count = DATASET_FRAMES * time_multiplier
    centers = torch.linspace(-2.15, -1.85, preset_count, device=device)
    momenta = torch.linspace(14.75, 15.25, preset_count, device=device)

    trajectories = torch.stack(
        [
            make_trajectory(x_grid, center.item(), momentum.item(), frame_count)
            for center, momentum in zip(centers, momenta)
        ]
    )
    conditions = torch.stack((centers, momenta), dim=1)
    return {
        "version": 1,
        "trajectories": trajectories,
        "conditions": conditions,
        "dt": DT,
        "steps_per_frame": STEPS_PER_FRAME,
        "frame_count": frame_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute wave packet trajectories for PsiAsInput.py."
    )
    parser.add_argument(
        "--time",
        type=int,
        default=DEFAULT_TIME_MULTIPLIER,
        help="duration multiplier (default: 7, about 1.05 seconds)",
    )
    parser.add_argument(
        "--presets",
        type=int,
        default=DEFAULT_PRESETS,
        help=f"number of nearby packet variants to generate (default: {DEFAULT_PRESETS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output tensor file (default: {DEFAULT_OUTPUT.name})",
    )
    args = parser.parse_args()
    if args.time < 1:
        parser.error("--time must be at least 1")
    if args.presets < 1:
        parser.error("--presets must be at least 1")

    data = generate_presets(args.presets, args.time)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output)
    duration = (data["frame_count"] - 1) * DT * STEPS_PER_FRAME # type: ignore
    print(
        f"Saved {args.presets:,} presets with {data['frame_count']:,} frames "
        f"({duration:.3f} seconds each) to {args.output}"
    )


if __name__ == "__main__":
    main()
