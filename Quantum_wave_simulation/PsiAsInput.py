"""Train an FNO to predict the next wavefunction from the current psi."""

import argparse
import time
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Match the network size and free-wave simulation used by 01_FNO_TRY.py.
MODES = 128
WIDTH = 128
CURVE_DEPTH = 8
STRAIGHT_DEPTH = 5
GRID_POINTS = 1000
X_MIN = -15.0
X_MAX = 15.0
DT = 0.0001
STEPS_PER_FRAME = 50
DATASET_FRAMES = 30
MODEL_PATH = Path(__file__).resolve().parents[2] / "psi_as_input_fno.pth"
DATA_PATH = Path(__file__).with_name("psi_presets.pt")


class StraightLineBlock(nn.Module):
    """Pointwise layers for straight structure. This block has no FFT."""

    def __init__(self, input_channels: int, width: int, depth: int):
        super().__init__()
        layers = [nn.Conv1d(input_channels, width, kernel_size=1), nn.GELU()]
        for _ in range(depth - 1):
            layers.extend((nn.Conv1d(width, width, kernel_size=1), nn.GELU()))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CurveFNOBlock(nn.Module):
    """One Fourier layer for spatial curves and wave progression."""

    def __init__(self, modes: int, width: int):
        super().__init__()
        self.modes = modes
        self.fourier_weights = nn.Parameter(
            torch.randn(width, width, modes, dtype=torch.cfloat) / width
        )
        self.local = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.fft(x, dim=-1)
        out_ft = torch.zeros_like(x_ft)
        modes = min(self.modes, x.shape[-1] // 2 + 1)
        out_ft[:, :, :modes] = torch.einsum(
            "bci,coi->boi", x_ft[:, :, :modes], self.fourier_weights[:, :, :modes]
        )
        return F.gelu(torch.fft.ifft(out_ft, dim=-1).real + self.local(x))


class CurveBranch(nn.Module):
    """Deep FFT branch for wave shape and its changes."""

    def __init__(self, input_channels: int):
        super().__init__()
        self.lift = nn.Conv1d(input_channels, WIDTH, kernel_size=1)
        self.blocks = nn.ModuleList(
            CurveFNOBlock(MODES, WIDTH) for _ in range(CURVE_DEPTH)
        )
        self.project = nn.Sequential(
            nn.Conv1d(WIDTH, WIDTH, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(WIDTH, 2, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.lift(features)
        for block in self.blocks:
            x = block(x)
        return self.project(x)


class PsiStepFNO(nn.Module):
    """Map current psi to the next psi using the selected input mode."""

    def __init__(self, input_mode: str):
        super().__init__()
        if input_mode not in {"psi", "psi-plus-conditions"}:
            raise ValueError(f"Unknown input mode: {input_mode}")
        self.input_mode = input_mode
        # Current psi always contributes real and imaginary channels.
        input_channels = 2 if input_mode == "psi" else 6
        self.straight_branch = StraightLineBlock(
            input_channels, WIDTH, STRAIGHT_DEPTH
        )
        self.straight_output = nn.Conv1d(WIDTH, 2, kernel_size=1)
        self.curve_branch = CurveBranch(input_channels)

    def forward(
        self,
        x_grid: torch.Tensor,
        psi: torch.Tensor,
        time_value: float | torch.Tensor = 0.0,
        center: float | torch.Tensor = -2.0,
        momentum: float | torch.Tensor = 15.0,
    ) -> torch.Tensor:
        # A 1D grid and wave become one-item batches.
        if x_grid.ndim == 1:
            x_grid = x_grid.unsqueeze(0)
        if psi.ndim == 1:
            psi = psi.unsqueeze(0)
        psi_features = torch.stack((psi.real, psi.imag), dim=1)

        if self.input_mode == "psi":
            features = psi_features
        else:
            batch, points = x_grid.shape

            def expand_condition(value: float | torch.Tensor) -> torch.Tensor:
                value_tensor = torch.as_tensor(
                    value, dtype=x_grid.dtype, device=x_grid.device
                )
                if value_tensor.numel() == 1:
                    return value_tensor.reshape(1, 1).expand(batch, points)
                return value_tensor.reshape(batch, 1).expand(batch, points)

            conditions = torch.stack(
                (
                    x_grid,
                    expand_condition(time_value),
                    expand_condition(center),
                    expand_condition(momentum),
                ),
                dim=1,
            )
            features = torch.cat((psi_features, conditions), dim=1)

        # The straight layers are local and never use an FFT.
        straight = self.straight_output(self.straight_branch(features))
        # The curve layers use FFTs to model spatial wave shape.
        curve = self.curve_branch(features)
        output = straight + curve
        return torch.complex(output[:, 0, :], output[:, 1, :])


def make_packet(x_grid: torch.Tensor, center: float, momentum: float) -> torch.Tensor:
    """Make the same normalized Gaussian packet as 01_single_barrier.py."""
    psi = torch.exp(-((x_grid - center) ** 2)).to(torch.cfloat)
    psi = psi * torch.exp(1j * momentum * x_grid)
    dx = x_grid[1] - x_grid[0]
    return psi / torch.sqrt(torch.sum(torch.abs(psi) ** 2) * dx)


def make_trajectory(
    x_grid: torch.Tensor, center: float, momentum: float, dataset_frames: int
) -> torch.Tensor:
    """Generate split-step snapshots with no potential barrier."""
    dx = x_grid[1] - x_grid[0]
    k_values = 2 * np.pi * torch.fft.fftfreq(
        len(x_grid), d=dx.item(), device=x_grid.device
    )
    kinetic_op = torch.exp(-1j * k_values**2 * DT / 2)
    psi = make_packet(x_grid, center, momentum)
    snapshots = [psi.clone()]
    for _ in range(dataset_frames - 1):
        for _ in range(STEPS_PER_FRAME):
            psi = torch.fft.ifft(torch.fft.fft(psi) * kinetic_op)
        snapshots.append(psi.clone())
    return torch.stack(snapshots)


def train_model(
    model: PsiStepFNO,
    loops: int,
    x_grid: torch.Tensor,
    trajectories: torch.Tensor,
    packet_conditions: torch.Tensor,
    batch_size: int,
) -> None:
    """Train batched next-step predictions from precomputed trajectories."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    model.train()
    started = time.time()
    preset_count, dataset_frames, _ = trajectories.shape

    for epoch in range(loops):
        preset_indices = torch.randint(preset_count, (batch_size,), device=DEVICE)
        frame_indices = torch.randint(dataset_frames - 1, (batch_size,), device=DEVICE)
        current_psi = trajectories[preset_indices, frame_indices]
        target_psi = trajectories[preset_indices, frame_indices + 1]
        centers, momenta = packet_conditions[preset_indices].unbind(dim=1)
        time_values = frame_indices * DT * STEPS_PER_FRAME
        x_batch = x_grid.unsqueeze(0).expand(batch_size, -1)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            x_batch,
            current_psi,
            time_value=time_values,
            center=centers,
            momentum=momenta,
        )
        loss = torch.mean(torch.abs(prediction - target_psi) ** 2)
        loss.backward()
        optimizer.step()
        print(
            f"Epoch {epoch + 1:,}/{loops:,} | Loss: {loss.item():.6e}",
            flush=True,
        )

    print(f"Training finished in {(time.time() - started) / 60:.1f} minutes.")


def show_animation(
    model: PsiStepFNO,
    x_grid: torch.Tensor,
    target_states: torch.Tensor,
    center: float,
    momentum: float,
) -> None:
    """Feed each prediction back in and compare it with split-step evolution."""
    dataset_frames = target_states.shape[0]
    x_plot = x_grid.detach().cpu().numpy()
    target_plot = target_states[0].real.detach().cpu().numpy()
    predicted_states = [target_states[0].clone()]
    current_psi = target_states[0].unsqueeze(0)
    model.eval()
    with torch.no_grad():
        for frame in range(dataset_frames - 1):
            current_psi = model(
                x_grid,
                current_psi,
                time_value=frame * DT * STEPS_PER_FRAME,
                center=center,
                momentum=momentum,
            )
            predicted_states.append(current_psi[0].clone())
    prediction_plot = torch.stack(predicted_states).real.cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    (target_line,) = ax.plot(x_plot, target_plot, label="Split-step target", linewidth=2)
    (prediction_line,) = ax.plot(
        x_plot, prediction_plot[0], label="Autoregressive FNO", color="tab:red", alpha=0.85
    )
    ax.set_xlim(-12, 17)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("Spatial Coordinate (x)")
    ax.set_ylabel("Real part of wave function")
    ax.set_title(f"Psi input ({model.input_mode}): one-step predictions fed back")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)

    def animate(frame: int):
        target_line.set_ydata(target_states[frame].real.detach().cpu().numpy())
        prediction_line.set_ydata(prediction_plot[frame])
        ax.set_title(
            f"Psi input ({model.input_mode}): autoregressive prediction\n"
            f"Time: {frame * DT * STEPS_PER_FRAME:.4f}"
        )
        return target_line, prediction_line

    ani = animation.FuncAnimation(
        fig, animate, frames=dataset_frames, interval=20, blit=False
    )
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an FNO to advance the wavefunction one step at a time."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--psi", action="store_const", const="psi", dest="input_mode",
        help="use only the current wavefunction as model input",
    )
    mode.add_argument(
        "--psi-plus-conditions", action="store_const", const="psi-plus-conditions",
        dest="input_mode", help="use psi plus position, time, center, and momentum",
    )
    parser.add_argument(
        "--loops", type=int, default=5000, help="training epochs (default: 5000)"
    )
    parser.add_argument(
        "--time",
        type=int,
        default=7,
        help="maximum preset duration multiplier (default: 7, about 1 second)",
    )
    parser.add_argument(
        "--data",
        default=DATA_PATH,
        help=f"precomputed preset file (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="training samples processed in parallel per optimizer step (default: 16)",
    )
    parser.add_argument(
        "--test-center",
        type=float,
        default=-2.0,
        help="center of the Gaussian used only for the evaluation test (default: -2.0)",
    )
    parser.add_argument(
        "--test-momentum",
        type=float,
        default=15.0,
        help="momentum of the Gaussian used only for the evaluation test (default: 15.0)",
    )
    parser.add_argument(
        "--fine-tune",
        nargs="?",
        const=MODEL_PATH,
        default=None,
        type=Path,
        metavar="WEIGHTS",
        help="load existing weights before training (default path is the saved model)",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="test loaded weights without training; requires --fine-tune",
    )
    args = parser.parse_args()
    if args.loops < 1 and not args.test_only:
        parser.error("--loops must be at least 1 unless --test-only is used")
    if args.time < 1:
        parser.error("--time must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.test_only and args.fine_tune is None:
        parser.error("--test-only requires --fine-tune to load trained weights")

    print(f"Device: {DEVICE}")
    x_grid = torch.linspace(X_MIN, X_MAX, GRID_POINTS, device=DEVICE)
    model = PsiStepFNO(args.input_mode).to(DEVICE)
    if args.fine_tune is not None:
        try:
            model.load_state_dict(
                torch.load(args.fine_tune, map_location=DEVICE, weights_only=True)
            )
        except FileNotFoundError:
            parser.error(f"weights file '{args.fine_tune}' was not found")
        print(f"Loaded weights: {args.fine_tune}")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Input mode: {args.input_mode}")
    print(f"Trainable parameters: {parameter_count:,}")

    if args.test_only:
        test_frames = DATASET_FRAMES * args.time
        print(f"Testing saved weights for {(test_frames - 1) * DT * STEPS_PER_FRAME:.3f} seconds")
        target_states = make_trajectory(
            x_grid, args.test_center, args.test_momentum, test_frames
        )
        show_animation(
            model, x_grid, target_states, args.test_center, args.test_momentum
        )
        return

    try:
        saved_data = torch.load(args.data, map_location=DEVICE, weights_only=True)
    except FileNotFoundError:
        parser.error(
            f"data file '{args.data}' was not found; generate it first with Data.py"
        )
    if saved_data.get("version") != 1:
        parser.error(f"unsupported or invalid preset file: {args.data}")
    trajectories = saved_data["trajectories"].to(DEVICE)
    packet_conditions = saved_data["conditions"].to(DEVICE)
    dataset_frames = min(DATASET_FRAMES * args.time, trajectories.shape[1])
    trajectories = trajectories[:, :dataset_frames]
    duration_seconds = (dataset_frames - 1) * DT * STEPS_PER_FRAME

    print(f"Training epochs: {args.loops:,}")
    print(f"Training presets: {trajectories.shape[0]:,}")
    print(f"Training batch size: {args.batch_size}")
    print(f"Preset duration: {duration_seconds:.3f} seconds")

    train_model(
        model,
        args.loops,
        x_grid,
        trajectories,
        packet_conditions,
        args.batch_size,
    )
    test_states = make_trajectory(
        x_grid, args.test_center, args.test_momentum, DATASET_FRAMES * args.time
    )
    show_animation(
        model, x_grid, test_states, args.test_center, args.test_momentum
    )

    answer = input("Save the trained parameters? [y/N]: ").strip().lower()
    if answer in {"y", "yes"}:
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Parameters saved to {MODEL_PATH}")
    else:
        print("Parameters were not saved.")


if __name__ == "__main__":
    main()
