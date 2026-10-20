"""Two-branch model for straight wave structure and evolving curves.

The straight branch is pointwise and never uses an FFT. The curve branch uses
deep Fourier blocks over space, with time and wave-packet settings as inputs.
"""


import argparse
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# High-capacity curve branch. Width 128, 8 layers and 128 modes use about
# 33.6 million real-valued spectral weight parameters, plus the other layers.
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
DATA_RESTART_EVERY = 1000
MODEL_PATH = "moving_wave_fno.pth"


class StraightLineBlock(nn.Module):
    """Pointwise hidden layers for straight or slowly varying structure."""

    def __init__(self, input_channels: int, width: int, depth: int):
        super().__init__()
        layers = [nn.Conv1d(input_channels, width, kernel_size=1), nn.GELU()]
        for _ in range(depth - 1):
            layers.extend((nn.Conv1d(width, width, kernel_size=1), nn.GELU()))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Kernel size 1 mixes features at each point and does not use an FFT.
        return self.layers(x)


class CurveFNOBlock(nn.Module):
    """One spatial Fourier layer and one local spatial layer."""

    def __init__(self, modes: int, width: int):
        super().__init__()
        self.modes = modes
        self.width = width
        scale = 1.0 / width
        self.fourier_weights = nn.Parameter(
            scale * torch.randn(width, width, modes, dtype=torch.cfloat)
        )
        self.local = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep only the learned low-frequency modes, as in the FNO example.
        x_ft = torch.fft.fft(x, dim=-1)
        out_ft = torch.zeros_like(x_ft)
        modes = min(self.modes, x.shape[-1] // 2 + 1)
        out_ft[:, :, :modes] = torch.einsum(
            "bci,coi->boi", x_ft[:, :, :modes], self.fourier_weights[:, :, :modes]
        )
        x_curve = torch.fft.ifft(out_ft, dim=-1).real
        return F.gelu(x_curve + self.local(x))


class CurveBranch(nn.Module):
    """Deep FFT branch for wave curves and their changes over time."""

    def __init__(self, input_channels: int, width: int, modes: int, depth: int):
        super().__init__()
        self.lift = nn.Conv1d(input_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            CurveFNOBlock(modes, width) for _ in range(depth)
        )
        self.project = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width, 2, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.lift(features)
        for block in self.blocks:
            x = block(x)
        return self.project(x)


class MovingWaveFNO(nn.Module):
    """Predict a complex wave profile from position, time and packet settings.

    Inputs are a spatial grid and scalar time, center, and momentum values.
    Both branches return real and imaginary components, which are added before
    forming the complex-valued wave function.
    """

    def __init__(
        self,
        modes: int = MODES,
        width: int = WIDTH,
        curve_depth: int = CURVE_DEPTH,
        straight_depth: int = STRAIGHT_DEPTH,
    ):
        super().__init__()
        # Inputs: x, time, packet center, and momentum.
        input_channels = 4
        self.straight_branch = StraightLineBlock(
            input_channels, width, straight_depth
        )
        self.straight_output = nn.Conv1d(width, 2, kernel_size=1)
        self.curve_branch = CurveBranch(
            input_channels, width, modes, curve_depth
        )

    def forward(
        self,
        x_grid: torch.Tensor,
        time: float | torch.Tensor,
        center: float | torch.Tensor = -2.0,
        momentum: float | torch.Tensor = 15.0,
    ) -> torch.Tensor:
        # Make each scalar condition available at every spatial point.
        if x_grid.ndim == 1:
            x_grid = x_grid.unsqueeze(0)
        batch, points = x_grid.shape

        def expand_condition(value: float | torch.Tensor) -> torch.Tensor:
            value_tensor = torch.as_tensor(
                value, dtype=x_grid.dtype, device=x_grid.device
            )
            if value_tensor.numel() == 1:
                value_tensor = value_tensor.reshape(1, 1).expand(batch, points)
            elif value_tensor.ndim == 1:
                value_tensor = value_tensor.reshape(batch, 1).expand(batch, points)
            return value_tensor

        features = torch.stack(
            (
                x_grid,
                expand_condition(time),
                expand_condition(center),
                expand_condition(momentum),
            ),
            dim=1,
        )

        # The straight branch sees local features only; no FFT is used here.
        straight = self.straight_output(self.straight_branch(features))
        # The curve branch learns spatial patterns and their time progression.
        curve = self.curve_branch(features)
        output = straight + curve
        return torch.complex(output[:, 0, :], output[:, 1, :])


def make_packet(x_grid: torch.Tensor, center: float, momentum: float) -> torch.Tensor:
    """Make and normalize the initial packet used by 01_single_barrier.py."""
    psi = torch.exp(-((x_grid - center) ** 2)).to(torch.cfloat)
    psi = psi * torch.exp(1j * momentum * x_grid)
    dx = x_grid[1] - x_grid[0]
    return psi / torch.sqrt(torch.sum(torch.abs(psi) ** 2) * dx)


def advance_free_packet(
    psi: torch.Tensor, kinetic_op: torch.Tensor, steps: int
) -> torch.Tensor:
    """Advance the packet with the barrier-free kinetic FFT step."""
    for _ in range(steps):
        psi = torch.fft.ifft(torch.fft.fft(psi) * kinetic_op)
    return psi


def make_trajectory(
    x_grid: torch.Tensor,
    center: float,
    momentum: float,
    frames: int = DATASET_FRAMES,
) -> torch.Tensor:
    """Generate snapshots at the same spacing as the single-barrier plot."""
    dx = x_grid[1] - x_grid[0]
    k_values = 2 * np.pi * torch.fft.fftfreq(
        len(x_grid), d=dx.item(), device=x_grid.device
    )
    kinetic_op = torch.exp(-1j * k_values**2 * DT / 2)
    psi = make_packet(x_grid, center, momentum)
    snapshots = [psi.clone()]
    for _ in range(frames - 1):
        psi = advance_free_packet(psi, kinetic_op, STEPS_PER_FRAME)
        snapshots.append(psi.clone())
    return torch.stack(snapshots)


def make_training_data(
    x_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Start fresh free-packet trajectories for this 1,000-epoch cycle."""
    centers = torch.empty(2).uniform_(-4.0, 0.0).tolist()
    momenta = torch.empty(2).uniform_(12.0, 18.0).tolist()
    trajectories = [
        make_trajectory(x_grid, center, momentum)
        for center, momentum in zip(centers, momenta)
    ]
    return (
        torch.tensor(centers, device=DEVICE),
        torch.tensor(momenta, device=DEVICE),
        torch.stack(trajectories),
    )


def train_model(model: MovingWaveFNO, loops: int, x_grid: torch.Tensor) -> None:
    """Train on newly generated trajectories every 1,000 epochs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    started = time.time()
    model.train()

    for epoch in range(loops):
        if epoch % DATA_RESTART_EVERY == 0:
            centers, momenta, trajectories = make_training_data(x_grid)
            print(f"\nGenerated fresh training data at epoch {epoch:,}.", flush=True)

        packet_index = torch.randint(centers.numel(), (1,), device=DEVICE).item()
        frame_index = torch.randint(DATASET_FRAMES, (1,), device=DEVICE).item()
        target = trajectories[packet_index, frame_index]
        time_value = frame_index * DT * STEPS_PER_FRAME

        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            x_grid,
            time=time_value,
            center=centers[packet_index],
            momentum=momenta[packet_index],
        )[0]
        loss = torch.mean(torch.abs(prediction - target) ** 2)
        loss.backward()
        optimizer.step()

        print(
            f"Epoch {epoch + 1:,}/{loops:,} | Loss: {loss.item():.6e}",
            flush=True,
        )

    print(f"Training finished in {(time.time() - started) / 60:.1f} minutes.")


def show_animation(model: MovingWaveFNO, x_grid: torch.Tensor) -> None:
    """Compare the trained wave with the free-packet simulation over time."""
    center = -2.0
    momentum = 15.0
    target_states = make_trajectory(x_grid, center, momentum, frames=30)
    time_values = np.arange(len(target_states)) * DT * STEPS_PER_FRAME
    x_plot = x_grid.detach().cpu().numpy()

    model.eval()
    with torch.no_grad():
        predicted = [
            model(x_grid, time_value, center=center, momentum=momentum)[0]
            for time_value in time_values
        ]
    prediction_plot = torch.stack(predicted).real.cpu().numpy()
    target_plot = target_states[0].real.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    (target_line,) = ax.plot(x_plot, target_plot, label="Split-step target", linewidth=2)
    (prediction_line,) = ax.plot(
        x_plot, prediction_plot[0], label="FNO prediction", color="tab:red", alpha=0.85
    )
    ax.set_xlim(-12, 17)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("Spatial Coordinate (x)")
    ax.set_ylabel("Real part of wave function")
    ax.set_title("Free Quantum Wave: FNO Prediction vs Split-Step Simulation")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)

    def animate(frame: int):
        target_line.set_ydata(target_states[frame].real.detach().cpu().numpy())
        prediction_line.set_ydata(prediction_plot[frame])
        ax.set_title(
            "Free Quantum Wave: FNO Prediction vs Split-Step Simulation\n"
            f"Time: {time_values[frame]:.4f}"
        )
        return target_line, prediction_line

    ani = animation.FuncAnimation(
        fig, animate, frames=len(target_states), interval=20, blit=False
    )
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the two-branch FNO on a moving free wave packet."
    )
    parser.add_argument(
        "--loops", type=int, default=5000, help="training epochs (default: 5000)"
    )
    args = parser.parse_args()
    if args.loops < 1:
        parser.error("--loops must be at least 1")

    print(f"Device: {DEVICE}")
    x_grid = torch.linspace(X_MIN, X_MAX, GRID_POINTS, device=DEVICE)
    model = MovingWaveFNO().to(DEVICE)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable parameters: {parameter_count:,}")
    print(f"Training epochs: {args.loops:,}")
    print(f"Training data refresh interval: {DATA_RESTART_EVERY:,} epochs")

    train_model(model, args.loops, x_grid)
    show_animation(model, x_grid)

    answer = input("Save the trained parameters? [y/N]: ").strip().lower()
    if answer in {"y", "yes"}:
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Parameters saved to {MODEL_PATH}")
    else:
        print("Parameters were not saved.")


if __name__ == "__main__":
    main()
