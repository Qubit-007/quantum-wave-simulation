"""Two-branch model for straight wave structure and evolving curves.

The straight branch is pointwise and never uses an FFT. The curve branch uses
deep Fourier blocks over space, with time and wave-packet settings as inputs.
"""

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


if __name__ == "__main__":
    # Initialize the model on the available device and report its capacity.
    model = MovingWaveFNO().to(DEVICE)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {DEVICE}")
    print(f"Trainable parameters: {parameter_count:,}")

    # Example profile at one time, using the same packet settings as 04_initial_psi.py.
    x_grid = torch.linspace(-10, 15, 1000, device=DEVICE)
    with torch.no_grad():
        wave = model(x_grid, time=0.0, center=5.0, momentum=15.0)
    print(f"Example wave shape: {tuple(wave.shape)}")
