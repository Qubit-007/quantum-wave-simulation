import time
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

# Track script initialization
start_time = time.time()

# Offload calculations to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. PHYSICAL GRID SETUP ---
x = torch.linspace(-15, 15, 1000, device=device)
dx = x[1] - x[0]

# --- 2. WAVE PACKET INITIALIZATION ---
k = 15  # Wave number / initial momentum
envelope = torch.exp(-(x + 2) ** 2)
infinite_wave = torch.exp(1j * k * x)
psi = envelope * infinite_wave

# Normalize wave function to unit probability
psi = psi / torch.sqrt(torch.sum(torch.abs(psi) ** 2) * dx)

# --- 3. SIMULATION CONFIGURATION ---
dt = 0.0001
steps_per_frame = 50

# Precompute kinetic energy operator in Fourier (Frequency) space
k_values = 2 * np.pi * torch.fft.fftfreq(len(x), d=dx, device=device)
kinetic_op = torch.exp(-1j * k_values**2 * dt / 2)


# --- 4. ENVIRONMENT GENERATION (Modular Potential Walls) ---
def square_potential(grid: torch.Tensor, x_min: float, x_max: float, height: float) -> torch.Tensor:
    """Generates an individual step barrier on the physical grid."""
    potential_grid = torch.zeros_like(grid, device=device)
    potential_grid[(grid > x_min) & (grid < x_max)] = height
    return potential_grid


def generate_quantum_cage(grid: torch.Tensor) -> torch.Tensor:
    """Combines multiple potential walls to trap the particle."""
    barriers = [
        square_potential(grid, x_min=5, x_max=5.5, height=3000.0),  # Right Wall
        square_potential(grid, x_min=-8, x_max=-7, height=1000.0),  # Left Wall
    ]
    return sum(barriers)


# Build the physical environment
V = generate_quantum_cage(x)
V_plot = V.cpu().numpy()
x_plot = x.cpu().numpy()

# --- 5. VISUALIZATION WINDOW SETUP ---
fig, ax = plt.subplots(figsize=(10, 6))
(line_probability,) = ax.plot(
    x_plot, np.zeros_like(x_plot), label=r"$\psi$ (Wave Function Profile)"
)
ax.fill_between(
    x_plot,
    0,
    V_plot / 30.0,
    color="red",
    alpha=0.3,
    label="Potential Enclosure",
)
ax.set_ylim(-1, 1)
ax.set_xlim(-14, 12)
ax.set_title("Quantum Prisoner Simulation: Multi-Barrier Enclosure")
ax.legend(loc="upper right")


# --- 6. ANIMATION ENGINE (Split-Step Fourier Loop) ---
def animate(t_step):
    global psi
    for _ in range(steps_per_frame):
        # Step A: Half-step potential in real space
        psi = psi * torch.exp(-1j * V * dt / 2)

        # Step B: Full-step kinetic energy in frequency space
        psi_fft = torch.fft.fft(psi)
        psi_fft = psi_fft * kinetic_op
        psi = torch.fft.ifft(psi_fft)

        # Step C: Half-step potential in real space
        psi = psi * torch.exp(-1j * V * dt / 2)

    # Plotting real component to bypass complex-to-real rendering limitations
    line_probability.set_ydata(psi.cpu().numpy().real)
    return (line_probability,)


# Execute real-time animation rendering
ani = animation.FuncAnimation(fig, animate, frames=30, interval=20, blit=False)
plt.show()
