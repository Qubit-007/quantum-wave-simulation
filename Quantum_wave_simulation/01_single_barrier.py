import time
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

# Track execution time
start_time = time.time()

# Leverage GPU processing if available
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

# Precompute kinetic energy operator in Fourier space
k_values = 2 * np.pi * torch.fft.fftfreq(len(x), d=dx, device=device)
kinetic_op = torch.exp(-1j * k_values**2 * dt / 2)

# --- 4. POTENTIAL BARRIER SETUP ---
V_1 = torch.zeros_like(x, device=device)
V_1[(x > 6) & (x < 7)] = 3.0

# Prepare arrays for visualization
x_plot = x.cpu().numpy()
V_plot_1 = V_1.cpu().numpy()

# --- 5. VISUALIZATION WINDOW SETUP ---
fig, ax = plt.subplots(figsize=(10, 6))
(line_psi,) = ax.plot(
    x_plot, np.zeros_like(x_plot), label=r"$\psi$ (Wave Function Profile)"
)
ax.fill_between(
    x_plot,
    0,
    V_plot_1 / 30.0,
    color="red",
    alpha=0.3,
    label="Potential Wall (V)",
)
ax.set_ylim(-1.5, 1.5)
ax.set_xlim(-12, 17)
ax.set_title("Quantum Wave Packet Split-Step Fourier Simulation")
ax.legend(loc="upper right")


# --- 6. ANIMATION ENGINE ---
def animate(t_step):
    global psi
    for _ in range(steps_per_frame):
        # Step A: Half-step potential in real space
        psi = psi * torch.exp(-1j * V_1 * dt / 2)

        # Step B: Full-step kinetic energy in Fourier (Frequency) space
        psi_fft = torch.fft.fft(psi)
        psi_fft = psi_fft * kinetic_op
        psi = torch.fft.ifft(psi_fft)

        # Step C: Half-step potential back in real space
        psi = psi * torch.exp(-1j * V_1 * dt / 2)

    # Update frame geometry (extracting real component for dynamic plotting)
    line_psi.set_ydata(psi.cpu().numpy().real)
    return (line_psi,)


# Run execution loop
ani = animation.FuncAnimation(fig, animate, frames=30, interval=20, blit=False)
plt.show()
