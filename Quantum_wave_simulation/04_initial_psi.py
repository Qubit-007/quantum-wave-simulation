"""
04_initial_psi.py

Learns the initial wave function (psi) profile across variable space shifts 
using a Deep 1D Fourier Neural Operator (FNO).
"""

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# --- 0. Configurations & Reproducibility ---
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODES = 64
WIDTH = 32
LEARNING_RATE = 0.002
STEPS = 3001
GRID_SIZE = 1000

# --- 1. Define Reusable FNO Block ---
class FNO1dBlock(nn.Module):
    """
    1D Fourier Neural Operator Block performing frequency-domain filtering
    paired with a spatial domain skip connection.
    """
    def __init__(self, modes: int, width: int):
        super().__init__()
        self.modes = modes
        self.width = width
        
        # Parameterized complex weights for spectral convolution
        self.fourier_weights = nn.Parameter(
            torch.view_as_complex(torch.randn(width, width, modes, 2) * (1.0 / width))
        )
        # Spatial shortcut path
        self.w = nn.Conv1d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, width, x_dim)
        x_complex = x.to(torch.cfloat)
        x_ft = torch.fft.fft(x_complex, dim=-1)
        
        # Filter higher frequencies out by truncating to specified modes
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :self.modes] = torch.einsum(
            "bci,coi->boi", 
            x_ft[:, :, :self.modes], 
            self.fourier_weights
        )
        
        # Invert back to physical space and combine paths
        x_fourier = torch.fft.ifft(out_ft, dim=-1)
        x_spatial = self.w(x)
        
        return F.gelu(torch.real(x_fourier) + x_spatial)

# --- 2. Build the Deep FNO ---
class DeepWaveFNO(nn.Module):
    """
    Deep FNO Network mapping spatial grids and parameter conditions 
    to a complex-valued wave function output.
    """
    def __init__(self, modes: int, width: int = 32):
        super().__init__()
        # Lifting layer: maps 2 input features [x coordinate, spatial shift i] to hidden width
        self.p = nn.Conv1d(2, width, 1) 
        
        # High-capacity learning block stack
        self.block1 = FNO1dBlock(modes, width)
        self.block2 = FNO1dBlock(modes, width)
        self.block3 = FNO1dBlock(modes, width)
        
        # Projection layer: maps width features to 2 output channels [Real, Imaginary]
        self.q = nn.Conv1d(width, 2, 1) 

    def forward(self, x_grid: torch.Tensor, i: float) -> torch.Tensor:
        batch, x_dim = x_grid.shape
        i_tensor = torch.tensor([i], dtype=x_grid.dtype, device=x_grid.device)
        i_tensor = i_tensor.view(1, 1).expand(batch, x_dim)
        
        # Construct input tensor setup
        x = torch.stack((x_grid, i_tensor), dim=1) 
        x = self.p(x)
        
        # Feature processing inside deep FNO blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        out = self.q(x)
        return torch.complex(out[:, 0, :], out[:, 1, :])

# --- 3. Execution Pipeline ---
if __name__ == "__main__":
    print(f"Using target hardware: {DEVICE}")
    
    # Model Initialization
    model = DeepWaveFNO(modes=MODES, width=WIDTH).to(DEVICE)  
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    x_grid = torch.linspace(-10, 15, GRID_SIZE, device=DEVICE)

    # --- Training Loop ---
    print("\n--- Starting Deep FNO Training ---")
    model.train()
    for step in range(STEPS):
        optimizer.zero_grad()
        
        # Dynamic variable mapping via random target shift parameterization
        shift_i = torch.empty(1).uniform_(-8.0, 8.0).item()
        
        # Compute ground truth complex wave packet profiles
        target_wave = torch.exp(-(x_grid - shift_i) ** 2) * torch.exp(1j * 15 * x_grid)  
        wave_pred = model(x_grid.unsqueeze(0), shift_i).squeeze() 
        
        # L2 complex-valued loss computation
        loss = torch.mean(torch.abs(wave_pred - target_wave) ** 2)
        loss.backward()
        optimizer.step()
        
        if step % 300 == 0:
            print(f"Step {step:4d} | Current L2 Loss: {loss.item():.6f}")

    # --- Verification & Inference Evaluation ---
    print("\n--- Evaluating Model Predictions ---")
    model.eval()
    
    test_i = 5.0
    test_k = 15.0
    
    with torch.no_grad():
        true_wave = torch.exp(-(x_grid - test_i) ** 2) * torch.exp(1j * test_k * x_grid)
        pred_wave = model(x_grid.unsqueeze(0), test_i).squeeze()

    # Convert tensors to CPU NumPy arrays for visualization
    x_cpu = x_grid.cpu().numpy()
    true_real = np.real(true_wave.cpu().numpy())
    pred_real = np.real(pred_wave.cpu().numpy())

    # Generate Performance Visualization Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_cpu, true_real, label="Ground Truth (Real)", linestyle="--", linewidth=3, alpha=0.7)
    ax.plot(x_cpu, pred_real, label="FNO Prediction (Real)", color="red", alpha=0.8)
    
    ax.set_title(f"FNO Generated Wave Packet\n(Shift $x_0$: {test_i}, Momentum $k$: {test_k})")   
    ax.set_xlabel("Spatial Coordinate (x)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # Save frame plot output
    plt.savefig("fno_initial_psi_eval.png", dpi=150, bbox_inches='tight')
    print("Verification plot successfully rendered and saved as 'fno_initial_psi_eval.png'.")
    plt.show()
