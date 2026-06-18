import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Define a Reusable FNO Block ---
class FNO1dBlock(nn.Module):
    def __init__(self, modes, width):
        super().__init__()
        self.modes = modes
        self.width = width
        
        # Fourier Weight Tensor
        self.fourier_weights = nn.Parameter(
            torch.view_as_complex(torch.randn(width, width, modes, 2) * (1.0 / width))
        )
        # Spatial Skip Connection
        self.w = nn.Conv1d(width, width, 1)

    def forward(self, x):
        # x shape: (batch, width, x_dim)
        x_complex = x.to(torch.cfloat)
        x_ft = torch.fft.fft(x_complex, dim=-1)
        
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :self.modes] = torch.einsum("bci,coi->boi", x_ft[:, :, :self.modes], self.fourier_weights)
        
        x_fourier = torch.fft.ifft(out_ft, dim=-1)
        x_spatial = self.w(x)
        
        # Combine and activate
        return F.gelu(torch.real(x_fourier) + x_spatial)

# --- 2. Build the Deep FNO ---
class Deep_Wave_FNO(nn.Module):
    def __init__(self, modes, width=64):
        super().__init__()
        self.p = nn.Conv1d(3, width, 1) # Input: [x, i, k]
        
        # Stack 5 FNO blocks for high-capacity learning!
        self.block1 = FNO1dBlock(modes, width)
        self.block2 = FNO1dBlock(modes, width)
        self.block3 = FNO1dBlock(modes, width)
        self.block4 = FNO1dBlock(modes, width)
        self.block5 = FNO1dBlock(modes, width)
        
        self.q = nn.Conv1d(width, 2, 1) # Output: [Real, Imag]

    def forward(self, x_grid, i, k):
        batch, x_dim = x_grid.shape
        i_tensor = torch.tensor([i], dtype=x_grid.dtype, device=x_grid.device).view(1, 1).expand(batch, x_dim)
        k_tensor = torch.tensor([k], dtype=x_grid.dtype, device=x_grid.device).view(1, 1).expand(batch, x_dim)
        
        x = torch.stack((x_grid, i_tensor, k_tensor), dim=1) 
        x = self.p(x)
        
        # Pass through the deep layers
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        out = self.q(x)
        return torch.complex(out[:, 0, :], out[:, 1, :])

# --- 3. Training and Testing Loop with FIXED logic ---
model = Deep_Wave_FNO(modes=64, width=64).to(device) 

try:
    model.load_state_dict(torch.load("wave_generator_fno_02.pth"))
    print("✓ Successfully loaded pre-trained weights from 'wave_generator_fno_02.pth'")
except FileNotFoundError:
    print("❌ Could not find the saved model! Make sure the file is in the same folder.")
    exit()

# lr is reduced to 0.0002 for fine-tuning stability
optimiser = optim.Adam(model.parameters(), lr=0.0002)
x = torch.linspace(-10, 15, 1000, device=device)

# batch_size = 8



# --- Testing ---
test_i = 3.0
test_k = 17.0
true_wave = torch.exp(-(x- test_i) ** 2) * torch.exp(1j * test_k * x)

# Let the AI generate it
pred_wave = model(x.unsqueeze(0), test_i, test_k).squeeze()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x.cpu().numpy(), np.real(true_wave.cpu().numpy()), label="True Real Part", linestyle="--", linewidth=3, alpha=0.7)
ax.plot(x.cpu().numpy(), np.real(pred_wave.detach().cpu().numpy()), label="AI Predicted Real Part", color="red", alpha=0.8)
ax.set_title(f"FNO Generated Wave Packet (Shift: {test_i}, Momentum: {test_k})")   
ax.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.show()  

# save the model's weights to a file in the current folder if it worked finely
print("\n--- Training Session Complete ---")
save_choice = input("The plot is closed. Do you want to save this model's weights? (y/n): ")

if save_choice.lower() == 'y':
    # Save the model's weights to a file in the current folder
    torch.save(model.state_dict(), "wave_generator_fno_03.pth")
    print("✓ Model saved successfully as 'wave_generator_fno_03.pth'.")
else:
    print("Model discarded. See you next time!")