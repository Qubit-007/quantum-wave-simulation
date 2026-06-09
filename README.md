# 🌌 Quantum Wave Packet Simulation & AI Acceleration Suite

A GPU-accelerated computational physics suite designed to solve and accelerate the **Time-Dependent Schrödinger Equation** (TDSE). This repository contains a production-ready numerical solver utilizing **PyTorch** and the **Split-Step Fourier Method (SSFM)**, alongside an ongoing implementation of a **Fourier Neural Operator (FNO)** framework for deep-learning-driven spatial-temporal wave acceleration.

---

## 📁 Repository Architecture

This suite is modularly organized to demonstrate the progression from numerical simulations to advanced AI modeling:

* **`01_single_barrier/`**
  * `single_wall.py`: Tracks a free-moving wave packet colliding with a finite potential step barrier. Extracts and visualizes the complex wave function profile ($\psi$).
  * 

https://github.com/user-attachments/assets/d1264561-2d9d-4936-ac48-b7d9089a0294


* **`02_quantum_prisoner/`**
  * `infinite_well.py`: Simulates dual-barrier configuration bounds ("Quantum Prisoner") trapping a fast-moving particle.
  * `probability_density.py`: Maps the pure real-valued probability density profile ($|\psi|^2$) showing localized particle likelihood maps.
* **`03_ai_acceleration/`**
  * *(In Active Development)*: Core architectures for data generation and a custom **Fourier Neural Operator (FNO)** model aimed at predicting quantum states instantly without discrete loop iterations.

---

## 🧮 Numerical Engine: Split-Step Fourier Method (SSFM)

Standard finite-difference methods scale poorly and introduce structural numeric drift. To preserve total system probability, this engine separates the non-commuting Kinetic ($\hat{T}$) and Potential ($\hat{V}$) operators via a second-order split-step scheme over each infinitesimal time interval ($\Delta t$):

1. **Potential Half-Step (Real Space):** Computes time propagation under local fields:  
   $$\psi \leftarrow \psi \cdot \exp\left(-i \hat{V} \frac{\Delta t}{2}\right)$$
2. **Kinetic Full-Step (Momentum/Fourier Space):** Maps spatial coordinates to frequency domain via Fast Fourier Transforms (FFT) to apply a precise kinetic momentum push:  
   $$\tilde{\psi} \leftarrow \text{FFT}(\psi) \cdot \exp\left(-i \frac{\hbar k^2}{2m} \Delta t\right)$$
3. **Potential Half-Step (Real Space):** Inverse transforms (IFFT) back to local coordinate frames to execute the terminal potential energy update:  
   $$\psi \leftarrow \text{IFFT}(\tilde{\psi}) \cdot \exp\left(-i \hat{V} \frac{\Delta t}{2}\right)$$

---

## 🤖 Next Phase: Fourier Neural Operator (FNO) Acceleration

Traditional fully connected networks struggle with the highly oscillatory structures of quantum waveforms, often requiring billions of parameters to look "smooth". 

To bypass this limit, the upcoming `03_ai_acceleration` pipeline introduces a **Fourier Neural Operator**:
* **The Concept:** Instead of learning point-by-point pixel data, the FNO parameterizes the network's weights directly within the frequency domain.
* **The Pipeline:** The working numerical models in folders `01` and `02` act as data generators. The FNO ingests initial wave conditions $\psi(x, t_0)$ and predicts the global wave state $\psi(x, t_n)$ globally across an arbitrary continuum without step-by-step math evaluation loops.

---

## 🚀 Installation & Local Execution

Ensure your environment is configured with PyTorch (CUDA capabilities heavily suggested for optimized batch generation):

```bash
pip install torch numpy matplotlib
```

To initialize one of the active physics simulation visualizers, switch directory contexts and trigger execution:
```bash
python 02_quantum_prisoner.py
```

If you have any questions for if something doesn't make sense you can ask...
Have a good day! 😊 
