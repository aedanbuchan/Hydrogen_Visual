import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre, sph_harm, factorial

# ── Physics functions ──────────────────────────────────────────────

def radial_wavefunction(n, l, r):
    a0 = 1.0
    rho = 2 * r / (n * a0)
    norm = np.sqrt(
        (2 / (n * a0))**3 *
        factorial(n - l - 1) /
        (2 * n * factorial(n + l)**3)
    )
    L = genlaguerre(n - l - 1, 2 * l + 1)
    return norm * np.exp(-rho / 2) * rho**l * L(rho)

def hydrogen_wavefunction_2d(n, l, m, grid_size=300):
    extent = 4 * n**2
    x = np.linspace(-extent, extent, grid_size)
    z = np.linspace(-extent, extent, grid_size)
    X, Z = np.meshgrid(x, z)
    Y = np.zeros_like(X)
    R = np.sqrt(X**2 + Y**2 + Z**2)
    R = np.where(R == 0, 1e-10, R)
    theta = np.arccos(np.clip(Z / R, -1, 1))
    phi = np.arctan2(Y, X)
    psi_r = radial_wavefunction(n, l, R)
    psi_ang = sph_harm(m, l, phi, theta).real
    density = (psi_r * psi_ang)**2
    return X, Z, density

# ── Streamlit UI ───────────────────────────────────────────────────

st.title("2D orbital cross-section")

n = st.slider("Principal quantum number (n)", 1, 5, 2)
l = st.slider("Angular momentum (l)", 0, max(n - 1, 0), 0)
m = st.slider("Magnetic quantum number (m)", -l, l, 0)

X, Z, density = hydrogen_wavefunction_2d(n, l, m)

fig, ax = plt.subplots(figsize=(6, 6))
extent = 4 * n**2
vmax = np.percentile(density, 99)
ax.imshow(density, extent=[-extent, extent, -extent, extent],
          origin='lower', cmap='magma', vmin=0, vmax=vmax)
ax.plot(0, 0, 'w+', markersize=10, markeredgewidth=1.5)
ax.set_xlabel("x (Bohr radii)", fontsize=12)
ax.set_ylabel("z (Bohr radii)", fontsize=12)
ax.set_title(f"Hydrogen |ψ|²  —  n={n}, l={l}, m={m}", fontsize=13)

st.pyplot(fig)
