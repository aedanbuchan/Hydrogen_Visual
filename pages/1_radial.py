import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre, factorial

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

def radial_probability(n, l, r):
    R = radial_wavefunction(n, l, r)
    return r**2 * R**2

# ── Streamlit UI ───────────────────────────────────────────────────

st.title("Radial probability density")

n = st.slider("Principal quantum number (n)", 1, 5, 1)
l = st.slider("Angular momentum (l)", 0, max(n - 1, 0), 0)

r = np.linspace(0, 4 * n**2 + 10, 1000)
P = radial_probability(n, l, r)

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(r, P, linewidth=2, color='royalblue')
ax.fill_between(r, P, alpha=0.15, color='royalblue')

r_peak = r[np.argmax(P)]
ax.axvline(r_peak, color='tomato', linestyle='--', linewidth=1.2,
           label=f"Most probable r = {r_peak:.2f} a₀")

n_nodes = n - l - 1
ax.set_xlabel("r  (Bohr radii a₀)", fontsize=12)
ax.set_ylabel("P(r) = r²|R(r)|²", fontsize=12)
ax.set_title(f"Radial probability density — {n}{['s','p','d','f'][l]} orbital", fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(0, None)
ax.set_ylim(0, None)
ax.text(0.97, 0.95, f"Radial nodes: {n_nodes}",
        transform=ax.transAxes, ha='right', va='top', fontsize=10, color='gray')

st.pyplot(fig)
