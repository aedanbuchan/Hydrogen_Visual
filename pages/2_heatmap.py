import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre, lpmv
from math import factorial

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

def spherical_harmonic(l, m, theta, phi):
    """Real spherical harmonic Y_lm."""
    m_abs = abs(m)
    norm = np.sqrt(
        (2 * l + 1) / (4 * np.pi) *
        factorial(l - m_abs) / factorial(l + m_abs)
    )
    P = lpmv(m_abs, l, np.cos(theta))
    if m > 0:
        return norm * P * np.cos(m_abs * phi) * np.sqrt(2)
    elif m < 0:
        return norm * P * np.sin(m_abs * phi) * np.sqrt(2)
    else:
        return norm * P

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
    psi_ang = spherical_harmonic(l, m, theta, phi)
    density = (psi_r * psi_ang)**2
    return X, Z, density

# ── Page config ────────────────────────────────────────────────────

st.title("2D orbital cross-section")
st.caption("Probability density |ψ|² shown as a heatmap on the x-z plane (y=0).")

orbital_names = ['s', 'p', 'd', 'f', 'g']

# ── Sidebar options ────────────────────────────────────────────────

st.sidebar.header("Plot options")
colormap = st.sidebar.selectbox("Colourmap", ['magma', 'inferno', 'viridis', 'plasma', 'hot'], index=0)
show_nucleus = st.sidebar.checkbox("Show nucleus", value=True)
grid_size = st.sidebar.select_slider("Resolution", options=[100, 200, 300, 400], value=200)

# ── Quantum number sliders ─────────────────────────────────────────

n = st.slider("Principal quantum number (n)", 1, 5, 2)

if n == 1:
    l = 0
    st.info("For n=1, l can only be 0  (1s orbital)")
else:
    l = st.slider("Angular momentum (l)", 0, n - 1, 0)

if l == 0:
    m = 0
    st.info("For l=0, m can only be 0")
else:
    m = st.slider("Magnetic quantum number (m)", -l, l, 0)

# ── Compute ────────────────────────────────────────────────────────

X, Z, density = hydrogen_wavefunction_2d(n, l, m, grid_size=grid_size)

# ── Metrics row ────────────────────────────────────────────────────

energy_ev = -13.6 / n**2
col1, col2, col3, col4 = st.columns(4)
col1.metric("Orbital",        f"{n}{orbital_names[l]}")
col2.metric("Energy",         f"{energy_ev:.2f} eV")
col3.metric("Angular nodes",  f"{l}")
col4.metric("Radial nodes",   f"{n - l - 1}")

# ── Plot ───────────────────────────────────────────────────────────

extent = 4 * n**2
vmax = np.percentile(density, 99)

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(
    density,
    extent=[-extent, extent, -extent, extent],
    origin='lower',
    cmap=colormap,
    vmin=0,
    vmax=vmax
)

if show_nucleus:
    ax.plot(0, 0, 'w+', markersize=12, markeredgewidth=2)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("|ψ|²  (probability density)", fontsize=10)

ax.set_xlabel("x  (Bohr radii a₀)", fontsize=12)
ax.set_ylabel("z  (Bohr radii a₀)", fontsize=12)
ax.set_title(f"Hydrogen |ψ|²  —  {n}{orbital_names[l]}  (m={m})", fontsize=13)

st.pyplot(fig)

# ── Expander with explanation ──────────────────────────────────────

with st.expander("What am I looking at?"):
    st.markdown(f"""
    This is a cross-section through the **x-z plane** (y=0) showing where the 
    electron is most likely to be found. Brighter regions = higher probability.

    - **n={n}** — principal quantum number, sets the energy shell
    - **l={l}** — angular momentum, determines orbital shape ({orbital_names[l]}-type)
    - **m={m}** — magnetic quantum number, sets the orientation
    - There are **{l}** angular nodes and **{n - l - 1}** radial nodes
    - The white **+** marks the nucleus at the origin
    """)
