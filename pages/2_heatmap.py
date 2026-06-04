import streamlit as st
import numpy as np
import plotly.graph_objects as go
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
    return x, z, density

# ── Page config ────────────────────────────────────────────────────

st.title("2D orbital cross-section")
st.caption("Probability density |ψ|² on the x-z plane. Hover to read values, scroll to zoom.")

orbital_names = ['s', 'p', 'd', 'f', 'g']

# ── Sidebar options ────────────────────────────────────────────────

st.sidebar.header("Plot options")
colormap = st.sidebar.selectbox(
    "Colourmap",
    ['Hot', 'Magma', 'Plasma', 'Viridis', 'Inferno', 'Turbo'],
    index=0
)
show_nucleus = st.sidebar.checkbox("Show nucleus", value=True)
grid_size = st.sidebar.select_slider(
    "Resolution", options=[100, 200, 300, 400], value=200
)

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

x, z, density = hydrogen_wavefunction_2d(n, l, m, grid_size=grid_size)
energy_ev = -13.6 / n**2
vmax = float(np.percentile(density, 99))

# ── Metrics row ────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
col1.metric("Orbital",       f"{n}{orbital_names[l]}")
col2.metric("Energy",        f"{energy_ev:.2f} eV")
col3.metric("Angular nodes", f"{l}")
col4.metric("Radial nodes",  f"{n - l - 1}")

# ── Plotly heatmap ─────────────────────────────────────────────────

fig = go.Figure()

fig.add_trace(go.Heatmap(
    x=x,
    z=density,
    y=z,
    colorscale=colormap,
    zmin=0,
    zmax=vmax,
    colorbar=dict(title="|ψ|²"),
    hovertemplate="x = %{x:.2f} a₀<br>z = %{y:.2f} a₀<br>|ψ|² = %{z:.4e}<extra></extra>"
))

# Nucleus marker
if show_nucleus:
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(symbol='cross', size=12, color='white', line=dict(width=2)),
        name='Nucleus',
        hovertemplate="Nucleus<extra></extra>"
    ))

extent = 4 * n**2
fig.update_layout(
    xaxis_title="x  (Bohr radii a₀)",
    yaxis_title="z  (Bohr radii a₀)",
    title=f"Hydrogen |ψ|²  —  {n}{orbital_names[l]}  (m={m})",
    xaxis=dict(range=[-extent, extent]),
    yaxis=dict(range=[-extent, extent], scaleanchor="x", scaleratio=1),
    height=550,
    margin=dict(t=60, b=40)
)

st.plotly_chart(fig, use_container_width=True)

# ── Expander ───────────────────────────────────────────────────────

with st.expander("What am I looking at?"):
    st.markdown(f"""
    A cross-section through the **x-z plane** (y=0). Brighter = higher probability of 
    finding the electron there.

    - **n={n}** sets the energy shell — higher n spreads the orbital further out
    - **l={l}** sets the shape — {orbital_names[l]}-type orbital
    - **m={m}** sets the orientation around the z-axis
    - **{l}** angular nodes and **{n - l - 1}** radial nodes

    💡 **Tip:** Scroll to zoom, click and drag to pan, hover for exact values.
    """)
