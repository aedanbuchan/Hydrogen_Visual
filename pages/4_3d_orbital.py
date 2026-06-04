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
        (2 * n * factorial(n + l))
    )
    L = genlaguerre(n - l - 1, 2 * l + 1)
    return norm * np.exp(-rho / 2) * rho**l * L(rho)

def spherical_harmonic_real(l, m, theta, phi):
    """Real form of the spherical harmonic Y_lm."""
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

def spherical_harmonic_complex(l, m, theta, phi):
    """Complex form — returns real and imaginary parts separately."""
    m_abs = abs(m)
    norm = np.sqrt(
        (2 * l + 1) / (4 * np.pi) *
        factorial(l - m_abs) / factorial(l + m_abs)
    )
    P = lpmv(m_abs, l, np.cos(theta))
    real_part = norm * P * np.cos(m * phi)
    imag_part = norm * P * np.sin(m * phi)
    return real_part, imag_part

def sample_electron_cloud(X, Y, Z, density, n_points=50000):
    probs = density.flatten().astype(np.float64)

    probs = np.maximum(probs, 0)

    total = probs.sum()

    if total == 0:
        raise ValueError("Probability density is zero everywhere.")

    probs /= total

    idx = np.random.choice(
        len(probs),
        size=min(n_points, len(probs)),
        replace=True,
        p=probs
    )

    return (
        X.flatten()[idx],
        Y.flatten()[idx],
        Z.flatten()[idx],
        probs[idx]
    )
    
def compute_wavefunction_3d(n, l, m, grid_size, use_complex):
    """
    Returns x, y, z grids and wavefunction values on a 3D grid.
    If use_complex: returns |ψ|² (probability density)
    If real: returns signed ψ (to show phase via lobes)
    """
    extent = 4 * n**2
    coords = np.linspace(-extent, extent, grid_size)
    X, Y, Z = np.meshgrid(coords, coords, coords)

    R = np.sqrt(X**2 + Y**2 + Z**2)
    R = np.where(R == 0, 1e-10, R)
    theta = np.arccos(np.clip(Z / R, -1, 1))
    phi   = np.arctan2(Y, X)

    psi_r = radial_wavefunction(n, l, R)

    if use_complex:
        real_part, imag_part = spherical_harmonic_complex(l, m, theta, phi)
        psi = psi_r * (real_part + 1j * imag_part)
        values = np.abs(psi)**2          # probability density
        phase  = None
    else:
        psi_ang = spherical_harmonic_real(l, m, theta, phi)
        values  = psi_r * psi_ang        # signed wavefunction
        phase   = np.sign(values)        # +1 / -1 for lobe colouring
        values  = np.abs(values)         # magnitude for isosurface

    return coords, coords, coords, values, phase


# ── Page config ────────────────────────────────────────────────────

st.title("3D orbital viewer")
st.caption("Rotate with mouse drag · Zoom with scroll · Hover for values")

orbital_names = ['s', 'p', 'd', 'f', 'g']

# ── Sidebar ────────────────────────────────────────────────────────

st.sidebar.header("Render options")

render_mode = st.sidebar.radio(
    "Render mode",
    [
        "Isosurface",
        "Electron cloud",
        "Volume (experimental)"
    ],
    index=0
)

wf_form = st.sidebar.radio(
    "Wavefunction form",
    ["Real  (signed lobes)", "Complex  (|ψ|² density)"],
    index=0
)
use_complex = wf_form.startswith("Complex")

isovalue = st.sidebar.slider(
    "Isosurface threshold",
    min_value=1, max_value=99, value=85,
    help="Higher = smaller surface closer to the peak density"
) / 100.0

density_cutoff = st.sidebar.slider(
    "Density cutoff (%)",
    min_value=0.0,
    max_value=99.9,
    value=95.0,
    step=0.5,
    help=(
        "Discard the lowest-density regions. "
        "Higher values show only the most probable electron locations."
    )
)

opacity = st.sidebar.slider("Opacity", 10, 100, 60) / 100.0

grid_size = st.sidebar.select_slider(
    "Grid resolution",
    options=[20, 30, 40, 50],
    value=30,
    help="Higher = smoother but slower to compute"
)

cloud_points = st.sidebar.slider(
    "Cloud particles",
    min_value=5000,
    max_value=100000,
    value=40000,
    step=5000
)

colorscale = st.sidebar.selectbox(
    "Colour scale",
    ["RdBu", "Viridis", "Plasma", "Hot", "Turbo"],
    index=0
)

# ── Quantum number controls ────────────────────────────────────────

n = st.slider("Principal quantum number (n)", 1, 5, 2)

if n == 1:
    l = 0
    st.info("For n=1, l can only be 0  (1s orbital)")
else:
    l = st.slider("Angular momentum (l)", 0, n - 1, 1)

if l == 0:
    m = 0
    st.info("For l=0, m can only be 0")
else:
    m = st.slider("Magnetic quantum number (m)", -l, l, 0)

# ── Metrics ────────────────────────────────────────────────────────

energy_ev = -13.6 / n**2
c1, c2, c3, c4 = st.columns(4)
c1.metric("Orbital",       f"{n}{orbital_names[l]}")
c2.metric("Energy",        f"{energy_ev:.2f} eV")
c3.metric("Angular nodes", f"{l}")
c4.metric("Radial nodes",  f"{n - l - 1}")

# ── Compute ────────────────────────────────────────────────────────

with st.spinner("Computing wavefunction..."):
    x, y, z, values, phase = compute_wavefunction_3d(
        n, l, m, grid_size, use_complex
    )

X, Y, Z = np.meshgrid(x, y, z)

# ── Convert to physical density ─────────────────────────────
if use_complex:
    density_values = np.abs(values) ** 2   # |ψ|² is correct probability
else:
    density_values = values ** 2           # signed ψ is not physical density

cutoff_value = np.percentile(density_values, density_cutoff)

# ── Log version for volume rendering ────────────────────────
log_density = np.log10(
    np.maximum(density_values, 1e-20)
)

# ── Isosurface threshold (keep your existing UI control) ────
threshold = np.percentile(
    density_values,
    isovalue * 100
)

# ── Build figure ───────────────────────────────────────────────────

fig = go.Figure()

if render_mode == "Isosurface":

    if use_complex:

        fig.add_trace(
            go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=values.flatten(),
                isomin=threshold,
                isomax=float(values.max()),
                surface_count=2,
                colorscale=colorscale,
                opacity=opacity,
                caps=dict(
                    x_show=False,
                    y_show=False,
                    z_show=False
                ),
                showscale=True,
                colorbar=dict(title="|ψ|²"),
                hovertemplate=(
                    "x=%{x:.1f}<br>"
                    "y=%{y:.1f}<br>"
                    "z=%{z:.1f}<br>"
                    "|ψ|²=%{value:.4e}"
                    "<extra></extra>"
                )
            )
        )

    else:

        pos_vals = np.where(phase > 0, values, 0.0)
        neg_vals = np.where(phase < 0, values, 0.0)

        if pos_vals.max() > threshold:

            fig.add_trace(
                go.Isosurface(
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=Z.flatten(),
                    value=pos_vals.flatten(),
                    isomin=threshold,
                    isomax=float(pos_vals.max()),
                    surface_count=1,
                    colorscale=[
                        [0, "royalblue"],
                        [1, "royalblue"]
                    ],
                    opacity=opacity,
                    showscale=False,
                    caps=dict(
                        x_show=False,
                        y_show=False,
                        z_show=False
                    ),
                    name="+ phase"
                )
            )

        if neg_vals.max() > threshold:

            fig.add_trace(
                go.Isosurface(
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=Z.flatten(),
                    value=neg_vals.flatten(),
                    isomin=threshold,
                    isomax=float(neg_vals.max()),
                    surface_count=1,
                    colorscale=[
                        [0, "tomato"],
                        [1, "tomato"]
                    ],
                    opacity=opacity,
                    showscale=False,
                    caps=dict(
                        x_show=False,
                        y_show=False,
                        z_show=False
                    ),
                    name="− phase"
                )
            )

elif render_mode == "Electron cloud":

    density = density_values.copy()

    density = np.where(
        density >= cutoff_value,
        density,
        0
    )

    cx, cy, cz, cd = sample_electron_cloud(
        X,
        Y,
        Z,
        density,
        cloud_points
    )

    fig.add_trace(
        go.Scatter3d(
            x=cx,
            y=cy,
            z=cz,
            mode="markers",
            marker=dict(
                size=2,
                opacity=0.12,
                color=np.log10(cd + 1e-20),
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title="log density")
            ),
            hoverinfo="skip",
            name="Electron cloud"
        )
    )
else:

    fig.add_trace(
        go.Volume(
            x=X.flatten(),
            y=X.flatten(),
            z=Z.flatten(),
            value=log_density.flatten(),
            isomin=-5,
            isomax=float(log_density.max()),
            opacity=0.03,
            surface_count=25,
            colorscale=colorscale,
            caps=dict(
                x_show=False,
                y_show=False,
                z_show=False
            ),
            colorbar=dict(
                title="log density"
            )
        )
    )

# Nucleus marker
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode='markers',
    marker=dict(size=4, color='yellow', symbol='circle'),
    name='Nucleus',
    hovertemplate="Nucleus<extra></extra>"
))

extent = 4 * n**2
fig.update_layout(
    height=620,
    margin=dict(l=0, r=0, t=40, b=0),
    title=f"{n}{orbital_names[l]} orbital  (m={m})  —  "
          f"{'Complex |ψ|²' if use_complex else 'Real ψ (blue=+, red=−)'}",
    scene=dict(
        xaxis=dict(title="x (a₀)", range=[-extent, extent],
                   backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(150,150,150,0.2)"),
        yaxis=dict(title="y (a₀)", range=[-extent, extent],
                   backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(150,150,150,0.2)"),
        zaxis=dict(title="z (a₀)", range=[-extent, extent],
                   backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(150,150,150,0.2)"),
        bgcolor="rgba(0,0,0,0)",
        aspectmode="cube"
    ),
    legend=dict(x=0.01, y=0.99)
)

st.plotly_chart(fig, use_container_width=True)

# ── Expander ───────────────────────────────────────────────────────

with st.expander("What am I looking at?"):
    st.markdown(f"""
    A 3D representation of the hydrogen **{n}{orbital_names[l]}** orbital with m={m}.

    **Render modes:**
    - **Isosurface** — draws a shell enclosing the region where |ψ| exceeds a threshold.
      Use the sidebar slider to move the surface in or out.
    - **Volume cloud** — shows the full density as a semi-transparent cloud.
      More physically accurate but slower to render.

    **Wavefunction forms:**
    - **Real** — uses real spherical harmonics. Lobes are coloured by phase:
      🔵 blue = positive ψ,  🔴 red = negative ψ
    - **Complex** — uses complex spherical harmonics and plots |ψ|² (probability density only,
      no phase information)

    **Quantum numbers:**
    - n={n} → energy shell,  l={l} → {orbital_names[l]}-type shape,  m={m} → orientation
    - Angular nodes: {l},  Radial nodes: {n - l - 1}

    💡 **Tips:**
    - Drag to rotate, scroll to zoom
    - Lower the isovalue threshold in the sidebar to see more of the orbital
    - Start with low resolution for speed, increase for a smoother surface
    """)
