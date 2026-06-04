import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.special import genlaguerre
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

def radial_probability(n, l, r):
    R = radial_wavefunction(n, l, r)
    return r**2 * R**2

# ── Page config ────────────────────────────────────────────────────

st.title("Radial probability density")
st.caption("Probability of finding the electron at distance r from the nucleus.")

orbital_names = ['s', 'p', 'd', 'f', 'g']

# ── Sidebar options ────────────────────────────────────────────────

st.sidebar.header("Plot options")
show_peak  = st.sidebar.checkbox("Show most probable radius", value=True)
show_nodes = st.sidebar.checkbox("Show radial nodes", value=True)
fill       = st.sidebar.checkbox("Fill under curve", value=True)
compare    = st.sidebar.toggle("Compare all l for this n")

# ── Quantum number sliders ─────────────────────────────────────────

n = st.slider("Principal quantum number (n)", 1, 5, 1)

if n == 1:
    l = 0
    st.info("For n=1, l can only be 0  (1s orbital)")
else:
    l = st.slider("Angular momentum (l)", 0, n - 1, 0)

# ── Compute ────────────────────────────────────────────────────────

r = np.linspace(0, 4 * n**2 + 10, 1000)
P = radial_probability(n, l, r)
r_peak = r[np.argmax(P)]
energy_ev = -13.6 / n**2

# ── Metrics row ────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
col1.metric("Orbital",         f"{n}{orbital_names[l]}")
col2.metric("Energy",          f"{energy_ev:.2f} eV")
col3.metric("Radial nodes",    f"{n - l - 1}")
col4.metric("Most probable r", f"{r_peak:.2f} a₀")

# ── Plotly figure ──────────────────────────────────────────────────

colors = ['royalblue', 'tomato', 'seagreen', 'darkorchid', 'darkorange']

fig = go.Figure()

if compare:
    for l_i in range(n):
        P_i = radial_probability(n, l_i, r)
        name = f"{n}{orbital_names[l_i]}"
        fig.add_trace(go.Scatter(
            x=r, y=P_i,
            mode='lines',
            name=name,
            line=dict(color=colors[l_i], width=2),
            fill='tozeroy' if fill else 'none',
            fillcolor=f'rgba({",".join(str(int(c*255)) for c in plt_to_rgb(colors[l_i]))},0.08)' if fill else None,
            hovertemplate=f"<b>{name}</b><br>r = %{{x:.2f}} a₀<br>P(r) = %{{y:.4f}}<extra></extra>"
        ))
else:
    # Main curve
    fig.add_trace(go.Scatter(
        x=r, y=P,
        mode='lines',
        name=f"{n}{orbital_names[l]}",
        line=dict(color='royalblue', width=2.5),
        fill='tozeroy' if fill else 'none',
        fillcolor='rgba(65,105,225,0.12)' if fill else None,
        hovertemplate="r = %{x:.2f} a₀<br>P(r) = %{y:.6f}<extra></extra>"
    ))

    # Most probable radius line
    if show_peak:
        fig.add_vline(
            x=r_peak,
            line=dict(color='tomato', dash='dash', width=1.5),
            annotation_text=f"r = {r_peak:.2f} a₀",
            annotation_position="top right",
            annotation_font_color='tomato'
        )

    # Radial nodes
    if show_nodes:
        zero_crossings = np.where(np.diff(np.sign(P[1:])))[0]
        for i, zc in enumerate(zero_crossings):
            fig.add_vline(
                x=r[zc],
                line=dict(color='gray', dash='dot', width=1),
                annotation_text="node" if i == 0 else "",
                annotation_font_color='gray',
                annotation_position="top left"
            )

fig.update_layout(
    xaxis_title="r  (Bohr radii a₀)",
    yaxis_title="P(r) = r²|R(r)|²",
    title=f"Radial probability density — "
          f"{'all l for n=' + str(n) if compare else str(n) + orbital_names[l] + ' orbital'}",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=60, b=40),
    height=450
)

st.plotly_chart(fig, use_container_width=True)

# ── Expander ───────────────────────────────────────────────────────

with st.expander("What am I looking at?"):
    st.markdown(f"""
    **P(r) = r²|R(r)|²** is the radial probability density — the probability of 
    finding the electron in a thin shell at distance **r** from the nucleus.

    - The **peak** at r = {r_peak:.2f} a₀ is the most likely distance
    - There are **{n - l - 1}** radial nodes where the probability drops to zero
    - Higher **n** pushes the electron further from the nucleus
    - Higher **l** suppresses the probability near the nucleus

    💡 **Tip:** Hover over the plot to read exact values. Click legend items to toggle traces.
    """)
