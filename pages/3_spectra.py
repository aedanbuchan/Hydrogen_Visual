import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ── Physics ────────────────────────────────────────────────────────

def energy(n):
    return -13.6 / n**2

def transition_wavelength_nm(n_i, n_f):
    delta_e = energy(n_i) - energy(n_f)   # eV
    if delta_e <= 0:
        return None
    # E = hc/λ → λ = hc/E
    hc = 1239.84  # eV·nm
    return hc / delta_e

def series_name(n_f):
    names = {1: "Lyman", 2: "Balmer", 3: "Paschen",
             4: "Brackett", 5: "Pfund"}
    return names.get(n_f, f"n={n_f} series")

def wavelength_to_rgb(wavelength_nm):
    """Convert visible wavelength to an RGB colour string."""
    wl = wavelength_nm
    if 380 <= wl < 440:
        r, g, b = (440 - wl) / 60, 0, 1.0
    elif 440 <= wl < 490:
        r, g, b = 0, (wl - 440) / 50, 1.0
    elif 490 <= wl < 510:
        r, g, b = 0, 1.0, (510 - wl) / 20
    elif 510 <= wl < 580:
        r, g, b = (wl - 510) / 70, 1.0, 0
    elif 580 <= wl < 645:
        r, g, b = 1.0, (645 - wl) / 65, 0
    elif 645 <= wl <= 780:
        r, g, b = 1.0, 0, 0
    else:
        r, g, b = 0.5, 0.5, 0.5   # UV or IR — grey
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

def spectral_region(wavelength_nm):
    if wavelength_nm is None:
        return "—"
    if wavelength_nm < 10:
        return "X-ray"
    elif wavelength_nm < 122:
        return "Far UV"
    elif wavelength_nm < 380:
        return "Ultraviolet"
    elif wavelength_nm <= 780:
        return "Visible"
    elif wavelength_nm <= 2500:
        return "Near infrared"
    else:
        return "Infrared"

# ── Page config ────────────────────────────────────────────────────

st.title("Energy level transitions")
st.caption("Select an initial and final energy level to simulate an electron transition.")

N_LEVELS = 7   # number of levels to draw

# ── Sidebar ────────────────────────────────────────────────────────

st.sidebar.header("Options")
show_all_series = st.sidebar.toggle("Show all Balmer series lines", value=False)
equal_spacing = st.sidebar.toggle("Equal Level Spacing", value=False)

# ── Level selectors ────────────────────────────────────────────────

col_a, col_b = st.columns(2)
with col_a:
    n_i = st.selectbox(
        "Initial level (excited state)",
        options=list(range(2, N_LEVELS + 1)),
        index=3,
        format_func=lambda n: f"n = {n}  ({energy(n):.2f} eV)"
    )
with col_b:
    n_f = st.selectbox(
        "Final level (ground / lower state)",
        options=list(range(1, n_i)),
        index=0,
        format_func=lambda n: f"n = {n}  ({energy(n):.2f} eV)"
    )

# ── Compute transition ─────────────────────────────────────────────

delta_e   = abs(energy(n_i) - energy(n_f))
wavelength = transition_wavelength_nm(n_i, n_f)
series     = series_name(n_f)
region     = spectral_region(wavelength)
arrow_col  = wavelength_to_rgb(wavelength) if wavelength else "gray"

# ── Metrics ────────────────────────────────────────────────────────

st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Series",          series)
m2.metric("Energy released", f"{delta_e:.3f} eV")
m3.metric("Wavelength",      f"{wavelength:.1f} nm" if wavelength else "—")
m4.metric("Region",          region)
st.divider()


# ── Y pos helper ─────────────────────────────

def y_pos(n):
    """Return y-axis values for levels of n in diagram""" 
    if equal_spacing:
        y_min, y_max = energy(1), energy(N_LEVELS)
        return y_min + (y_max - y_min) * (n - 1) / (N_LEVELS - 1)
    return energy(n)

# ── Build Plotly energy level diagram ─────────────────────────────

fig = go.Figure()

level_x   = [0.2, 0.8]   # left and right extent of each level line
arrow_x   = 0.55          # x position of the transition arrow
label_x   = 0.82          # x position of energy labels
n_label_x = 0.17          # x position of n= labels

for n in range(1, N_LEVELS + 1):
    e = y_pos(n)
    # Energy level line
    is_active = (n == n_i or n == n_f)
    fig.add_shape(
        type="line",
        x0=level_x[0], x1=level_x[1],
        y0=e, y1=e,
        line=dict(
            color="royalblue" if is_active else "lightsteelblue",
            width=3 if is_active else 1.5,
            dash="solid"
        )
    )
    # n= label on left
    fig.add_annotation(
        x=n_label_x, y=e,
        text=f"<b>n={n}</b>",
        showarrow=False,
        xanchor="right",
        font=dict(size=13, color="royalblue" if is_active else "gray")
    )
    # Energy label on right
    fig.add_annotation(
        x=label_x, y=e,
        text=f"{e:.2f} eV",
        showarrow=False,
        xanchor="left",
        font=dict(size=11, color="gray")
    )

# Ionisation limit
fig.add_shape(
    type="line",
    x0=level_x[0], x1=level_x[1],
    y0=0, y1=0,
    line=dict(color="tomato", width=1, dash="dash")
)
fig.add_annotation(
    x=label_x, y=0,
    text="0.00 eV  (ionisation)",
    showarrow=False, xanchor="left",
    font=dict(size=11, color="tomato")
)

# Transition arrow
e_initial = y_pos(n_i)
e_final   = y_pos(n_f)

fig.add_annotation(
    x=arrow_x, y=e_final,
    ax=arrow_x, ay=e_initial,
    xref="x", yref="y",
    axref="x", ayref="y",
    showarrow=True,
    arrowhead=3,
    arrowsize=1.5,
    arrowwidth=3,
    arrowcolor=arrow_col,
)

# Wavelength label on arrow
fig.add_annotation(
    x=arrow_x + 0.04,
    y=(y_pos(n_i) + y_pos(n_f)) / 2,
    text=f"<b>{wavelength:.1f} nm</b><br>{series}",
    showarrow=False,
    xanchor="left",
    font=dict(size=12, color=arrow_col)
)

# Optional: all Balmer series lines
if show_all_series:
    for n_upper in range(3, N_LEVELS + 1):
        wl = transition_wavelength_nm(n_upper, 2)
        col = wavelength_to_rgb(wl)
        e_up = y_pos(n_upper)
        e_dn = y_pos(2)
        fig.add_annotation(
            x=arrow_x - 0.08, y=e_dn,
            ax=arrow_x - 0.08, ay=e_up,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor=col,
            opacity=0.4
        )

fig.update_layout(
    height=600,
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis=dict(visible=False, range=[0, 1]),
    yaxis=dict(
        title="Energy (eV)",
        range=[y_pos(N_LEVELS) - 1, 0.8],
        gridcolor="rgba(200,200,200,0.2)"
    ),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig, use_container_width=True)

# ── Emission spectrum bar ──────────────────────────────────────────

st.subheader("Emission line")

spec_fig = go.Figure()

# Background spectrum gradient (380–780nm visible range)
for wl_start in range(380, 780, 5):
    col = wavelength_to_rgb(wl_start)
    spec_fig.add_shape(
        type="rect",
        x0=wl_start, x1=wl_start + 5,
        y0=0, y1=0.4,
        fillcolor=col,
        line=dict(width=0),
        opacity=0.35
    )

# Emission line
if 380 <= wavelength <= 780:
    spec_fig.add_shape(
        type="line",
        x0=wavelength, x1=wavelength,
        y0=0, y1=1,
        line=dict(color=arrow_col, width=3)
    )
    spec_fig.add_annotation(
        x=wavelength, y=1.05,
        text=f"{wavelength:.1f} nm",
        showarrow=False,
        font=dict(size=12, color=arrow_col)
    )
else:
    spec_fig.add_annotation(
        x=580, y=0.5,
        text=f"{wavelength:.1f} nm — outside visible range ({region})",
        showarrow=False,
        font=dict(size=13, color="gray")
    )

spec_fig.update_layout(
    height=130,
    margin=dict(l=20, r=20, t=30, b=30),
    xaxis=dict(
        title="Wavelength (nm)",
        range=[300, 850],
        gridcolor="rgba(200,200,200,0.1)"
    ),
    yaxis=dict(visible=False, range=[0, 1.3]),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(spec_fig, use_container_width=True)

# ── Expander ───────────────────────────────────────────────────────

with st.expander("What am I looking at?"):
    st.markdown(f"""
    When an electron drops from **n={n_i}** to **n={n_f}**, it releases energy as a photon.

    - **Energy released:** ΔE = {delta_e:.3f} eV
    - **Wavelength:** λ = hc/ΔE = {wavelength:.1f} nm
    - **Series:** Transitions ending at n={n_f} are called the **{series} series**
    - **Region:** This photon is in the **{region}** part of the spectrum

    The coloured bar shows where this emission line sits in the visible spectrum.
    Lines outside 380–780 nm are ultraviolet (Lyman) or infrared (Paschen+).
    """)
