import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ── Constants ──────────────────────────────────────────────────────

ALPHA      = 1 / 137.036
BOHR_MAG   = 5.7884e-5
HFS_1S     = 5.87433e-6
LAMB_SHIFT = 4.372e-6
HC_NM      = 1239.84

# ── Physics functions ──────────────────────────────────────────────

def energy_bohr(n):
    return -13.6 / n**2

def fine_structure_correction(n, l, j):
    if l == 0:
        return 0.0
    so = (ALPHA**2 / n**2) * (n / (j + 0.5) - 0.75)
    return energy_bohr(n) * so

def relativistic_correction(n, l):
    denom = 0.5 if l == 0 else l + 0.5
    return -(ALPHA**2 / 4) * (13.6 / n**4) * (4 * n / denom - 3)

def lamb_shift_correction(n, l, j):
    if n == 2 and l == 0:
        return LAMB_SHIFT
    return 0.0

def zeeman_shift(m, B_field):
    return m * BOHR_MAG * B_field

def hyperfine_shift(n, l, spin_parallel):
    if n == 1 and l == 0:
        return HFS_1S / 2 if spin_parallel else -HFS_1S / 2
    return 0.0

def get_j_values(l):
    if l == 0:
        return [0.5]
    return [l - 0.5, l + 0.5]

def total_energy(n, l, j, m, B_field, include_fine,
                 include_relativistic, include_lamb, include_zeeman):
    E = energy_bohr(n)
    if include_fine:
        E += fine_structure_correction(n, l, j)
    if include_relativistic:
        E += relativistic_correction(n, l)
    if include_lamb:
        E += lamb_shift_correction(n, l, j)
    if include_zeeman:
        E += zeeman_shift(m, B_field)
    return E

def transition_wavelength(E_i, E_f):
    delta_e = E_i - E_f
    if delta_e <= 0:
        return None
    return HC_NM / delta_e

def check_selection_rules(l_i, l_f, m_i, m_f, j_i, j_f):
    delta_l = l_f - l_i
    delta_m = m_f - m_i
    delta_j = abs(j_f - j_i)
    violations = []
    if delta_l not in [-1, 1]:
        violations.append(f"Δl = {delta_l} (must be ±1)")
    if delta_m not in [-1, 0, 1]:
        violations.append(f"Δm = {delta_m} (must be 0, ±1)")
    if delta_j not in [0.0, 1.0]:
        violations.append(f"Δj = {delta_j:.1f} (must be 0 or 1)")
    if j_i == 0 and j_f == 0:
        violations.append("j=0 → j=0 forbidden")
    if not violations:
        return True, "Electric dipole allowed", "allowed"
    if delta_l in [0, 2] and delta_m in [-2, -1, 0, 1, 2]:
        return False, "E1 forbidden — " + ", ".join(violations) + " (E2 quadrupole possible)", "quadrupole"
    return False, "Forbidden — " + ", ".join(violations), "forbidden"

def series_name(n_f):
    names = {1: "Lyman", 2: "Balmer", 3: "Paschen",
             4: "Brackett", 5: "Pfund", 6: "Humphreys"}
    return names.get(n_f, f"n={n_f} series")

def wavelength_to_rgb(wl):
    if wl is None:
        return "rgb(128,128,128)"
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
        return "rgb(180,180,180)"
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

def spectral_region(wl):
    if wl is None: return "—"
    if wl < 10:    return "X-ray"
    if wl < 122:   return "Far UV"
    if wl < 380:   return "Ultraviolet"
    if wl <= 780:  return "Visible"
    if wl <= 2500: return "Near IR"
    return "Infrared"

# ── Y position helper (equal spacing toggle) ──────────────────────

def y_pos(n, equal_spacing, N_LEVELS):
    if equal_spacing:
        y_min = energy_bohr(1)
        y_max = energy_bohr(N_LEVELS)
        return y_min + (y_max - y_min) * (n - 1) / (N_LEVELS - 1)
    return energy_bohr(n)

# ── Page config ────────────────────────────────────────────────────

st.title("Energy level transitions")
st.caption("Electric dipole selection rules · Fine structure · Zeeman effect · Hyperfine splitting")

N_LEVELS = 6
orbital_names = ['s', 'p', 'd', 'f', 'g']

# ── Sidebar — display options ──────────────────────────────────────

st.sidebar.header("Display options")
equal_spacing  = st.sidebar.toggle("Equal level spacing",          value=False)
show_forbidden = st.sidebar.checkbox("Show forbidden transitions",  value=True)
show_all_balmer = st.sidebar.toggle("Show all Balmer series",      value=False)
show_fine_split = st.sidebar.toggle("Show fine structure splitting",value=False)

st.sidebar.divider()
st.sidebar.header("Physics corrections")
include_fine         = st.sidebar.toggle("Fine structure (spin-orbit)", value=True)
include_relativistic = st.sidebar.toggle("Relativistic correction",     value=False)
include_lamb         = st.sidebar.toggle("Lamb shift",                  value=False)
include_zeeman       = st.sidebar.toggle("Zeeman effect",               value=False)
include_hyperfine    = st.sidebar.toggle("Hyperfine structure (1s)",    value=False)

B_field = 0.0
if include_zeeman:
    B_field = st.sidebar.slider("Magnetic field B (Tesla)", 0.1, 10.0, 1.0, 0.1)

# ── Quantum number selectors ───────────────────────────────────────

st.subheader("Initial state  (excited)")
ci1, ci2, ci3, ci4 = st.columns(4)
with ci1:
    n_i = st.selectbox("n  (initial)", list(range(2, N_LEVELS + 1)), index=2,
                       format_func=lambda n: f"n={n}  ({energy_bohr(n):.2f} eV)")
with ci2:
    l_i = st.selectbox("l  (initial)", list(range(0, n_i)),
                       format_func=lambda l: f"{l} ({orbital_names[l]})")
with ci3:
    j_vals_i = get_j_values(l_i)
    j_i = st.selectbox("j  (initial)", j_vals_i,
                       format_func=lambda j: f"{int(2*j)}/2")
with ci4:
    m_vals_i = [x/2 for x in range(-int(2*j_i), int(2*j_i)+1, 1)]
    m_i = st.selectbox("mⱼ  (initial)", m_vals_i,
                       format_func=lambda m: f"{int(2*m)}/2")

st.subheader("Final state  (lower)")
cf1, cf2, cf3, cf4 = st.columns(4)
with cf1:
    n_f = st.selectbox("n  (final)", list(range(1, n_i)),
                       format_func=lambda n: f"n={n}  ({energy_bohr(n):.2f} eV)")
with cf2:
    l_f = st.selectbox("l  (final)", list(range(0, n_f)),
                       format_func=lambda l: f"{l} ({orbital_names[l]})")
with cf3:
    j_vals_f = get_j_values(l_f)
    j_f = st.selectbox("j  (final)", j_vals_f,
                       format_func=lambda j: f"{int(2*j)}/2")
with cf4:
    m_vals_f = [x/2 for x in range(-int(2*j_f), int(2*j_f)+1, 1)]
    m_f = st.selectbox("mⱼ  (final)", m_vals_f,
                       format_func=lambda m: f"{int(2*m)}/2")

# ── Compute transition ─────────────────────────────────────────────

E_i = total_energy(n_i, l_i, j_i, m_i, B_field,
                   include_fine, include_relativistic,
                   include_lamb, include_zeeman)
E_f = total_energy(n_f, l_f, j_f, m_f, B_field,
                   include_fine, include_relativistic,
                   include_lamb, include_zeeman)

if include_hyperfine:
    E_i += hyperfine_shift(n_i, l_i, spin_parallel=True)
    E_f += hyperfine_shift(n_f, l_f, spin_parallel=False)

delta_e    = abs(E_i - E_f)
wavelength = transition_wavelength(E_i, E_f)
series     = series_name(n_f)
region     = spectral_region(wavelength)
arrow_col  = wavelength_to_rgb(wavelength)

allowed, rule_msg, transition_type = check_selection_rules(
    l_i, l_f, m_i, m_f, j_i, j_f
)

# ── Status banner ──────────────────────────────────────────────────

st.divider()
if transition_type == "allowed":
    st.success(f"✅ **Allowed transition** — {rule_msg}")
elif transition_type == "quadrupole":
    st.warning(f"⚠️ **Weakly allowed** — {rule_msg}")
else:
    st.error(f"🚫 **Forbidden transition** — {rule_msg}")

# ── Metrics ────────────────────────────────────────────────────────

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Series",          series)
m2.metric("ΔE",              f"{delta_e:.4f} eV")
m3.metric("Wavelength",      f"{wavelength:.2f} nm" if wavelength else "—")
m4.metric("Region",          region)
m5.metric("Transition type", transition_type.capitalize())
st.divider()

# ── Build Plotly energy level diagram ─────────────────────────────

fig = go.Figure()

level_x0  = 0.20
level_x1  = 0.75
arrow_x   = 0.50
label_x   = 0.77
n_label_x = 0.18

for n in range(1, N_LEVELS + 1):
    e      = y_pos(n, equal_spacing, N_LEVELS)
    e_true = energy_bohr(n)
    is_active = (n == n_i or n == n_f)

    # Main energy level line
    fig.add_shape(
        type="line",
        x0=level_x0, x1=level_x1,
        y0=e, y1=e,
        line=dict(
            color="royalblue" if is_active else "lightsteelblue",
            width=3 if is_active else 1.5
        )
    )

    # n= label
    fig.add_annotation(
        x=n_label_x, y=e,
        text=f"<b>n={n}</b>",
        showarrow=False, xanchor="right",
        font=dict(size=13, color="royalblue" if is_active else "gray")
    )

    # Energy label — always shows true energy even in equal spacing mode
    fig.add_annotation(
        x=label_x, y=e,
        text=f"{e_true:.2f} eV",
        showarrow=False, xanchor="left",
        font=dict(size=11, color="gray")
    )

    # Fine structure splitting
    if show_fine_split and include_fine and n <= 4:
        seen = []
        for l_fs in range(n):
            for j_fs in get_j_values(l_fs):
                e_fs = e_true + fine_structure_correction(n, l_fs, j_fs)
                if e_fs not in seen:
                    seen.append(e_fs)
                    split_offset = (e_fs - e_true) * 5000
                    fig.add_shape(
                        type="line",
                        x0=level_x0 + 0.02, x1=level_x1 - 0.02,
                        y0=e + split_offset, y1=e + split_offset,
                        line=dict(color="orange", width=1, dash="dot")
                    )

    # Zeeman splitting
    if include_zeeman and B_field > 0:
        for m_z in [-1, 0, 1]:
            z_shift = zeeman_shift(m_z, B_field) * 200
            fig.add_shape(
                type="line",
                x0=level_x0 + 0.05, x1=level_x1 - 0.05,
                y0=e + z_shift, y1=e + z_shift,
                line=dict(color="mediumseagreen", width=1, dash="dot")
            )

    # Hyperfine splitting on 1s
    if include_hyperfine and n == 1:
        for sign, label in [(1, "F=1"), (-1, "F=0")]:
            hfs_offset = sign * HFS_1S * 50000
            fig.add_shape(
                type="line",
                x0=level_x0 + 0.03, x1=level_x1 - 0.03,
                y0=e + hfs_offset, y1=e + hfs_offset,
                line=dict(color="mediumpurple", width=1, dash="dot")
            )
            fig.add_annotation(
                x=level_x1 - 0.02, y=e + hfs_offset,
                text=label, showarrow=False, xanchor="right",
                font=dict(size=9, color="mediumpurple")
            )

# Ionisation limit
fig.add_shape(
    type="line", x0=level_x0, x1=level_x1,
    y0=y_pos(N_LEVELS, equal_spacing, N_LEVELS) + (
        1.0 if equal_spacing else 0.8
    ),
    y1=y_pos(N_LEVELS, equal_spacing, N_LEVELS) + (
        1.0 if equal_spacing else 0.8
    ),
    line=dict(color="tomato", width=1, dash="dash")
)
fig.add_annotation(
    x=label_x,
    y=y_pos(N_LEVELS, equal_spacing, N_LEVELS) + (
        1.0 if equal_spacing else 0.8
    ),
    text="0.00 eV  (ionisation)",
    showarrow=False, xanchor="left",
    font=dict(size=11, color="tomato")
)

# Transition arrow — uses y_pos so arrow tracks equal spacing correctly
e_i_plot = y_pos(n_i, equal_spacing, N_LEVELS)
e_f_plot = y_pos(n_f, equal_spacing, N_LEVELS)

fig.add_annotation(
    x=arrow_x,  y=e_f_plot,
    ax=arrow_x, ay=e_i_plot,
    xref="x", yref="y", axref="x", ayref="y",
    showarrow=True,
    arrowhead=3 if allowed else 2,
    arrowsize=1.5,
    arrowwidth=3 if allowed else 2,
    arrowcolor=arrow_col if allowed else "gray",
)

# Arrow label
label_text = (
    f"<b>{wavelength:.1f} nm</b><br>{series}<br>"
    f"{'✅ Allowed' if allowed else '🚫 Forbidden'}"
)
fig.add_annotation(
    x=arrow_x + 0.04,
    y=(e_i_plot + e_f_plot) / 2,
    text=label_text,
    showarrow=False, xanchor="left",
    font=dict(size=11, color=arrow_col if allowed else "gray")
)

# All Balmer series
if show_all_balmer:
    for n_up in range(3, N_LEVELS + 1):
        wl_b  = transition_wavelength(energy_bohr(n_up), energy_bohr(2))
        col_b = wavelength_to_rgb(wl_b)
        fig.add_annotation(
            x=arrow_x - 0.10, y=y_pos(2, equal_spacing, N_LEVELS),
            ax=arrow_x - 0.10, ay=y_pos(n_up, equal_spacing, N_LEVELS),
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2,
            arrowsize=1, arrowwidth=1.5,
            arrowcolor=col_b, opacity=0.5
        )

# Legend
legend_y = y_pos(1, equal_spacing, N_LEVELS) - (
    1.5 if equal_spacing else 1.5
)
if show_fine_split and include_fine:
    fig.add_annotation(x=0.22, y=legend_y, text="── Fine structure",
                       showarrow=False, font=dict(size=10, color="orange"))
if include_zeeman:
    fig.add_annotation(x=0.40, y=legend_y, text="── Zeeman levels",
                       showarrow=False, font=dict(size=10, color="mediumseagreen"))
if include_hyperfine:
    fig.add_annotation(x=0.58, y=legend_y, text="── Hyperfine (1s)",
                       showarrow=False, font=dict(size=10, color="mediumpurple"))

y_min_plot = y_pos(1, equal_spacing, N_LEVELS)
y_max_plot = y_pos(N_LEVELS, equal_spacing, N_LEVELS)
y_range    = y_max_plot - y_min_plot

fig.update_layout(
    height=650,
    margin=dict(l=20, r=20, t=30, b=60),
    xaxis=dict(visible=False, range=[0, 1]),
    yaxis=dict(
        title="Energy (eV)" if not equal_spacing else "Energy level  (equally spaced)",
        range=[y_min_plot - y_range * 0.15, y_max_plot + y_range * 0.15],
        gridcolor="rgba(200,200,200,0.15)",
        showticklabels=not equal_spacing,
        ticks="" if equal_spacing else "outside"
    ),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig, use_container_width=True)

# ── Emission spectrum bar ──────────────────────────────────────────

st.subheader("Emission line")
spec_fig = go.Figure()

for wl_s in range(380, 780, 5):
    spec_fig.add_shape(
        type="rect",
        x0=wl_s, x1=wl_s + 5, y0=0, y1=0.4,
        fillcolor=wavelength_to_rgb(wl_s),
        line=dict(width=0), opacity=0.4
    )

if wavelength and 380 <= wavelength <= 780:
    spec_fig.add_shape(
        type="line",
        x0=wavelength, x1=wavelength, y0=0, y1=1,
        line=dict(
            color=arrow_col if allowed else "gray",
            width=3,
            dash="solid" if allowed else "dash"
        )
    )
    spec_fig.add_annotation(
        x=wavelength, y=1.1,
        text=f"{wavelength:.1f} nm" + ("" if allowed else "  (forbidden)"),
        showarrow=False,
        font=dict(size=12, color=arrow_col if allowed else "gray")
    )
else:
    spec_fig.add_annotation(
        x=580, y=0.5,
        text=f"{wavelength:.1f} nm — outside visible range ({region})"
             if wavelength else "No emission",
        showarrow=False,
        font=dict(size=13, color="gray")
    )

spec_fig.update_layout(
    height=130,
    margin=dict(l=20, r=20, t=20, b=30),
    xaxis=dict(title="Wavelength (nm)", range=[300, 850],
               gridcolor="rgba(200,200,200,0.1)"),
    yaxis=dict(visible=False, range=[0, 1.4]),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(spec_fig, use_container_width=True)

# ── Physics expanders ──────────────────────────────────────────────

with st.expander("Selection rules"):
    st.markdown(f"""
    Electric dipole (E1) transitions require:
    - **Δl = ±1** — parity must change
    - **Δm = 0, ±1** — angular momentum conservation with photon spin
    - **Δj = 0, ±1** — but j=0 → j=0 is forbidden

    This transition has Δl = {l_f - l_i}, Δm = {int(2*(m_f - m_i))}/2, Δj = {abs(j_f - j_i):.1f}

    **{rule_msg}**

    Forbidden transitions can still occur via higher-order multipoles
    (electric quadrupole E2, magnetic dipole M1) but are typically
    10⁶× weaker than allowed transitions.
    """)

with st.expander("Fine structure"):
    st.markdown(f"""
    Fine structure arises from two corrections of order α² ≈ (1/137)²:

    **1. Spin-orbit coupling** — interaction between electron spin and the
    magnetic field it experiences moving through the nuclear electric field:

    ΔE_SO = (α²/2n³) · (13.6 eV) · [1/(j+½) − 1/l]  for l ≠ 0

    **2. Relativistic (Darwin) correction** — accounts for the relativistic
    mass increase of the electron and zitterbewegung (only for l=0):

    ΔE_rel = −(α²/4n⁴) · (13.6 eV) · (4n/(l+½) − 3)

    For n={n_i}, l={l_i}, j={int(2*j_i)}/2:
    Fine structure correction = {fine_structure_correction(n_i, l_i, j_i)*1e6:.2f} μeV
    Relativistic correction   = {relativistic_correction(n_i, l_i)*1e6:.2f} μeV
    """)

with st.expander("Lamb shift"):
    st.markdown("""
    The **Lamb shift** is a QED correction that lifts the degeneracy between
    the 2s₁/₂ and 2p₁/₂ states, which are degenerate in the Dirac equation.

    ΔE_Lamb ≈ 1057.8 MHz ≈ 4.37 × 10⁻⁶ eV

    It arises from:
    - **Vacuum fluctuations** — the electron interacts with virtual photons
    - **Vacuum polarisation** — virtual electron-positron pairs screen the nucleus

    Historically crucial — its measurement by Lamb and Retherford (1947)
    confirmed quantum electrodynamics as the correct theory of light-matter interaction.
    """)

with st.expander("Zeeman effect"):
    st.markdown(f"""
    In an external magnetic field **B**, degenerate mⱼ sublevels split:

    ΔE = mⱼ · g_J · μ_B · B

    where μ_B = 9.274 × 10⁻²⁴ J/T is the Bohr magneton.

    **Normal Zeeman** (implemented here) — assumes g_J = 1, valid when
    spin effects are negligible.

    **Anomalous Zeeman** — requires the full Landé g-factor:
    g_J = 1 + [j(j+1) + s(s+1) − l(l+1)] / [2j(j+1)]

    At B = {B_field:.1f} T, the level splitting is
    {zeeman_shift(1, B_field)*1e6:.2f} μeV per mⱼ unit.
    """)

with st.expander("Hyperfine structure"):
    st.markdown(f"""
    Hyperfine structure arises from the interaction between the **electron
    spin** and the **nuclear magnetic moment** of the proton.

    For the hydrogen ground state (1s), this produces two levels:
    - **F = 1** (parallel spins) — higher energy
    - **F = 0** (antiparallel spins) — lower energy

    Splitting: ΔE = 5.874 × 10⁻⁶ eV  →  **λ = 21.1 cm**

    The **21 cm line** is one of the most important in astrophysics —
    used to map neutral hydrogen throughout the galaxy and observe
    structure that optical telescopes cannot penetrate.

    The transition is **magnetic dipole (M1)** — forbidden by E1 rules,
    with a spontaneous emission lifetime of ~10 million years.
    """)
