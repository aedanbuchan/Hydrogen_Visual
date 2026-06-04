import streamlit as st

st.set_page_config(
    page_title="Hydrogen Atom Visualiser",
    page_icon="⚛️",
    layout="wide"
)

st.title("⚛️ Hydrogen atom visualiser")
st.caption("An interactive tool for exploring hydrogen wavefunctions and energy transitions.")

st.markdown("""
Use the sidebar to explore:

- **Radial plots** — probability density P(r) vs distance from the nucleus
- **Orbital heatmaps** — 2D cross-section of |ψ|² on the x-z plane
- **Energy transitions** — drag between levels and view emission wavelengths
- **3D orbital viewer** — interactive isosurface and volume cloud of |ψ|²
""")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Quick reference")
    st.markdown("""
    | Symbol | Meaning |
    |---|---|
    | n | Principal quantum number |
    | l | Angular momentum |
    | m | Magnetic quantum number |
    | a₀ | Bohr radius = 0.529 Å |
    """)

with col2:
    st.subheader("Valid quantum numbers")
    st.markdown("""
    - n = 1, 2, 3, ...
    - l = 0, 1, ... , n−1
    - m = −l, ... , 0, ... , +l
    """)

with col3:
    st.subheader("Energy levels")
    st.markdown("""
    Eₙ = −13.6 / n²  eV

    | n | Energy |
    |---|---|
    | 1 | −13.60 eV |
    | 2 | −3.40 eV |
    | 3 | −1.51 eV |
    | 4 | −0.85 eV |
    """)
