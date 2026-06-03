import streamlit as st

st.set_page_config(page_title="Hydrogen Atom Visualiser", layout="wide")

st.title("Hydrogen atom visualiser")
st.markdown("""
Use the sidebar to explore:
- **Radial plots** — probability density vs distance from nucleus
- **Orbital heatmaps** — 2D cross-section of |ψ|²
""")
