import streamlit as st

# Set the app-wide configuration
st.set_page_config(
    page_title="Sprint Management Suite",
    page_icon="🚀",
    layout="wide",
)

# Create a nice sidebar navigation
with st.sidebar:
    st.title("📚 Navigation")
    st.markdown("Select a tool to continue:")

# Define your pages
pg = st.navigation([
    st.Page("SprintTaskPlanner.py", title="🗂 Sprint Task Planner"),
    st.Page("RetrospectiveAnalysisTool.py", title="📊 Retrospective Analysis"),
])

# Run the selected page
pg.run()
