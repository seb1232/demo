import streamlit as st
st.title("THE ULTIMATE RETROTASK")
pg=st.navigation([st.Page("SprintTaskPlanner.py"),st.Page("RetrospectiveAnalysisTool.py")])
pg.run()
