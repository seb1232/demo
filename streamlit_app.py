import streamlit as st
pg=st.navigation([st.Page("SprintTaskPlanner.py"),st.Page("RetrospectiveAnalysisTool.py")])
st.set_page_config(page_title="Data manager", page_icon=":material/edit:")
pg.run()
