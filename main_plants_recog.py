# main.py
import streamlit as st
from Exploration_page_1 import main as Exploration_page_1_main
from Data_Visualization_page_2 import main as Data_Visualization_page_2_main
from Modelling_page_3 import main as Modelling_page_3_main
from Interpreting_page_4 import main as Interpreting_page_4_main

st.title("Plant seeds classification project")
st.sidebar.title("Table of contents")
pages = ["Exploration", "DataVizualization", "Modelling", "Interpreting"]
page = st.sidebar.radio("Go to", pages)

if page == "Exploration":
    Exploration_page_1_main()
elif page == "DataVizualization":
    Data_Visualization_page_2_main()
elif page == "Modelling":
    Modelling_page_3_main()
elif page == "Interpreting":
    Interpreting_page_4_main()