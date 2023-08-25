import streamlit as st
st.title("My second Streamlit")
st.write("Introduction")
if st.checkbox("Display"):
  st.write("Streamlit continuation")
