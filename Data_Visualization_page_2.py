import streamlit as st
from Functions import load_and_show_images

def main():
    st.title("DataVizualization")
    st.write("### DataVizualization")
    load_and_show_images("Test_original", (200, 200))

if __name__ == "__main__":
    main()