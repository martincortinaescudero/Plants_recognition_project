import streamlit as st
import pandas as pd
import os
from Functions import show_stats_plots

def main():
    st.title("Exploration")
    st.write("### Presentation of data")
    route = "Saved_Models"
    df_statistics = pd.read_csv(os.path.join(route, "statistics.csv"))
    show_stats_plots(df_statistics, "image_counts")
    show_stats_plots(df_statistics, "max_size")
    show_stats_plots(df_statistics, "min_size")
    show_stats_plots(df_statistics, "average_sizes")
    show_stats_plots(df_statistics, "rgb_histogram")

if __name__ == "__main__":
    main()