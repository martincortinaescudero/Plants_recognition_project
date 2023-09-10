import streamlit as st
import pandas as pd
import os
from Functions import show_stats_plots

def main():
    figure1 = """
    **Figure 1**
    represents the number of images by class. This figure shows an unbalanced
    dataset. Some classes, such as Loose Silky-bent, are well-represented with over 700 images,
    while others like Maize have fewer than 300 images. There is a wide variation in the
    number of images among classes, which should be considered during the modeling
    process.
    """

    figure2 = """
    **Figure 2**
    shows the maximum image sizes by class. The size of the images was calculated
    by multiplying the height by the width in pixels. The maximum image size spans over 10 megapixels,
    representing a significant variation in image sizes among classes.
    This variation poses challenges in terms of data analysis and normalization,
    as the data may need to be rescaled or transformed to ensure consistent processing and modeling.
    """

    figure3 = """
    **Figure 3**
    displays the minimum image sizes by class. Similarly, the size of the images was calculated by
    multiplying the height by the width in pixels. The minimum image size is around 2000 pixels,
    indicating a variation of more than 3 orders of magnitude in image sizes among classes.
    This substantial variation in image sizes also presents challenges in data analysis and normalization,
    highlighting the need for careful preprocessing in the modeling process.
    """

    figure4 = """
    **Figure 4**
    shows the average image size for all classes. This figure shows that the average
    image size varies greatly from one class to another. However, we can observe the
    similarity between the average height and width within each class, suggesting that the
    images, on average, tend to be square.
    """

    figure5 = """
    **Figure 5**
    displays the three RGB average channel values for each class. It is worth
    noting that while plants are predominantly green, we might have expected the dominant
    channel to be green. However, our analysis reveals that the average channel values
    indicate that the dominant channel is blue. This suggests that the presence of non-plant
    elements in the images can affect this and make our classification task more
    challenging.
    """

    route = "Saved_Models"
    df_statistics = pd.read_csv(os.path.join(route, "statistics.csv"))
    st.title("Exploration")

    col1, col2, col3 = st.columns(3)
    with col1:
        show_stats_plots(df_statistics, "image_counts")
        st.write(figure1)
    with col2:
        show_stats_plots(df_statistics, "max_size")
        st.write(figure2)
    with col3:
        show_stats_plots(df_statistics, "min_size")
        st.write(figure3)
    col1, col2, col3 = st.columns(3)
    with col1:
        show_stats_plots(df_statistics, "average_sizes")
        st.write(figure4)
    with col2:
        show_stats_plots(df_statistics, "rgb_histogram")
        st.write(figure5)
    #with col3:
        #show_stats_plots(df_statistics, "min_size")

if __name__ == "__main__":
    main()