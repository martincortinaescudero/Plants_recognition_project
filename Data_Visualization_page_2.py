import streamlit as st
from Functions import load_and_show_images

def main():
    st.title("Data Visualization")

    page2_intro = """
    **On the following pages**, we present a random sample of 5 images for each class,
    providing a closer look at the dataset.
    """

    limitations = """
    **Limitations of the Data Set:**
    - Image size variability: image size varies from one class to another, making it
    difficult to classify plants.
    - Presence of non-plant elements in images: the surrounding soil, with
    heterogeneity of colors, can be an obstacle for the proper classification of plants.
    - Unbalanced classes: the number of images per plant varies, which can create a
    bias.
    """

    preprocessing = """
    **Pre-processing and Feature Engineering:**
    - Resizing images: this is done to have a common size format.
    - Reshaping images: this is done to have a common square format.
    - Image Masking: this is done to remove non-green pixels (did not work).
    - Cropping: this is done to enhance the recognition of distinct leaf patterns by
    preserving the sharpness of the image.
    - Data Augmentation: this is done to avoid overfitting.
    """

    st.write(page2_intro)
    st.write(limitations)
    st.write(preprocessing)

    load_and_show_images("Test_original", (200, 200))

if __name__ == "__main__":
    main()