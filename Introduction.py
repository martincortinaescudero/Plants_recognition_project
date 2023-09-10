import streamlit as st

def main():
    st.title("Introduction")

    introduction = """
    Computer vision for image classification has been improving in the recent decades.
    This promising field offers new possibilities for technological development. In this
    context, our project aims to build a Deep Learning Model that can classify plants to
    improve agricultural processes.

    **Goal:** build a model that can classify plant seedlings correctly.

    **Practical applications:**
    - Automation of weed extractions using robots.
    - Identification of plants for trading purposes.
    - Scientific research on plants (automation of processes).
    """

    dataset_info = """
    **Dataset: V2 Plant Seedlings Dataset**
    - Twelve types of plant seedlings at different growth stages common in the
    Danish agriculture.
    - This dataset contains 5,539 images of various size and in PNG format.
    - Ownership of dataset: Computer Vision and Signal Processing Group at Aarhus
    University (it is available on Kaggle).
    """

    operationalization_info = """
    **Operationalization**
    - Feature variables: Images of plant seedlings.
    - Target variable: Labels (names) of the plants.
    """

    st.write(introduction)
    st.write(dataset_info)
    st.write(operationalization_info)

if __name__ == "__main__":
    main()