import streamlit as st

def main():

    st.title("Conclusion")

    # Summary
    summary_text = """
    **Summary**
    This study aimed to classify twelve types of plants. We began by employing simple
    machine learning models, but all yielded unsatisfactory results: the Support Vector
    Machine achieved an accuracy of 0.48, the Random Forest Classifier scored an accuracy
    of 0.50, and the K-Nearest Neighbors classifier registered a mere accuracy of 0.26.
    Thereafter, we used neural networks to improve performance. Initially, we employed a
    basic CNN model, which achieved an accuracy of only 0.56. We then implemented a
    LeNet model that led to improved results: an accuracy of 0.77.
    However, the most effective models emerged when using Transfer Learning. The
    VGG16 model demonstrated strong performance, achieving an accuracy of 0.95. The
    model using Feature Extraction (which combines the VGG16 model with a SVC
    algorithm for classification) further enhanced the accuracy to 0.96. Finally, the
    ResNet34 model, implemented using fastai, showcased an also high performance by
    attaining an accuracy score of 0.95.
    Nevertheless, the best models confused black grass plants and Loose Silky Bent plants.
    Even in our best model, the feature extraction model (VGG16+SVC), this problem
    persisted. This could be due to the similarity of these two plants.
    """



    # Contribution
    contribution_text = """
    **Contribution**
    The main contribution to reach the goals of the project was the creation of
    Convolutional Neural Networks to classify the images in the Plant Seedling dataset. The
    best performing models were those using Transfer Learning. Among these models, the
    best one was the feature extraction model (combining VGG16 and SVC) which attained
    an impressive accuracy score of 0.96. However, our best model still confused some of
    the images, mainly from two types of plants (i.e. the Black Grass plants and Loose Silky
    Bent plants).
    This project contributed to scientific knowledge by exploring different improvements to
    convolutional neural networks in order to achieve outstanding classification
    performance. The incremental improvements show that combining different
    optimization techniques can lead to outstanding model performance. In this sense, our
    main contribution to scientific knowledge is that optimization techniques and fine-
    tuning is of great importance for the whole process of classification.
    """

    st.write(summary_text)
    st.write(contribution_text)

if __name__ == "__main__":
    main()