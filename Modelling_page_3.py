import streamlit as st
from Functions import predecir_imagen
from Functions import show_confusion_matrix
from Functions import show_confusion_matrix_from_data
from Functions import show_accuracy_loss_plot
from Functions import show_accuracy_loss_plot_fastai
from Functions import load_history_classes_cm
from Functions import select_plant_for_prediction
from Functions import load_fastai_model
from Functions import load_vgg16
from Functions import load_vgg16_svm
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from fastai.vision.all import *

st.set_page_config(layout="wide")

def main():

    models_intro = """
    **Models**

    In our project, we trained models to classify plants. The main performance metric we
    used to compare our models is accuracy. The reason for this choice is that accuracy
    provides an overall picture of how well the model is performing. We utilized accuracy
    to monitor the training progress across different epochs and even considered it for early
    stopping if we noticed that precision wasn't improving. Additionally, we observed the
    disparity between the accuracy of training and evaluation samples as an early detection
    method for overfitting, occasionally stopping the training process prematurely.

    However, in the final evaluation, we also observed both the precision and the recall,
    which are useful for assessing the model's performance in detecting specific classes. By
    considering both precision and recall, we gained deeper insights into the model's
    strengths and weaknesses in handling different classes.

    Additionally, we used confusion matrices to understand which plants are not being
    properly detected by our models. In fact, confusion matrices provide a direct view of
    both precision and recall.
    """

    ml_models = """
    **Machine Learning Models**

    Initially, simple machine learning models were used, all with bad results. We used a
    Support Vector Machine with 0.48 of accuracy, a Random Forest Classifier with 0.50 of
    accuracy, and a K-Nearest Neighbors classifier with 0.26 of accuracy.
    """

    dl_models = """
    **Deep Learning Models**

    Deep learning models are sophisticated machine learning models rooted in the concept
    of neural networks. Neural networks consist of interconnected nodes called
    "perceptrons." These networks are designed to learn patterns within data in order to
    classify information or predict values. Particularly, we employed Convolutional Neural
    Networks (CNNs). These types of neural networks can detect patterns in images and
    subsequently classify these images through supervised learning.
    """

    neural_network_structure = """
    **The Basic Structure of a Neural Network**

    Initially, we employed a basic CNN model with augmented data to avoid overfitting.
    Nevertheless, the accuracy score remained low, reaching only 0.56. 
    We then attempted to enhance performance by implementing the LeNet architecture. As
    we expected, the accuracy score increased to 0.77, meaning that this was the best model
    without using transfer learning.
    """

    transfer_learning = """
    **Transfer Learning**

    We employed transfer learning, a technique used to enhance the classification
    performance of our model. In transfer learning, a pre-trained model is used to improve
    the performance of our model. Such pre-trained models are originally trained on large
    datasets for image classification. Therefore, we can take advantage of this pre-training
    by using the pre-trained model as the basis for our model.

    We primarily experimented with two models: a VGG16 model and a ResNet34 model.
    In addition, we used feature extraction to combine the VGG16 model with a Support
    Vector Machine (SVC) for classification. Feature extraction is another technique in
    which the pre-trained model is combined with a machine learning model for
    classification.

    All transfer learning models were pre-trained on the ImageNet database. Data
    augmentation was applied in all cases to mitigate overfitting. However, only in the
    ResNet34 model, we used Fastai, a library that simplifies the process of creating and
    training deep learning models, provides pre-implemented architectures like ResNet34,
    and facilitates transfer learning with pre-trained models from ImageNet. Fastai's user-
    friendly interface and powerful functionalities allowed us to experiment more efficiently
    and achieve better results in our ResNet34-based model.
    """

    vgg16_info = """
    **VGG16**

    In the first attempt using VGG16 with ImageNet, the accuracy was very low. This was
    probably due to the way in which augmented data was created (i.e., using the .flow
    method) and the low number of epochs used to train the model (n = 5). When changing
    the way in which augmented data was created (i.e., using the .flow_from_directory
    method) and the number of epochs was raised from five to 10 (as well as the number of
    steps per epoch from 20 to 61), the accuracy rose from 0.25 to 0.82 in the training set.
    However, strangely, the validation accuracy was higher (0.88) than the training set
    (0.82) in this last model.

    We later discovered that data augmentation distorted the training set images to an extent
    that spoiled our model. In particular, using a zoom_range value of 1.1 was causing
    significant distortion, which resulted in the model underperforming when training with
    images from the dataset. However, adjusting this parameter to 0.4 resolved the issue.
    After this correction, the training set accuracy rose from 0.82 to 0.97, and the test set
    accuracy rose from 0.88 to 0.95.
    """

    feature_extraction = """
    **Feature Extraction (VGG16 and SVC)**

    When using feature extraction, we combined the already trained VGG16 model with a
    Support Vector Classification. We also used augmented data to avoid overfitting. The
    result was slightly better than the VGG16 model, as the accuracy for the test set rose
    from 0.9380 to 0.96. However, the model was confusing images of black grass plants
    with those of Loose Silky Bent. On one hand, it misclassified 34 out of 100 images of
    the black grass plant as Silky Bent. On the other hand, it also misclassified 10 out of
    101 images of the Silky Bent plant as Black Grass.
    """

    resnet34_info = """
    **ResNet34 Model**

    As we will see in this section, the ResNet34 Model we implemented has a high
    performance. Its accuracy score is 0.95, in other words, this model correctly classifies
    95% of the images. However, the confusion between black grass plants and Loose Silky
    Bent plants still persists. In this model, we used Fastai, which simplified model training
    and enabled easy transfer learning with pre-trained models from ImageNet.
    """

    def prediction(classifier):
        route = "Saved_Models"
        if classifier == 'Random Forest':
            st.write(ml_models)
            show_confusion_matrix('matriz_confusion_rf_classifier.npy', 'class_names.npy', "Random Forest")
        elif classifier == 'SVM':
            st.write(ml_models)
            show_confusion_matrix('matriz_confusion_svm_classifier.npy', 'class_names.npy', "SVM")
        elif classifier == 'KNN-PCA':
            st.write(ml_models)
            show_confusion_matrix('matriz_confusion_knn_pca_classifier.npy', 'class_names.npy', "KNN-PCA")
        elif classifier == 'A simple CNN':
            history_list, loaded_cm, classes = load_history_classes_cm(route, 'histories_simple_cnn_224x224.pkl', 'matriz_confusion_simple_cnn_224x224.npy', 'class_names_simple_cnn_224x224.npy')
            show_accuracy_loss_plot(history_list)
            show_confusion_matrix_from_data(loaded_cm, classes, "A simple CNN")
        elif classifier == 'LeNet':
            history_list, loaded_cm, classes = load_history_classes_cm(route, 'histories_lenet_200x200_balanced.pkl', 'matriz_confusion_lenet_balanced.npy', 'category_to_label.json')
            show_accuracy_loss_plot(history_list)
            show_confusion_matrix_from_data(loaded_cm, classes, "LeNET")
        elif classifier == 'VGG16':
            model = load_vgg16()
            img_path = select_plant_for_prediction()
            predecir_imagen(img_path, model, 'VGG16', "Saved_Models", 'class_names_vgg16.npy')
        elif classifier == 'ResNet34':
            learner_load = load_fastai_model()
            img_path = select_plant_for_prediction()
            predecir_imagen(img_path, learner_load, 'Resnet34', null, null)
        elif classifier == 'VGG16 + SVM':
            model = load_vgg16_svm()
            # Obtener la prediccion segun mi modelo
            img_path = select_plant_for_prediction()
            predecir_imagen(img_path, model, 'VGG16+SVM', "Saved_Models", 'class_names_VGG16+SVM.npy')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.title("Modelling")
    with col2:
        st.set_option('deprecation.showPyplotGlobalUse', False)
        choice = ['Random Forest', 'SVM', "KNN-PCA", 'A simple CNN', 'LeNet', 'VGG16', 'ResNet34', 'VGG16 + SVM']
        option = st.selectbox('Choice of the model', choice)

    route = "Saved_Models"
    col1, col2, col3 = st.columns(3)
    with col1:
        if (option in ['Random Forest', 'SVM', "KNN-PCA"]):
            st.write(models_intro)
        elif (option in ['A simple CNN', 'LeNet']):
            st.write(dl_models)
        elif (option in ['VGG16', 'ResNet34', 'VGG16 + SVM']):
            st.write(transfer_learning)
    with col2:
        if (option in ['Random Forest', 'SVM', "KNN-PCA"]):
            prediction(option)
        elif (option in ['A simple CNN', 'LeNet']):
            st.write(neural_network_structure)
            image = Image.open("neural_network.jpg")
            st.image(image, caption="The Basic Structure of a Neural Network", use_column_width=True)
        elif (option == 'VGG16'):
            st.write(vgg16_info)
            history_list, loaded_cm, classes = load_history_classes_cm(route, 'histories_VGG16_balanced_training_data.pkl', 'matriz_confusion_vgg16.npy', 'class_names_vgg16.npy')
            show_accuracy_loss_plot(history_list, accuracy = 'acc')
        elif (option == 'ResNet34'):
            st.write(resnet34_info)
            history_list, loaded_cm, classes = load_history_classes_cm(route, 'histories_fastai.pkl', 'matriz_confusion_fastai.npy', 'class_names_fastai.npy')
            show_accuracy_loss_plot_fastai(history_list)
        elif (option == 'VGG16 + SVM'):
            st.write(feature_extraction)
            show_confusion_matrix('matriz_confusion_VGG16+SVM.npy', 'class_names_VGG16+SVM.npy', "VGG16+SVM")
    with col3:
        if (option not in ['Random Forest', 'SVM', "KNN-PCA"]):
            prediction(option)
            if (option == 'VGG16'):
                show_confusion_matrix_from_data(loaded_cm, classes, "VGG16")
            if (option == 'ResNet34'):
                show_confusion_matrix_from_data(loaded_cm, classes, "ResNet34")

if __name__ == "__main__":
    main()