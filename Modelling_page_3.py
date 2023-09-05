import streamlit as st
from Functions import predecir_imagen
from Functions import show_confusion_matrix
from Functions import show_confusion_matrix_from_data
from Functions import show_accuracy_loss_plot
from Functions import load_history_classes_cm
from Functions import select_plant_for_prediction
from Functions import load_fastai_model
from Functions import load_vgg16
from Functions import load_vgg16_svm
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from fastai.vision.all import *

def main():
    st.title("Modelling")
    st.write("### Modelling")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    def prediction(classifier):
        route = "Saved_Models"
        if classifier == 'Random Forest':
            show_confusion_matrix('matriz_confusion_rf_classifier.npy', 'class_names.npy', "Random Forest")
        elif classifier == 'SVM':
            show_confusion_matrix('matriz_confusion_svm_classifier.npy', 'class_names.npy', "SVM")
        elif classifier == 'KNN-PCA':
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
            # TODO show history
            model = load_vgg16()
            img_path = select_plant_for_prediction()
            predecir_imagen(img_path, model, 'VGG16', "Saved_Models", 'class_names_vgg16.npy')
            show_confusion_matrix('matriz_confusion_vgg16.npy', 'class_names_vgg16.npy', "VGG16")
        elif classifier == 'Fastai':
            # TODO show history
            learner_load = load_fastai_model()
            img_path = select_plant_for_prediction()
            predecir_imagen(img_path, learner_load, 'Resnet34', null, null)
            show_confusion_matrix('matriz_confusion_fastai.npy', 'class_names_fastai.npy', "Fastai")
        elif classifier == 'VGG16 + SVM':
            model = load_vgg16_svm()
            # Obtener la prediccion segun mi modelo
            img_path = select_plant_for_prediction()
            predecir_imagen(img_path, model, 'VGG16+SVM', "Saved_Models", 'class_names_VGG16+SVM.npy')
            show_confusion_matrix('matriz_confusion_VGG16+SVM.npy', 'class_names_VGG16+SVM.npy', "VGG16+SVM")

    choice = ['Random Forest', 'SVM', "KNN-PCA", 'A simple CNN', 'LeNet', 'VGG16', 'Fastai', 'VGG16 + SVM']
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is :', option)
    prediction(option)

if __name__ == "__main__":
    main()