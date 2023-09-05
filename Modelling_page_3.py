import streamlit as st
from Functions import preproces_image
from Functions import predecir_imagen
from Functions import show_confusion_matrix
from Functions import show_confusion_matrix_from_data
from Functions import show_accuracy_loss_plot
from Functions import load_history_classes_cm
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from fastai.vision.all import *
#import pathlib
#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath

def main():
    st.title("Modelling")
    st.write("### Modelling")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    def prediction(classifier):
        route = "Saved_Models"
        choice_plant = ['Loose Silky-bent', 'Cleavers', 'Black-grass', 'Scentless Mayweed', 'Maize', 'Charlock', 'Sugar beet', 'Fat Hen', 'Small-flowered Cranesbill', 'Common wheat', 'Common Chickweed', 'Shepherd Purse']
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
            file_model_vgg16 = '../GITHUB_LF/model_vgg16.h5'
            model = load_model(file_model_vgg16, compile=False)
            option = st.selectbox('Choice of the plant', choice_plant)
            st.write('The chosen plant is :', option)
            # image path
            img_path = os.path.join("Sample_images", option, "image1.jpg")
            # Display original image
            img = image.load_img(img_path, target_size=(224, 224))
            plt.matshow(img)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            x = preproces_image(img_path)
            # Obtener la prediccion segun mi modelo
            class_index = np.argmax(model.predict(x))
            loaded_category_to_label = np.load(os.path.join(route, 'class_names_vgg16.npy'))
            st.write('Prediction :', loaded_category_to_label[class_index])
            show_confusion_matrix('matriz_confusion_vgg16.npy', 'class_names_vgg16.npy', "VGG16")
        elif classifier == 'Fastai':
            file_model_fastai = '../GITHUB_LF/model_fastai.pkl'
            learner_load = load_learner(file_model_fastai)
            option = st.selectbox('Choice of the plant', choice_plant)
            st.write('The chosen plant is :', option)
            img_path = os.path.join("Test_original", option, "image1.jpg")
            # Display original image
            img = image.load_img(img_path, target_size=(224, 224))
            plt.matshow(img)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            predecir_imagen(img_path, learner_load)
            show_confusion_matrix('matriz_confusion_fastai.npy', 'class_names_fastai.npy', "Fastai")
        elif classifier == 'VGG16 + SVM':
            file_model_vgg16_svm_intermediate_layer = '../GITHUB_LF/intermediate_layer_model.h5'
            file_model_vgg16_svm = '../GITHUB_LF/vgg16+svm_classifier.pkl'
            model_vgg16_svm_intermediate_layer = load_model(file_model_vgg16_svm_intermediate_layer, compile=False)
            model_vgg16_svm = joblib.load(file_model_vgg16_svm)
            option = st.selectbox('Choice of the plant', choice_plant)
            st.write('The chosen plant is :', option)
            # image path
            img_path = os.path.join("Sample_images", option, "image1.jpg")
            # Display original image
            img = image.load_img(img_path, target_size=(224, 224))
            plt.matshow(img)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            # Obtener la prediccion segun mi modelo
            features_of_image = model_vgg16_svm_intermediate_layer.predict(preproces_image(img_path))
            prediction = model_vgg16_svm.predict(features_of_image)
            loaded_category_to_label = np.load(os.path.join(route, 'class_names_VGG16+SVM.npy'))
            st.write('Prediction :', loaded_category_to_label[prediction-1])
            show_confusion_matrix('matriz_confusion_VGG16+SVM.npy', 'class_names_VGG16+SVM.npy', "VGG16+SVM")

    choice = ['Random Forest', 'SVM', "KNN-PCA", 'A simple CNN', 'LeNet', 'VGG16', 'Fastai', 'VGG16 + SVM']
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is :', option)
    prediction(option)

if __name__ == "__main__":
    main()