import streamlit as st
from Functions import preproces_image
from Functions import predecir_imagen
from Functions import show_plot_history_list
from Functions import show_confusion_matrix
from Functions import show_accuracy_plot
#
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import pickle
import matplotlib.pyplot as plt
import json
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from fastai.vision.all import *
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

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
            path_histories = os.path.join(route, 'histories_simple_cnn_224x224.pkl')
            # Cargar la lista de historias
            with open(path_histories, 'rb') as file:
                history_list = pickle.load(file) # in CNN a history list was made
            show_plot_history_list(history_list)
        elif classifier == 'LeNet':
            path_histories = os.path.join(route, 'histories_lenet_200x200.pkl')
            # Cargar la lista de historias
            with open(path_histories, 'rb') as file:
                model_history = pickle.load(file) # in LeNet a single history was made
            #show_accuracy_plot(model_history, 30)
            st.write('Testing')
        elif classifier == 'LeNet Balanced':
            path_histories = os.path.join(route, 'histories_lenet_200x200_balanced.pkl')
            # Cargar la lista de historias
            with open(path_histories, 'rb') as file:
                model_history = pickle.load(file) # in LeNet a single history was made
            loaded_cm = np.load(os.path.join(route, 'matriz_confusion.npy'))
            with open(os.path.join(route, 'category_to_label.json'), 'r') as json_file:
                loaded_category_to_label = json.load(json_file)
            show_accuracy_plot(model_history, 30)
            #classes
            classes = list(loaded_category_to_label.keys())
            #show_confusion_matrix(loaded_cm, classes, "LeNET")
            show_confusion_matrix('matriz_confusion.npy', 'category_to_label.json', "LeNET")
        elif classifier == 'VGG16':
            file_model_vgg16 = '../GITHUB_LF/model_vgg16.h5'
            from tensorflow.keras.models import load_model
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
            from tensorflow.keras.models import load_model
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

    choice = ['Random Forest', 'SVM', "KNN-PCA", 'A simple CNN', 'LeNet', 'LeNet Balanced', 'VGG16', 'Fastai', 'VGG16 + SVM']
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is :', option)
    prediction(option)

if __name__ == "__main__":
    main()