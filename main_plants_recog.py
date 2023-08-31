import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from keras.models import load_model
import os
import pickle
import matplotlib.pyplot as plt
import itertools
import json
import keras
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import io
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import h5py
#from sklearn.metrics import confusion_matrix

st.title("Plant seeds classification project")
st.sidebar.title("Table of contents")
pages=["Exploration", "DataVizualization", "Modelling", "Interpreting"]
page=st.sidebar.radio("Go to", pages)

if page == pages[0] : 
    st.write("### Presentation of data")

if page == pages[1] : 
    st.write("### DataVizualization")

if page == pages[2] : 
    st.write("### Modelling")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    def prediction(classifier):
        route = "Saved_Models"
        if classifier == 'Random Forest':
            loaded_cm = np.load(os.path.join(route, 'matriz_confusion_rf_classifier.npy'))
            loaded_category_to_label = np.load(os.path.join(route, 'class_names.npy'))
            accuracy = np.trace(loaded_cm) / np.sum(loaded_cm)
            st.write("### Accuracy:", f"{accuracy:.2f}")
            show_confusion_matrix(loaded_cm, loaded_category_to_label, "Random Forest")
        elif classifier == 'SVM':
            loaded_cm = np.load(os.path.join(route, 'matriz_confusion_svm_classifier.npy'))
            loaded_category_to_label = np.load(os.path.join(route, 'class_names.npy'))
            accuracy = np.trace(loaded_cm) / np.sum(loaded_cm)
            st.write("### Accuracy:", f"{accuracy:.2f}")
            show_confusion_matrix(loaded_cm, loaded_category_to_label, "SVM")
        elif classifier == 'SVM-PCA':
            loaded_cm = np.load(os.path.join(route, 'matriz_confusion_svm_pca_classifier.npy'))
            loaded_category_to_label = np.load(os.path.join(route, 'class_names.npy'))
            accuracy = np.trace(loaded_cm) / np.sum(loaded_cm)
            st.write("### Accuracy:", f"{accuracy:.2f}")
            show_confusion_matrix(loaded_cm, loaded_category_to_label, "SVM-PCA")
        elif classifier == 'SVM-Balanced':
            loaded_cm = np.load(os.path.join(route, 'matriz_confusion_svm_balanced_classifier.npy'))
            loaded_category_to_label = np.load(os.path.join(route, 'class_names.npy'))
            accuracy = np.trace(loaded_cm) / np.sum(loaded_cm)
            st.write("### Accuracy:", f"{accuracy:.2f}")
            show_confusion_matrix(loaded_cm, loaded_category_to_label, "SVM Balanced")
        elif classifier == 'KNN':
            loaded_cm = np.load(os.path.join(route, 'matriz_confusion_knn_classifier.npy'))
            loaded_category_to_label = np.load(os.path.join(route, 'class_names.npy'))
            accuracy = np.trace(loaded_cm) / np.sum(loaded_cm)
            st.write("### Accuracy:", f"{accuracy:.2f}")
            show_confusion_matrix(loaded_cm, loaded_category_to_label, "KNN")
        elif classifier == 'KNN-PCA':
            loaded_cm = np.load(os.path.join(route, 'matriz_confusion_knn_pca_classifier.npy'))
            loaded_category_to_label = np.load(os.path.join(route, 'class_names.npy'))
            accuracy = np.trace(loaded_cm) / np.sum(loaded_cm)
            st.write("### Accuracy:", f"{accuracy:.2f}")
            show_confusion_matrix(loaded_cm, loaded_category_to_label, "KNN-PCA")
        elif classifier == 'KNN':
            loaded_cm = np.load(os.path.join(route, 'matriz_confusion_knn_classifier.npy'))
            loaded_category_to_label = np.load(os.path.join(route, 'class_names.npy'))
            accuracy = np.trace(loaded_cm) / np.sum(loaded_cm)
            st.write("### Accuracy:", f"{accuracy:.2f}")
            show_confusion_matrix(loaded_cm, loaded_category_to_label, "KNN")
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
            show_confusion_matrix(loaded_cm, classes, "LeNET")
        elif classifier == 'VGG16':
            def preproces_image(img_path):
                img = image.load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                return preprocess_input(x)
            # model was splitted to upload in github
            part_filenames = ['Saved_Models/model_part01', 'Saved_Models/model_part02', 'Saved_Models/model_part03', 'Saved_Models/model_part04']
            # Crear una lista para almacenar los contenidos de las partes
            part_contents = []
            # Leer cada parte y almacenar su contenido en la lista
            for part_filename in part_filenames:
                with open(part_filename, 'rb') as part_file:
                    part_contents.append(part_file.read())
            # Combinar las partes en un solo contenido
            full_file_data = b''.join(part_contents)
            # Abre el archivo h5 directamente desde el contenido en memoria
            with io.BytesIO(full_file_data) as in_memory_file:
                with h5py.File(in_memory_file, 'r') as h5_file:
                    # Cargar el modelo desde el archivo HDF5
                    model = load_model(h5_file)
            choice = ['Loose Silky-bent', 'Cleavers', 'Black-grass', 'Scentless Mayweed', 'Maize', 'Charlock', 'Sugar beet', 'Fat Hen', 'Small-flowered Cranesbill', 'Common wheat', 'Common Chickweed', 'Shepherd Purse']
            option = st.selectbox('Choice of the plant', choice)
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
            batch_size = 64
            test_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)
            test_generator = test_data_generator.flow_from_directory(directory="Sample_images",
                                                                    class_mode ="sparse",
                                                                    target_size = (224 , 224),
                                                                    batch_size = batch_size,
                                                                    shuffle=False)
            class_names = list(test_generator.class_indices.keys())
            #with open(os.path.join(route, 'category_to_label.json'), 'r') as json_file:
            #    loaded_category_to_label = json.load(json_file)
            #classes = list(loaded_category_to_label.keys())
            st.write('Prediction :', class_names[class_index])
        elif classifier == 'Fastai':
            st.write('Option not available')
        elif classifier == 'VGG16 + SVM':
            st.write('Option not available')

    def scores(clf, choice):
        if choice == 'Accuracy':
            st.write("### Accuracy")
        elif choice == 'Confusion matrix':
            st.write("### Confusion matrix")

    def show_plot_history_list(history_list):
        # Crear una figura y ejes para el gráfico
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

        history_acc = []
        history_val_acc = []
        history_loss = []
        history_val_loss = []

        # Recopilar las precisiones (accuracy) de todas las historias en history_list
        for history in history_list:
            history_acc.extend(history['accuracy'])
            history_val_acc.extend(history['val_accuracy'])
            history_loss.extend(history['loss'])
            history_val_loss.extend(history['val_loss'])

        # Graficar la precisión (accuracy) en función de las épocas
        axes[0].plot(history_acc, label='Train Accuracy')
        axes[0].plot(history_val_acc, label='Validation Accuracy')
        axes[0].set_title('Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()

        axes[1].plot(history_loss, label='Train Loss')
        axes[1].plot(history_val_loss, label='Validation Loss')
        axes[1].set_title('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()

        plt.tight_layout()

        # Utilizar Streamlit para mostrar la figura
        st.pyplot(fig)

    def show_accuracy_plot(model_history, epochs):
        # Crear una figura y ejes para el gráfico
        fig, ax = plt.subplots(figsize=(10, 6))

        # Labels de los ejes
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')

        # Courbe de la précision sur l'échantillon d'entrainement
        ax.plot(np.arange(1 , epochs + 1, 1),
                model_history.history['accuracy'],
                label = 'Training Accuracy',
                color = 'blue')

        # Courbe de la précision sur l'échantillon de test
        ax.plot(np.arange(1 , epochs + 1, 1),
                model_history.history['val_accuracy'], 
                label = 'Validation Accuracy',
                color = 'red')

        # Affichage de la légende
        ax.legend()

        # Utilizar Streamlit para mostrar la figura
        st.pyplot(fig)

    def show_confusion_matrix(cm, classes, title):
        # Normalize the confusion matrix to show percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        plt.figure()

        plt.imshow(cm_percent, interpolation='nearest', cmap='Blues')
        plt.title("Normalized Confusion Matrix for " + title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, f"{cm_percent[i, j]:.0f}",  # Display percentage value
                    horizontalalignment="center",
                    color="white" if cm_percent[i, j] > (cm_percent.max() / 2) else "black")

        plt.ylabel('True labels')
        plt.xlabel('Predicted Labels')

        # Utilizar Streamlit para mostrar la figura
        st.pyplot()


    choice = ['Random Forest', 'SVM', "SVM-PCA", "SVM-Balanced", 'KNN', "KNN-PCA", 'A simple CNN', 'LeNet', 'LeNet Balanced', 'VGG16', 'Fastai', 'VGG16 + SVM']
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is :', option)
    prediction(option)

if page == pages[3] : 
    st.write("### Interpreting")

    choice = ['Loose Silky-bent', 'Cleavers', 'Black-grass', 'Scentless Mayweed', 'Maize', 'Charlock', 'Sugar beet', 'Fat Hen', 'Small-flowered Cranesbill', 'Common wheat', 'Common Chickweed', 'Shepherd Purse']

    option = st.selectbox('Choice of the plant', choice)
    st.write('The chosen plant is :', option)

    # image path
    img_path = os.path.join("Sample_images", option, "image1.jpg")
    img_path_grad_cam = os.path.join("Grad_Cam_images", option, "grad_cam_image1.jpg")
    st.write('Path :', img_path)

    # Display original image
    img = image.load_img(img_path, target_size=(224, 224))
    plt.matshow(img)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    # Display Grad CAM image
    img = image.load_img(img_path_grad_cam, target_size=(224, 224))
    plt.matshow(img)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

