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

    def prediction(classifier):
        route = "Saved_Models"
        if classifier == 'Random Forest':
            st.write('Option not available')
        elif classifier == 'SVM':
            st.write('Option not available')
        elif classifier == 'KNN':
            st.write('Option not available')
        elif classifier == 'A simple CNN':
            path_histories = os.path.join(route, 'histories_simple_cnn_224x224.pkl')
            # Cargar la lista de historias
            with open(path_histories, 'rb') as file:
                history_list = pickle.load(file) # in CNN a history list was made
            show_plot_history_list(history_list)
        elif classifier == 'LeNet':
            #path_histories = os.path.join(route, 'histories_lenet_200x200.pkl')
            # Cargar la lista de historias
            #with open(path_histories, 'rb') as file:
            #    model_history = pickle.load(file) # in LeNet a single history was made
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
            st.set_option('deprecation.showPyplotGlobalUse', False)
            show_confusion_matrix(loaded_cm, classes)
        elif classifier == 'VGG16':
            st.write('Option not available')
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

    def show_confusion_matrix(cm, classes):
        # Normalize the confusion matrix to show percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        plt.figure()

        plt.imshow(cm_percent, interpolation='nearest', cmap='Blues')
        plt.title("Confusion Matrix")
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


    choice = ['Random Forest', 'SVM', 'KNN', 'A simple CNN', 'LeNet', 'LeNet Balanced', 'VGG16', 'Fastai', 'VGG16 + SVM']
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is :', option)
    prediction(option)

if page == pages[3] : 
    st.write("### Interpreting")

    ##############################################
    # Definir una función para Grad-CAM:
    ##############################################

    def grad_cam(model, image, layer_name):
        last_conv_layer = model.get_layer(layer_name)
        gradient_model = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
       
        with tf.GradientTape() as tape:
            preds, conv_output = gradient_model(image)
            class_output = preds

        grads = tape.gradient(class_output, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        #heatmap = tf.reduce_mean(conv_output * pooled_grads[..., tf.newaxis], axis=-1)
        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        ########

        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.reduce_max(heatmap)

        return heatmap.numpy()

    ##############################################
    # Definir una función resultado superpuesto:
    ##############################################

    def display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
        # Load the original image
        img = keras.utils.load_img(img_path)
        img = keras.utils.img_to_array(img)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.utils.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = keras.utils.array_to_img(superimposed_img)

        # Display Grad CAM
        #display(Image(cam_path))
        # Display Grad CAM
        st.image(superimposed_img, caption='Grad CAM', use_column_width=True)

    ##############################################
    # Definir funcion para Preprocesar la imagen
    ##############################################

    def preproces_image(img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return preprocess_input(x)

    choice = ['Loose Silky-bent', 'Cleavers', 'Black-grass', 'Scentless Mayweed', 'Maize', 'Charlock', 'Sugar beet', 'Fat Hen', 'Small-flowered Cranesbill', 'Common wheat', 'Common Chickweed', 'Shepherd Purse']

    option = st.selectbox('Choice of the plant', choice)
    st.write('The chosen plant is :', option)

    # Preprocesar la imagen
    img_path = os.path.join("Sample_images", option, "image1.jpg")
    st.write('Path :', img_path)
    x = preproces_image(img_path)

    # Cargar el modelo
    #model = load_model('Saved_Models/model_resize_15-15.h5')

    # Cargar la lista de historias
    #with open('Saved_Models/histories_resize_15-15.pkl', 'rb') as file:
    #    history_list = pickle.load(file)

    # Obtener la prediccion segun mi modelo
    #class_index = np.argmax(model.predict(x))

    # Obtener la ultima capa convolucional de mi modelo
    #base_model = model.get_layer('vgg16')
    #layer_name = 'block5_conv3'
        
    # Obtener la interpretacion Grad-CAM
    #heatmap = grad_cam(base_model, x, layer_name)

    # Display original image
    img = image.load_img(img_path, target_size=(224, 224))
    plt.matshow(img)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()



    # Display heatmap
    #plt.matshow(heatmap)
    #st.pyplot()

    # Save and diplay superimposed heatmap
    #display_gradcam(img_path, heatmap)
