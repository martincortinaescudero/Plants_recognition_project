import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import itertools
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input#, decode_predictions
from fastai.vision.all import *
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import random
from PIL import Image
#import cv2

def preproces_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

# Define una función para cargar una imagen y hacer una predicción
def predecir_imagen(ruta_de_la_imagen, learner):
    # Cargar la imagen
    img = PILImage.create(ruta_de_la_imagen)  # Utiliza PILImage.create en lugar de open_image
    # Obtener la predicción
    pred_class, pred_idx, outputs = learner.predict(img)
    # Imprimir la clase predicha y las probabilidades de cada clase
    st.write('Prediction :', pred_class)

def show_plot_history_list(history_list):
    # Crear una figura y ejes para el gráfico
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

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

def show_confusion_matrix(matrix_file, class_names_file, title):
    route = "Saved_Models"
    loaded_cm = np.load(os.path.join(route, matrix_file))
    loaded_category_to_label = np.load(os.path.join(route, class_names_file))
    accuracy = np.trace(loaded_cm) / np.sum(loaded_cm)
    st.write("### Accuracy:", f"{accuracy:.2f}")
    # Normalize the confusion matrix to show percentages
    cm_percent = loaded_cm.astype('float') / loaded_cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure()
    plt.imshow(cm_percent, interpolation='nearest', cmap='Blues')
    plt.title("Normalized Confusion Matrix for " + title)
    plt.colorbar()
    tick_marks = np.arange(len(loaded_category_to_label))
    plt.xticks(tick_marks, loaded_category_to_label, rotation=90)
    plt.yticks(tick_marks, loaded_category_to_label)
    for i, j in itertools.product(range(loaded_cm.shape[0]), range(loaded_cm.shape[1])):
        plt.text(j, i, f"{cm_percent[i, j]:.0f}",  # Display percentage value
                horizontalalignment="center",
                color="white" if cm_percent[i, j] > (cm_percent.max() / 2) else "black")
    plt.ylabel('True labels')
    plt.xlabel('Predicted Labels')
    # Utilizar Streamlit para mostrar la figura
    st.pyplot()

def show_confusion_matrix2(cm, classes, title):
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

def show_stats_plots(df, plot_type):
    plot_functions = {
        "image_counts": ("Number of Images per Class", df['image_count']),
        "max_size": ("Maximum Image Sizes (pixels)", df['max_size']),
        "min_size": ("Minimum Image Sizes (pixels)", df['min_size']),
        "average_sizes": ("Average Image Sizes", (df['avg_height'], df['avg_width'])),
        #"RGB": ("Histogram of RGB Channels", (df['R'], df['G'], df['B'])),
        "rgb_histogram": ("Histogram of RGB Channels", df[['B', 'G', 'R']])
    }
    if plot_type in plot_functions:
        title, data = plot_functions[plot_type]
        fig, ax = plt.subplots(figsize=(8, 6))
        if isinstance(data, tuple):
            indices = range(len(df))
            width = 0.35
            ax.bar(indices, data[0], width, label='Average Height')
            ax.bar([i + width for i in indices], data[1], width, label='Average Width')
            ax.set_xticks(indices)
            ax.set_xticklabels(df['subdirectory'], rotation=90)
            ax.legend()
        elif isinstance(data, pd.DataFrame):
            directory_names = df['subdirectory']
            bar_positions = np.arange(len(directory_names))
            bar_width = 0.2
            colors = ['blue', 'green', 'red']
            for i, color in enumerate(colors):
                ax.bar(bar_positions - i * bar_width, data.iloc[:, i], width=bar_width, color=color, label=data.columns[i])
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(directory_names, rotation=90)
            ax.legend()
        else:
            ax.bar(df['subdirectory'], data)
            ax.set_xticklabels(df['subdirectory'], rotation=90)
        ax.set_xlabel('Class')
        ax.set_ylabel('Size')
        ax.set_title(title)
        st.pyplot(fig)
    else:
        st.write("Plot type not recognized. Please choose from 'image_counts', 'max_size', 'min_size', 'average_sizes' or rgb_histogram.")

# Función para cargar imágenes y mostrar un conjunto de 5 imágenes aleatorias por directorio en Streamlit
def load_and_show_images(path, size):
    # Iterate over the images in the directory
    for dirname, _, filenames in os.walk(path):
        # Extract the subdirectory name
        subdirectory_name = os.path.basename(dirname)
        # List to store images for the current directory
        directory_images = []
        # Randomly shuffle the filenames to get random images
        random.shuffle(filenames)
        for filename in filenames:
            image_path = os.path.join(dirname, filename)
            # Open and resize the image using PIL
            with Image.open(image_path) as img:
                img = img.resize(size)
                img = img.convert("RGB")
                image = plt.imread(image_path)  # Convert to numpy array for matplotlib
                directory_images.append(image)
            # If there are 5 images in the list, show them and clear the list
            if len(directory_images) == 5:
                show_image_mosaic(subdirectory_name, directory_images)
                directory_images.clear()
                # Break the loop to only show 5 images per directory
                break

# Función para mostrar el mosaico de imágenes
def show_image_mosaic(subdirectory_name, images):
    num_rows = 1
    num_cols = len(images)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))
    fig.tight_layout()
    for j, image in enumerate(images):
        ax = axes[j] if num_cols > 1 else axes
        ax.imshow(image, extent=[0, 1, 0, 1])  # Ajusta el tamaño de la imagen
        ax.axis('off')
    # Ajusta el tamaño de la figura
    fig.subplots_adjust(wspace=0.05)
    st.write(f"{subdirectory_name}")
    st.pyplot(fig)
