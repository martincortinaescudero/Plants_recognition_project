import numpy as np
import streamlit as st
from Functions import load_vgg16_svm
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import random
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

def main():
    st.title("Interpreting")
    st.write("### Interpreting")
    choice = ['Loose Silky-bent', 'Cleavers', 'Black-grass', 'Scentless Mayweed', 'Maize', 'Charlock', 'Sugar beet', 'Fat Hen', 'Small-flowered Cranesbill', 'Common wheat', 'Common Chickweed', 'Shepherd Purse']
    option = st.selectbox('Choice of the plant', choice)
    st.write(option)
    # image path
    img_path = os.path.join("Sample_images", option, "image1.jpg")
    img_path_grad_cam = os.path.join("Grad_Cam_images", option, "grad_cam_image1.jpg")

    col1, col2 = st.columns(2)
    with col1:
        img = image.load_img(img_path, target_size=(224, 224))
        plt.matshow(img)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    with col2:
        img = image.load_img(img_path_grad_cam, target_size=(224, 224))
        plt.matshow(img)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    def select_plant_for_prediction(predictions, X, Y, X_test, Y_test, option, class_names):
        st.write('The chosen plant is:', option)
        # Get the indices of misclassified images
        misclassified_indices = [i for i in range(len(Y)) if predictions[i] != Y[i]]
        # Get a random misclassified image index
        if misclassified_indices:
            # Get the misclassified images
            misclassified_images = X[misclassified_indices]
            idx = predictions[misclassified_indices] - 1
            misclassified_labels = [class_names[i] for i in idx]
            images_from_confused_classes = get_images(idx, X_test, Y_test)
            for i in range(len(misclassified_indices)):
                col1, col2 = st.columns(2)
                with col1:
                    st.write('Misclassified Image')
                    st.image(misclassified_images[i], use_column_width=False)
                with col2:
                    st.write(f'Confused with {misclassified_labels[i]}')
                    st.image(images_from_confused_classes[i], use_column_width=False)
        else:
            st.write('No misclassified images for the chosen plant.')

    def get_images(misclassified_labels, X_test, Y_test):
        images_from_confused_classes = []
        Y_test = Y_test - 1
        for label in misclassified_labels:
            # Buscar todas las imágenes en X_test que tienen la misma etiqueta que label
            indices = [i for i, y in enumerate(Y_test) if y == label]
            # Tomar una imagen aleatoria de las encontradas
            random_index = random.choice(indices)
            confused_image = X_test[random_index]
            images_from_confused_classes.append(confused_image)

        return images_from_confused_classes

    def load_data(path, class_names):
        img_list = []
        label_list = []
        for i in range(12):
            PATH = os.path.join(path,class_names[i])+'/'
            for filename in os.listdir(PATH):
                img=cv2.imread(os.path.join(PATH,filename))
                # Resize image
                img=cv2.resize(img,(224,224))
                # for the black and white image
                if img.shape==(224, 224):
                    img=img.reshape([224,224,1])
                    img=np.concatenate([img,img,img],axis=2)
                # cv2 load the image BGR sequence color (not RGB)
                img_list.append(img[...,::-1])
                label_list.append(i+1)
        return np.array(img_list), np.array(label_list)

    batch_size = 64
    test_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)
    test_generator = test_data_generator.flow_from_directory(directory="Test_original_non_balanced",
                                                            class_mode ="sparse",
                                                            target_size = (224 , 224),
                                                            batch_size = batch_size,
                                                            shuffle=False)
    class_names = list(test_generator.class_indices.keys())
    intermediate_layer_model, svm = load_vgg16_svm()
    X_test, Y_test = load_data('Test_original_non_balanced', class_names)
    index_class = class_names.index(option) + 1
    X = X_test[Y_test == index_class]
    Y = Y_test[Y_test == index_class]
    X_test_features = intermediate_layer_model.predict(preprocess_input(X))
    predictions = svm.predict(X_test_features)
    select_plant_for_prediction(predictions, X, Y, X_test, Y_test, option, class_names)

if __name__ == "__main__":
    main()