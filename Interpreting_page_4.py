import streamlit as st
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

def main():
    st.title("Interpreting")
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

    # TODO show_confused_images
    
if __name__ == "__main__":
    main()