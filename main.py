import streamlit as st
st.title("My second Streamlit")
st.write("Introduction")
if st.checkbox("Display"):
  st.write("Streamlit continuation")
  ######################################################
  # load my model and history
  ######################################################
  import pickle
  import numpy as np
  import keras
  import matplotlib.pyplot as plt
  from keras.models import load_model

  
  # Cargar el modelo
  #model = load_model('model_resize_15-15.h5')

  # Cargar la lista de historias
  with open('histories_resize_15-15.pkl', 'rb') as file:
      history_list = pickle.load(file)

  history_acc = []
  history_val_acc = []
  history_loss = []
  history_val_loss = []

  # Recopilar las precisiones (accuracy) de todas las historias en history_list
  for history in history_list:
      history_acc.extend(history['acc'])
      history_val_acc.extend(history['val_acc'])
      history_loss.extend(history['loss'])
      history_val_loss.extend(history['val_loss'])

  
  # Crear una figura y ejes para el gráfico
  #fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
  
  # Graficar la precisión (accuracy) en función de las épocas
  #axes[0].plot(history_acc, label='Train Accuracy')
  #axes[0].plot(history_val_acc, label='Validation Accuracy')
  #axes[0].set_title('Accuracy')
  #axes[0].set_xlabel('Epoch')
  #axes[0].set_ylabel('Accuracy')
  #axes[0].legend()

  #axes[1].plot(history_loss, label='Train Loss')
  #axes[1].plot(history_val_loss, label='Validation Loss')
  #axes[1].set_title('Loss')
  #axes[1].set_xlabel('Epoch')
  #axes[1].set_ylabel('Loss')
  #axes[1].legend()

  #plt.tight_layout()
  #plt.show()


  fig = sns.catplot(x='Epoch', y='Accuracy', data=history_val_acc, kind='point')
  st.pyplot(fig)
