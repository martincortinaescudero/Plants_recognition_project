import streamlit as st
st.title("VGG16 Model")
st.write("Introduction")
if st.checkbox("Display"):
  st.write("Plot accuracy")
  ######################################################
  # load my model and history
  ######################################################
  import pickle
  import numpy as np
  import keras
  import matplotlib.pyplot as plt
  from keras.models import load_model
  import seaborn as sns

  
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

  # Crear un DataFrame de pandas con los datos
  df = pd.DataFrame({'Epoch': range(1, len(history_val_acc) + 1),
                     'Train Accuracy': history_acc,
                     'Validation Accuracy': history_val_acc})

  # Crear un gráfico de puntos usando seaborn
  fig = sns.catplot(x='Epoch', y='Validation Accuracy', data=df, kind='point')
  plt.title('Validation Accuracy vs. Epoch')
  st.pyplot(fig)

  st.write("Done!")
