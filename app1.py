import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import streamlit as st

model = load_model("/content/face-expression.keras")
st.header("FACE EXPRESSION CLASSIFICATION")

image_path = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if image_path is not None:

  img = Image.open(image_path).convert('L') # Open image and convert to grayscale
  img = img.resize((48, 48))  # Resize to match the input your model expects
  x = img_to_array(img)
  x=np.expand_dims(x,axis=0)
  x=np.expand_dims(x,axis=-1)
  images = np.vstack([x])
  val = model.predict(images)
  st.image(image_path)
  if val[0][0] == 1:
      st.write("Angry")

  elif val[0][1] == 1:
      st.write("disgust")

  elif val[0][2] == 1:
      st.write("fear")

  elif val[0][3] == 1:
      st.write("happy")

  elif val[0][4] == 1:
      st.write("neutral")

  elif val[0][5] == 1:
      st.write("sad")

  elif val[0][6] == 1:
      st.write("surprise")

  else:
      st.write("none")
else:
  st.write("please upload the file")
