import streamlit as st
import tensorflow as tf


@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('pages/my_cotton_classification_model.hdf5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Cotton Water Stress Classification
         """
         )

file = st.file_uploader("Please upload an brain scan file", type=["TIF"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(image_data, model):
    
        size = (64,64)
        x = []    
        # full_size_image = image_data
        # x.append(cv2.resize(full_size_image, size, interpolation=cv2.INTER_CUBIC))
        # x = np.array(x)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        
        return prediction

class_names=['rainfed', 'fully irrigated', 'percent deficit', 'time delay']
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    st.write(predictions)
    st.write(score)
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
    st.write(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))

