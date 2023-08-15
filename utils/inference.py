from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


model = load_model("model/infastate_segm_model.h5")

def detection(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    
    return predictions

