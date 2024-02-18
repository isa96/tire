"""
# How To Run?
python app.py
"""

# Load libraries
import numpy as np
import gradio as gr
from random import choice
from tensorflow.keras.models import load_model

# Load trained VGG16 model
model = load_model("models/model_tire.h5")

# Define additional function namely `classify_image` to apply prediction on input image form.
def classify_image(image, class_to_idx = {0 : "Cracked", 1 : "Normal"}):
    """
        Create function took image from input forms which already preprocess as (224, 224) and np.ndarray format. 
    Step by Step Explanation:
        1. `image / 255.0` this step called image normalization to convert RGB pixel range from (0, 256) into (0, 1). Impact on this approach is to reduce model computation.
        2. `image.reshape(-1, 224, 224, 3)` this step called image reshape, because tensorflow only accept 4D array we convert 3D Image into 4D Image (None, 224, 224, 3), consider None is batch_size of dataset. 
        3. `model.predict(image, verbose = 0)` this step called image inference, based on preprocessed image we apply those image into feed-forward to get probabilities of each label score.
        4. `result[0]` this step we unpact 2D array of probabilities into 1D array.
        5. `result_proba.argmax()` this step we determine which index has the highest score based on probabilities given.
        6. `class_to_idx[result_index]` convert numerical index into string based on `class_to_idx` parameter.
        7. `class_label` this step we apply string-formating to perform easier reading result of model predicted.
    """
    image = image / 255.0
    image = image.reshape(-1, 224, 224, 3)
    result = model.predict(image, verbose = 0)
    result_proba = result[0]
    result_index = result_proba.argmax()
    class_predict = class_to_idx[result_index]
    class_label = f"Predict: {class_predict} with Confidences Score: {np.round(result_proba[result_index] * 100, 2)}"
    return class_label

if __name__ == "__main__":
    gr.Interface(
        fn = classify_image, # Define function used on form image
        inputs = gr.Image(shape = (224, 224)), # Define input size of image according model.inputs dimension, (None, 224, 224, 3)
        outputs = gr.Label(num_top_classes = 2), # Define class available on our task, we only contains 2 label -> Cracked & Normal
        examples = [
            "./examples/Cracked-1.jpg", # Example cracked label image
            "./examples/IMG_4255.jpg"   # Example normal  label image
        ], 
        description = f"Nama: Ahmad Fudoli Zaenun Nazhirin, NIM: 191351108", # Put personal information on description in polite purpose
        title = "Klasifikasi kondisi ban kendaraan menggunakan arsitektur ".title() + "VGG16" # Define title based on our proposal title to entertain what is the interface purpose
    ).launch()