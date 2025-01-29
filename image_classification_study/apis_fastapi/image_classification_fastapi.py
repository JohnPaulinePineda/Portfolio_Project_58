##################################
# Loading Python libraries
##################################
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.metrics import PrecisionAtRecall, Recall 
from tensorflow.keras.utils import img_to_array, array_to_img, load_img
import cv2
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import io
import os
import random
import math
import base64
import matplotlib
matplotlib.use('Agg')

##################################
# Setting random seed options
# for the analysis
##################################
def set_seed(seed=123):
    np.random.seed(seed) 
    tf.random.set_seed(seed) 
    keras.utils.set_random_seed(seed)
    random.seed(seed)
    tf.config.experimental.enable_op_determinism()
    os.environ['TF_DETERMINISTIC_OPS'] = "1"
    os.environ['TF_CUDNN_DETERMINISM'] = "1"
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed()

##################################
# Defining file paths
##################################
MODELS_PATH = "models"

##################################
# Loading the model
##################################
try:
    final_cnn_model = load_model(os.path.join("..", MODELS_PATH, "cdrbnr_complex_best_model.keras"))
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

##################################
# Recreating the CNN model defined as
# complex CNN with dropout and batch normalization regularization
# using the Functional API structure
##################################

##################################
# Defining the input layer
##################################
fcnnmodel_input_layer = Input(shape=(227, 227, 1), name="input_layer")

##################################
# Using the layers from the Sequential model
# as functions in the Functional API
##################################
set_seed()
fcnnmodel_conv2d_layer = final_cnn_model.layers[0](fcnnmodel_input_layer)
fcnnmodel_maxpooling2d_layer = final_cnn_model.layers[1](fcnnmodel_conv2d_layer)
fcnnmodel_conv2d_1_layer = final_cnn_model.layers[2](fcnnmodel_maxpooling2d_layer)
fcnnmodel_maxpooling2d_1_layer = final_cnn_model.layers[3](fcnnmodel_conv2d_1_layer)
fcnnmodel_conv2d_2_layer = final_cnn_model.layers[4](fcnnmodel_maxpooling2d_1_layer)
fcnnmodel_batchnormalization_layer = final_cnn_model.layers[5](fcnnmodel_conv2d_2_layer)
fcnnmodel_activation_layer = final_cnn_model.layers[6](fcnnmodel_batchnormalization_layer)
fcnnmodel_maxpooling2d_2_layer = final_cnn_model.layers[7](fcnnmodel_activation_layer)
fcnnmodel_flatten_layer = final_cnn_model.layers[8](fcnnmodel_maxpooling2d_2_layer)
fcnnmodel_dense_layer = final_cnn_model.layers[9](fcnnmodel_flatten_layer)
fcnnmodel_dropout_layer = final_cnn_model.layers[10](fcnnmodel_dense_layer)
fcnnmodel_output_layer = final_cnn_model.layers[11](fcnnmodel_dropout_layer)

##################################
# Creating the Functional API model
##################################
final_cnn_model_functional_api = Model(inputs=fcnnmodel_input_layer, outputs=fcnnmodel_output_layer, name="final_cnn_model_fapi")

##################################
# Compiling the Functional API model
# with the same parameters
##################################
set_seed()
final_cnn_model_functional_api.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall(name='recall')])

##################################
# Creating a gradient model for the
# gradient class activation map
# of the first convolutional layer
##################################
grad_model_first_conv2d = Model(inputs=fcnnmodel_input_layer, 
                                outputs=[fcnnmodel_conv2d_layer, fcnnmodel_output_layer], 
                                name="final_cnn_model_fapi_first_conv2d")
set_seed()
grad_model_first_conv2d.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall(name='recall')])

##################################
# Creating a gradient model for the
# gradient class activation map
# of the second convolutional layer
##################################
grad_model_second_conv2d = Model(inputs=fcnnmodel_input_layer, 
                                outputs=[fcnnmodel_conv2d_1_layer, fcnnmodel_output_layer], 
                                name="final_cnn_model_fapi_second_conv2d")
set_seed()
grad_model_second_conv2d.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall(name='recall')])

##################################
# Creating a gradient model for the
# gradient class activation map
# of the third convolutional layer
##################################
grad_model_third_conv2d = Model(inputs=fcnnmodel_input_layer, 
                                outputs=[fcnnmodel_conv2d_2_layer, fcnnmodel_output_layer], 
                                name="final_cnn_model_fapi_third_conv2d")
set_seed()
grad_model_third_conv2d.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall(name='recall')])

##################################
# Defining the input schema for the function that
# predicts the image category,
# estimates class probabilities,
# formulates the gradient class activation map
# from the output of the first to third convolutional layers and
# and superimposes on the actual image
##################################
class ImageInput(BaseModel):
    file: UploadFile = File(..., description="Image file to be processed.")

##################################
# Formulating the API endpoints
##################################

##################################
# Initializing the FastAPI app
##################################
app = FastAPI()

##################################
# Defining a GET endpoint for
# validating API service connection
##################################
@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Classification API!"}

##################################
# Defining a POST endpoint for
# predicting the image category and
# estimating class probabilities
# of an individual test image
##################################
@app.post("/predict-image-category-class-probability/")
async def predict_image_category_class_probability(image_input: ImageInput):
    try:
        # Reading and preprocessing the image
        image_bytes = await image_input.file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = image.resize((227, 227))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=-1)
        image_array = image_array * (1.0 / 255.0)
        image_array = np.expand_dims(image_array, axis=0)

        # Predicting the image category
        predictions = final_cnn_model_functional_api.predict(image)
        classes = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = classes[predicted_class_index]
        # Predicting the class probabilities
        probabilities = {cls: float(prob) * 100 for cls, prob in zip(classes, predictions[0])}

        # Returning the endpoint response
        return {
            "predicted_class": predicted_class, 
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

        # Returning the endpoint response
        return {
            "predicted_class": predicted_class, 
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

##################################
# Defining a POST endpoint for
# formulating the gradient class activation map
# from the output of the first to third convolutional layers and
# and superimposing on the actual image
##################################
@app.post("/visualize-image-gradcam/")
async def visualize_image_gradcam(image_input: ImageInput):
    try:
        # Reading and preprocessing the image
        image_bytes = await image_input.file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = image.resize((227, 227))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=-1)
        image_array = image_array * (1.0 / 255.0)
        image_array = np.expand_dims(image_array, axis=0)

        # Generating the Grad-CAM heatmaps
        heatmap_first_conv2d, _ = make_gradcam_heatmap(image, grad_model_first_conv2d)
        heatmap_second_conv2d, _ = make_gradcam_heatmap(image, grad_model_second_conv2d)
        heatmap_third_conv2d, _ = make_gradcam_heatmap(image, grad_model_third_conv2d)

        # Converting the heatmaps to images
        def process_heatmap(heatmap):
            heatmap = np.uint8(255 * heatmap)
            jet = plt.colormaps["turbo"]
            jet_heatmap = jet(heatmap)[:, :, :3]
            jet_heatmap = Image.fromarray(jet_heatmap).resize((227, 227))
            return jet_heatmap

        grad_image_first_conv2d = process_heatmap(heatmap_first_conv2d)
        grad_image_second_conv2d = process_heatmap(heatmap_second_conv2d)
        grad_image_third_conv2d = process_heatmap(heatmap_third_conv2d)

        # Creating the Grad-CAM visualization plots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(grad_image_first_conv2d)
        axs[0].set_title('First Conv2D', fontsize=10, weight='bold')
        axs[0].axis('off')
        axs[1].imshow(grad_image_second_conv2d)
        axs[1].set_title('Second Conv2D', fontsize=10, weight='bold')
        axs[1].axis('off')
        axs[2].imshow(grad_image_third_conv2d)
        axs[2].set_title('Third Conv2D', fontsize=10, weight='bold')
        axs[2].axis('off')
       
        # Saving the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        base64_image = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Closing the plot to release resources
        plt.close(fig)

        # Returning the endpoint response
        return {"plot": base64_image}       
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

##################################
# Formulating a function to
# estimate the gradient class activation mapping
# for a test image
##################################
def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    """Generate the Grad-CAM heatmap visualization for a test image."""
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), preds

##################################
# Running the FastAPI app
##################################
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)  
