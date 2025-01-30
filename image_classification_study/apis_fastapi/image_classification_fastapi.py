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
# No input schema was defined for the functions that
# predict the image category,
# estimate class probabilities,
# formulate the gradient class activation map
# from the output of the first to third convolutional layers and
# and superimpose on the actual image
##################################
# The UploadFile parameter was directly declatred in the endpoints because
# FastAPI and Pydantic cannot directly handle UploadFile as a field in a Pydantic BaseModel,
# The UploadFile type is not a standard Pydantic-compatible type 
# and Pydantic does not know how to validate or process it when it is part of a model.
##################################

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
# ensuring that the file upload mechanism is working
# by returning the the file metadata
##################################
@app.post("/test-file-upload/")
async def test_file_upload(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="File must be a JPEG image")
    try:       
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": file.size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

##################################
# Defining a POST endpoint for
# predicting the image category and
# estimating class probabilities
# of an individual test image
##################################
@app.post("/predict-image-category-class-probability/")
async def predict_image_category_class_probability(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="File must be a JPEG image")
    try:
        # Reading and preprocessing the image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = image.resize((227, 227))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=-1)
        image_array = image_array * (1.0 / 255.0)
        image_array = np.expand_dims(image_array, axis=0)
        
        # Predicting the image category
        predictions = final_cnn_model_functional_api.predict(image_array)
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

##################################
# Defining a POST endpoint for
# formulating the gradient class activation map
# from the output of the first to third convolutional layers and
# and superimposing on the actual image
##################################
@app.post("/visualize-image-gradcam/")
async def visualize_image_gradcam(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="File must be a JPEG image")
    try:
        # Reading and preprocessing a grayscale-converted image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = image.resize((227, 227))
        image_array = img_to_array(image)
        image_array = image_array * (1.0 / 255.0)

        # Replicating the original image
        image_nonexpanded = Image.open(io.BytesIO(image_bytes))

        # Generating the Grad-CAM heatmap images
        grad_image_first_conv2d, grad_image_second_conv2d, grad_image_third_conv2d = superimpose_gradcam_heatmap(image_array, image_nonexpanded)

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
    # Computing the output of the last convolutional layer and predictions using GradientTape
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Calculating the gradients of the class output with respect to the last convolutional layer output
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Averaging the gradients spatially to obtain a single vector per filter
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Selecting the output of the last convolutional layer for the first image in the batch
    last_conv_layer_output = last_conv_layer_output[0]

    # Computing the heatmap by multiplying the last convolutional layer output with the pooled gradients
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]

    # Removing the extra dimensions from the heatmap to make it 2D
    heatmap = tf.squeeze(heatmap)

    # Normalizing the heatmap to ensure values are between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    # Returning the heatmap as a NumPy array along with the model predictions
    return heatmap.numpy(), preds

##################################
# Formulating a function to
# superimpose the gradient class activation mapping
# to a test image
##################################
def superimpose_gradcam_heatmap(img, img_nonexpanded):
    """Estimate the Grad-CAM heatmap from the convolutional layers and superimpose onto the original test image."""
    # Expanding the image dimensions to match the model input shape
    img = np.expand_dims(img, axis=0)

    # Generating Grad-CAM heatmaps for three different convolutional layers
    heatmap_first_conv2d, preds_first_conv2d = make_gradcam_heatmap(img, grad_model_first_conv2d)
    heatmap_second_conv2d, preds_second_conv2d = make_gradcam_heatmap(img, grad_model_second_conv2d)
    heatmap_third_conv2d, preds_third_conv2d = make_gradcam_heatmap(img, grad_model_third_conv2d)

    # Scaling the heatmap values to the range [0, 255] for visualization
    heatmap_first_conv2d = np.uint8(255 * heatmap_first_conv2d)
    heatmap_second_conv2d = np.uint8(255 * heatmap_second_conv2d)
    heatmap_third_conv2d = np.uint8(255 * heatmap_third_conv2d)

    # Converting the non-expanded image to a NumPy array for processing
    img = img_to_array(img_nonexpanded)

    # Loading a color map for visualizing the heatmap
    jet = plt.colormaps["turbo"]
    jet_colors = jet(np.arange(256))[:, :3]

    # Defining a function to process and superimpose the heatmap onto the image
    def process_heatmap(heatmap, img):
        # Applying the color map to the heatmap
        jet_heatmap = jet_colors[heatmap]
        # Converting the heatmap array to an image
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        # Resizing the heatmap to match the original image dimensions
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        # Converting the resized heatmap back to a NumPy array
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
        # Superimposing the heatmap onto the original image with transparency
        superimposed_img = jet_heatmap * 0.80 + img
        # Converting the superimposed image array back to an image
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
        # Resizing the final superimposed image to a fixed size
        superimposed_img = superimposed_img.resize((277, 277))
        return superimposed_img

    # Processing and superimposing heatmaps for all three convolutional layers
    superimposed_img_first_conv2d = process_heatmap(heatmap_first_conv2d, img)
    superimposed_img_second_conv2d = process_heatmap(heatmap_second_conv2d, img)
    superimposed_img_third_conv2d = process_heatmap(heatmap_third_conv2d, img)

    # Returning the final superimposed images for all three heatmaps
    return superimposed_img_first_conv2d, superimposed_img_second_conv2d, superimposed_img_third_conv2d

##################################
# Running the FastAPI app
##################################
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)  
