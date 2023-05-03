import os
os.environ['USE_XNNPACK'] = '0'
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")


interpreter.allocate_tensors()
classes = ['Apple___Apple_scab',
           'Apple___Black_rot',
           'Apple___Cedar_apple_rust',
           'Apple___healthy',
           'Blueberry___healthy',
           'Cherry_(including_sour)___Powdery_mildew',
           'Cherry_(including_sour)___healthy',
           'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
           'Corn_(maize)__Common_rust',
           'Corn_(maize)___Northern_Leaf_Blight',
           'Corn_(maize)___healthy',
           'Grape___Black_rot',
           'Grape__Esca(Black_Measles)',
           'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
           'Grape___healthy',
           'Orange__Haunglongbing(Citrus_greening)',
           'Peach___Bacterial_spot',
           'Peach_healthy', 'Pepper,_bell_Bacterial_spot',
           'Pepper,bell__healthy',
           'Potato___Early_blight',
           'Potato___Late_blight',
           'Potato___healthy',
           'Raspberry___healthy',
           'Soybean___healthy',
           'Squash___Powdery_mildew',
           'Strawberry___Leaf_scorch',
           'Strawberry___healthy',
           'Tomato___Bacterial_spot',
           'Tomato___Early_blight',
           'Tomato___Late_blight',
           'Tomato___Leaf_Mold',
           'Tomato___Septoria_leaf_spot',
           'Tomato___Spider_mites Two-spotted_spider_mite',
           'Tomato___Target_Spot',
           'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
           'Tomato___Tomato_mosaic_virus',
           'Tomato___healthy']

# Define API endpoint for receiving image data
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load image from request
        image_file = request.files['image']
        image = Image.open(image_file)

        # Preprocess the image
        image = image.resize((224, 224))  # Resize to the input size of the model
        image = np.asarray(image)
        image = (image / 255.0).astype(np.float32)  # Normalize pixel values to [0, 1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Run inference with the TFLite model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        class_idx = np.argmax(prediction)
        # return jsonify({class_name})
        return jsonify(classes[class_idx])


    except Exception as e:
        return str(e), 400
if __name__ == '_main_':
    app.run(host='0.0.0.0', port=5000)  # Run the Flask app
