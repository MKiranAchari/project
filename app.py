from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf  # Assuming you have a TensorFlow model

app = Flask(__name__)

# Load your trained ML model
model = tf.keras.models.load_model('path/to/your/trained/model')

# Define an API endpoint for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Ensure that the request contains an image file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Perform any necessary preprocessing on the image
    # For example, resize and convert to numpy array
    image = preprocess_image(file)

    # Make the prediction using your ML model
    prediction = model.predict(np.expand_dims(image, axis=0))

    # Assuming your model outputs a single class as a prediction
    predicted_class = int(np.argmax(prediction))

    # Return the prediction as a JSON response
    return jsonify({'prediction': predicted_class})

def preprocess_image(file):
    # Implement any necessary image preprocessing here
    # For example, using TensorFlow's image processing functions
    # and converting the image to a numpy array
    # This function should return the processed image as a numpy array
    pass

if __name__ == '__main__':
    app.run(debug=True)
