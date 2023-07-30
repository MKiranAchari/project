from __future__ import division,print_function
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io

# app = Flask(__name__)

# # Load your trained ML model
# model = tf.keras.models.load_model(r"E:\aits_materials\lvp\fe_bc\breastcancer.h5")

# def preprocess_image(image):
#     # Implement any necessary image preprocessing here
#     # For example, resizing and converting the image to a numpy array
#     # This function should return the processed image as a numpy array
#     image = image.resize((64,64))  # Adjust the size as per your model's input size
#     image = np.array(image) / 255.0  # Normalize the image
#     return image

# # Define an API endpoint for image prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Ensure that the request contains an image file
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})

#     file = request.files['file']

#     # Read the image from the request and preprocess it
#     image = Image.open(io.BytesIO(file.read()))
#     image = preprocess_image(image)

#     # Make the prediction using your ML model
#     prediction = model.predict(np.expand_dims(image, axis=0))

#     # Assuming your model outputs a single class as a prediction
#     predicted_class = int(np.argmax(prediction))



#     # Return the prediction as a JSON response
#     return jsonify({'prediction': predicted_class})

# @app.route('/templates/index.html')
# def index():
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True,port=5008)


import os
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.models import load_model,Sequential
from flask import Flask,request,render_template
from werkzeug.utils import secure_filename

global graph
graph=tf.get_default_graph()
app=Flask(__name__)


model=load_model("breastcancerpredictmodel.h5")

@app.route('/',methods=['GET'])
def index():
    return render_template('/templates/index.html')


@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(
            basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        img=image.load_img(file_path,target_size=(64,64))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)

        with graph.as_default():
            preds=model.predict_classes(x)
        if preds[0][0]==0:
            text = "The tumor is Benign.. Need not worry!"
        else:
            text = "The tumor is Malignant tumor.. Please Consult Doctor"
        text=text
        return text
if __name__=='__main__':
    app.run(debug=True,threaded=False)