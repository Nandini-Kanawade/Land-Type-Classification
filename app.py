import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
app = Flask(__name__)

model =load_model('model_file.keras')
print('Model loaded. Check http://127.0.0.1:5000/')

#labels = {0: 'AnnualCrop', 1: 'Forest', 2: 'HerbaceousVegetation', 3: 'Highway',4:'Industrial',5:'Pasture',6:'PermanentCrop',7:'Residential',8:'River',9:'SeaLake'}


def getResult(image_path):
    img = load_img(image_path, target_size=(64, 64))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    print(predictions)
    return predictions
    

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])

def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        predictions=getResult(file_path)
        index=(np.argmax(predictions))
        labels = {0: 'AnnualCrop', 1: 'Forest', 2: 'HerbaceousVegetation', 3: 'Highway',4:'Industrial',5:'Pasture',6:'PermanentCrop',7:'Residential',8:'River',9:'SeaLake'}
        predicted_label =labels[index]
        return str(predicted_label)
    return None


if __name__ == '__main__':
    app.run(debug=True)