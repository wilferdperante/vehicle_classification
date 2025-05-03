from flask import Flask, render_template, request, url_for
import numpy as np
import cv2 as cv
import os
from tensorflow.keras import models
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Changed to serve as static files

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the pre-trained model
model = models.load_model('image_classifier.keras')

# Class names matching the filtered model
class_names = ['plane', 'car', 'ship', 'truck']

def preprocess_image(img_path):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert from BGR to RGB
    img = cv.resize(img, (32, 32))
    img = img / 255.0
    img = img[None, :]
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_path = None
    
    if request.method == 'POST':
        # Handle the uploaded image
        file = request.files['file']
        if file and file.filename != '':
            # Secure filename and save
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)

            # Preprocess the image
            img = preprocess_image(img_path)

            # Make a prediction
            prediction = model.predict(img)
            index = np.argmax(prediction)
            result = class_names[index]
            
            # Use url_for to generate proper image path
            image_path = url_for('static', filename=f'uploads/{filename}')

    return render_template('index.html', result=result, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
