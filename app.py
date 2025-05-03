from flask import Flask, render_template_string, request, url_for
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

# HTML template as a string
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Classifier</title>
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f8f9fa;
            padding: 20px;
        }
        .upload-container {
            max-width: 600px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .result-image {
            max-width: 300px;
            margin-top: 20px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="upload-container mt-5">
            <h1 class="text-center mb-4">Vehicle Image Classifier</h1>
            <form method="post" enctype="multipart/form-data" class="mb-4">
                <div class="mb-3">
                    <input class="form-control" type="file" name="file" accept=".png, .jpg, .jpeg" required>
                </div>
                <div class="d-grid">
                    <button type="submit" class="btn btn-primary btn-lg">Classify Image</button>
                </div>
            </form>

            {% if result %}
            <div class="text-center">
                <h2 class="text-success">Prediction: <span class="fw-bold">{{ result }}</span></h2>
                <img src="{{ image_path }}" alt="Uploaded Image" class="result-image img-fluid">
            </div>
            {% endif %}
        </div>
    </div>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

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

    return render_template_string(HTML_TEMPLATE, result=result, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
