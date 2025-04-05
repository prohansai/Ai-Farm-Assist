
# to run, type : python app.py


from flask import Flask, render_template, request, redirect
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pickle

app = Flask(__name__)

# Define the path to the models directory
MODEL_DIR = "models/"

# Load the plant disease classification model
plant_model_path = os.path.join(MODEL_DIR, "plant_disease_model.h5")
plant_model = tf.keras.models.load_model(plant_model_path)

# Load the crop recommendation and fertilizer prediction models
crop_model_path = os.path.join(MODEL_DIR, 'crop_recommendation_model.pkl')
label_encoder_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
fertilizer_model_path = os.path.join(MODEL_DIR, 'Fertclassifier.pkl')

crop_model = pickle.load(open(crop_model_path, 'rb'))
label_encoder = pickle.load(open(label_encoder_path, 'rb'))
fertilizer_model = pickle.load(open(fertilizer_model_path, 'rb'))

# Define class labels for plant disease classification
class_labels = ["Healthy", "Powdery", "Rust"]  # Update this list as needed

# Define dictionaries for soil and crop types
soil_dict = ['Loamy', 'Sandy', 'Clayey', 'Black', 'Red']
crop_dict = ['Sugarcane', 'Cotton', 'Millets', 'Paddy', 'Pulses', 'Wheat', 'Tobacco', 'Barley', 'Oil seeds', 'Ground Nuts', 'Maize']

# Ensure the uploads folder exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Home route with navigation
@app.route("/")
def home():
    return render_template("index.html")

# Route for plant disease classification
@app.route("/plant_disease_prediction", methods=["GET", "POST"])
def plant_disease():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Preprocess the image
            img = image.load_img(file_path, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Make prediction
            predictions = plant_model.predict(img_array)
            predicted_class = class_labels[np.argmax(predictions)]

            return render_template("plant_disease_prediction.html", prediction=predicted_class, image_path=file_path)
    return render_template("plant_disease_prediction.html")

# Route for crop recommendation
@app.route('/crop_recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        features = np.array([N, P, K, temperature, humidity, ph, rainfall]).reshape(1, -1)
        prediction = crop_model.predict(features)
        predicted_crop = label_encoder.inverse_transform(prediction)[0]

        return render_template('crop_recommendation.html', prediction_text=f'Recommended crop: {predicted_crop}')
    
    return render_template('crop_recommendation.html')

# Route for fertilizer prediction
@app.route('/fertilizer_prediction', methods=['GET', 'POST'])
def fertilizer_prediction():
    selected_soil = None
    selected_crop = None
    
    if request.method == 'POST':
        features = [
            float(request.form['temp']),
            float(request.form['humidity']),
            float(request.form['moisture']),
            float(request.form['n']),
            float(request.form['p']),
            float(request.form['k']),
            soil_dict.index(request.form['soil_type']) + 1,
            crop_dict.index(request.form['crop_type']) + 1
        ]
        
        selected_soil = request.form['soil_type']
        selected_crop = request.form['crop_type']

        prediction = fertilizer_model.predict([features])[0]

        return render_template('fertilizer_prediction.html', fertilizer=prediction,
                               soil_types=soil_dict, crop_types=crop_dict,
                               selected_soil=selected_soil, selected_crop=selected_crop)
    
    return render_template('fertilizer_prediction.html', soil_types=soil_dict, crop_types=crop_dict)

if __name__ == "__main__":
    app.run(debug=True)
