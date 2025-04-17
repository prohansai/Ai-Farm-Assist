from flask import Flask, render_template, request, redirect
import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pickle

app = Flask(__name__)

# At the top level of your Flask file (global scope)
fertilizer_encoder = joblib.load("fertilizer_label_encoder.pkl")

# Directories
MODEL_DIR = "models/"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load plant disease model (Keras)
plant_model_path = os.path.join(MODEL_DIR, "plant_disease_model")
plant_model = tf.keras.models.load_model(plant_model_path)

# Load crop and fertilizer models (Pickle)
with open("crop_model.pkl", "rb") as f:
    crop_model = pickle.load(f)

with open("fertilizer_model.pkl", "rb") as f:
    fertilizer_model = pickle.load(f)

# Define plant class labels
# Optional: you can replace this with a manual list if your model doesn’t have class_names attribute
class_labels = ["Healthy", "Powdery", "Rust"]

# ------------------ ROUTES ------------------

@app.route("/")
def home():
    return render_template("index.html")

# ------------ PLANT DISEASE PREDICTION ------------
@app.route("/plant_disease_prediction", methods=["GET", "POST"])
def plant_disease():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return redirect(request.url)

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        img = image.load_img(file_path, target_size=(150, 150))  # Match MobileNetV2 input
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = plant_model.predict(img_array)

        predicted_class = class_labels[np.argmax(predictions)]

        return render_template("plant_disease_prediction.html", prediction=predicted_class, image_path=file_path)

    return render_template("plant_disease_prediction.html")

# ------------ CROP RECOMMENDATION ------------
@app.route("/crop_recommendation", methods=["GET", "POST"])
def crop_recommendation():
    if request.method == "POST":
        try:
            # Collecting the input values from the form
            inputs = [float(request.form[key]) for key in ["nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall"]]
            
            # Getting the prediction (index of the crop), extracting the integer value
            prediction_index = int(crop_model.predict([inputs])[0])  # Ensure it is an integer
            
            # Map the prediction index to a crop name
            crop_names = {
                0: "rice",
                1: "maize",
                2: "chickpea",
                3: "kidneybeans",
                4: "pigeonpeas",
                5: "mothbeans",
                6: "mungbean",
                7: "blackgram",
                8: "lentil",
                9: "pomogranate",
                10: "banana",
                11: "mango",
                12: "grapes",
                13: "watermelon",
                14: "muskmelon",
                15: "apple",
                16: "orange",
                17: "papaya",
                18:"coconut",
                19:"cotton",
                20:"jute",
                21:"coffee"



                # Add more mappings as needed based on your model's output
            }

            # Get the crop name using the prediction index
            prediction_name = crop_names.get(prediction_index, "Unknown Crop")
            
            # Render the template with the crop name
            return render_template("crop_recommendation.html", prediction=prediction_name)
        except Exception as e:
            return f"Error in crop recommendation: {e}"

    return render_template("crop_recommendation.html")




# ------------ FERTILIZER RECOMMENDATION ------------
@app.route("/fertilizer_prediction", methods=["GET", "POST"])
def fertilizer_recommendation():
    soil_dict = ['Loamy', 'Sandy', 'Clayey', 'Black', 'Red']
    crop_dict = ['Sugarcane', 'Cotton', 'Millets', 'Paddy', 'Pulses', 'Wheat', 'Tobacco', 'Barley', 'Oil seeds', 'Ground Nuts', 'Maize']


    if request.method == "POST":
        try:
            inputs = [
                float(request.form["temperature"]),
                float(request.form["humidity"]),
                float(request.form["moisture"]),
                int(request.form["soil_type"]),
                int(request.form["crop_type"]),
                float(request.form["nitrogen"]),
                float(request.form["phosphorus"]),
                float(request.form["potassium"])
            ]

            # Predict
            prediction = fertilizer_model.predict([inputs])[0]
            fertilizer_name = fertilizer_encoder.inverse_transform([prediction])[0]

            return render_template(
                "fertilizer_prediction.html",
                prediction=fertilizer_name,
                soil_dict=soil_dict,
                crop_dict=crop_dict
            )
        except Exception as e:
            return f"Error in fertilizer prediction: {e}"

    return render_template(
        "fertilizer_prediction.html",
        soil_dict=soil_dict,
        crop_dict=crop_dict
    )



# ------------------ RUN APP ------------------
if __name__ == "__main__":
    app.run(debug=True)
