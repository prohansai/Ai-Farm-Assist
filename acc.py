import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder

# ---------- Crop Model Evaluation ----------
print("Evaluating Crop Recommendation Model...")
crop_data = pd.read_csv('crop_recommendation.csv')
X_crop = crop_data.drop('label', axis=1)
y_crop = crop_data['label']

with open('crop_model.pkl', 'rb') as f:
    crop_model = pickle.load(f)
with open('crop_label_encoder.pkl', 'rb') as f:
    crop_encoder = pickle.load(f)

y_crop_encoded = crop_encoder.transform(y_crop)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_crop, y_crop_encoded, test_size=0.2, random_state=42)
crop_accuracy = crop_model.score(X_test, y_test)
print(f"Crop Model Accuracy: {crop_accuracy:.2f}")

# ---------- Fertilizer Model Evaluation ----------
print("\nEvaluating Fertilizer Prediction Model...")
fertilizer_data = pd.read_csv("Fertilizer_Prediction.csv")
soil_map = {'Loamy': 1, 'Sandy': 2, 'Clayey': 3, 'Black': 4, 'Red': 5}
crop_map = {'Sugarcane': 1, 'Cotton': 2, 'Millets': 3, 'Paddy': 4, 'Pulses': 5,
            'Wheat': 6, 'Tobacco': 7, 'Barley': 8, 'Oil seeds': 9, 'Ground Nuts': 10, 'Maize': 11}
fertilizer_data['Soil_Num'] = fertilizer_data['Soil Type'].map(soil_map)
fertilizer_data['Crop_Num'] = fertilizer_data['Crop Type'].map(crop_map)
fertilizer_data.drop(['Soil Type', 'Crop Type'], axis=1, inplace=True)

X_fert = fertilizer_data.drop('Fertilizer Name', axis=1)
y_fert = fertilizer_data['Fertilizer Name']

with open('fertilizer_model.pkl', 'rb') as f:
    fert_model = pickle.load(f)
with open('fertilizer_label_encoder.pkl', 'rb') as f:
    fert_encoder = pickle.load(f)

y_fert_encoded = fert_encoder.transform(y_fert)
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_fert, y_fert_encoded, test_size=0.2, random_state=42)
fert_accuracy = fert_model.score(X_test_f, y_test_f)
print(f"Fertilizer Model Accuracy: {fert_accuracy:.2f}")

# ---------- Plant Disease Model Evaluation ----------
print("\nEvaluating Plant Disease Detection Model...")
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size = (150, 150)
batch_size = 16
test_datagen = ImageDataGenerator(rescale=1./255)
test_dir = "dataset/Test/Test"

test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)

plant_model = load_model("models/plant_disease_model")
loss, plant_accuracy = plant_model.evaluate(test_gen)
print(f"Plant Disease Detection Model Accuracy: {plant_accuracy:.2f}")
