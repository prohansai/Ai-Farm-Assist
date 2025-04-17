import os
import pandas as pd
import numpy as np
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------- CROP RECOMMENDATION (Commented) ----------------------
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("Training Crop Recommendation Model (CatBoost)...")
crop_data = pd.read_csv('crop_recommendation.csv')
X_crop = crop_data.drop('label', axis=1)
y_crop = crop_data['label']
crop_encoder = LabelEncoder()
y_crop_encoded = crop_encoder.fit_transform(y_crop)

X_train_crop, X_test_crop, y_train_crop, y_test_crop = train_test_split(X_crop, y_crop_encoded, test_size=0.2, random_state=42)
crop_model = CatBoostClassifier(verbose=0)
crop_model.fit(X_train_crop, y_train_crop)

with open('crop_model.pkl', 'wb') as f:
    pickle.dump(crop_model, f)
with open('crop_label_encoder.pkl', 'wb') as f:
    pickle.dump(crop_encoder, f)
print(f"Crop Recommendation Model Accuracy: {crop_model.score(X_test_crop, y_test_crop):.2f}")

# ---------------------- FERTILIZER PREDICTION (Commented) ----------------------
from xgboost import XGBClassifier

print("\nTraining Fertilizer Prediction Model (XGBoost)...")
fertilizer_data = pd.read_csv("Fertilizer_Prediction.csv")
soil_map = {'Loamy': 1, 'Sandy': 2, 'Clayey': 3, 'Black': 4, 'Red': 5}
crop_map = {'Sugarcane': 1, 'Cotton': 2, 'Millets': 3, 'Paddy': 4, 'Pulses': 5,
            'Wheat': 6, 'Tobacco': 7, 'Barley': 8, 'Oil seeds': 9, 'Ground Nuts': 10, 'Maize': 11}
fertilizer_data['Soil_Num'] = fertilizer_data['Soil Type'].map(soil_map)
fertilizer_data['Crop_Num'] = fertilizer_data['Crop Type'].map(crop_map)
fertilizer_data.drop(['Soil Type', 'Crop Type'], axis=1, inplace=True)

X_fert = fertilizer_data.drop('Fertilizer Name', axis=1)
y_fert = fertilizer_data['Fertilizer Name']
fertilizer_encoder = LabelEncoder()
y_fert_encoded = fertilizer_encoder.fit_transform(y_fert)

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_fert, y_fert_encoded, test_size=0.2, random_state=42)
fert_model = XGBClassifier(eval_metric='mlogloss')
fert_model.fit(X_train_f, y_train_f)

with open('fertilizer_model.pkl', 'wb') as f:
    pickle.dump(fert_model, f)
with open('fertilizer_label_encoder.pkl', 'wb') as f:
    pickle.dump(fertilizer_encoder, f)
print(f"Fertilizer Prediction Model Accuracy: {fert_model.score(X_test_f, y_test_f):.2f}")

# ---------------------- PLANT DISEASE DETECTION ----------------------
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2  # Use MobileNetV2 instead of EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------- PLANT DISEASE DETECTION ----------------------
print("\nTraining Plant Disease Detection Model (MobileNetV2)...")

dataset_dir = "dataset"
train_dir = os.path.join(dataset_dir, "Train", "Train")
test_dir = os.path.join(dataset_dir, "Test", "Test")

# Lower image resolution for memory optimization
img_size = (150, 150)  # Reduced image size
batch_size = 16  # Reduced batch size for memory management

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)

# Using MobileNetV2 as the base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  # Freeze the base model

# Build the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(train_gen.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Clear previous session to avoid memory issues
tf.keras.backend.clear_session()

# Train the model with 3 epochs
model.fit(train_gen, validation_data=test_gen, epochs=1)

# Save the model in TensorFlow's SavedModel format
if not os.path.exists("models"):
    os.makedirs("models")
model.save("models/plant_disease_model")  # Save in TensorFlow's SavedModel format

# Evaluate and print accuracy
loss, accuracy = model.evaluate(test_gen)
print(f"Plant Disease Detection Model Accuracy: {accuracy:.2f}")
