import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load Crop Recommendation Dataset
crop_data = pd.read_csv('crop_recommendation.csv')
X = crop_data.drop('label', axis=1)
y = crop_data['label']

# Encode crop labels
y_encoder = LabelEncoder()
y_encoded = y_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest Model
crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(X_train, y_train)

# Save Model and Label Encoder
with open('crop_recommendation_model.pkl', 'wb') as f:
    pickle.dump(crop_model, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(y_encoder, f)

# Load Fertilizer Prediction Dataset
fertilizer_data = pd.read_csv("Fertilizer_Prediction.csv")

# Mapping categorical variables
gr = {
    'Loamy': 1, 'Sandy': 2, 'Clayey': 3, 'Black': 4, 'Red': 5
}
cr = {
    'Sugarcane': 1, 'Cotton': 2, 'Millets': 3, 'Paddy': 4, 'Pulses': 5,
    'Wheat': 6, 'Tobacco': 7, 'Barley': 8, 'Oil seeds': 9,
    'Ground Nuts': 10, 'Maize': 11
}

fertilizer_data['Soil_Num'] = fertilizer_data['Soil Type'].map(gr)
fertilizer_data['Crop_Num'] = fertilizer_data['Crop Type'].map(cr)
fertilizer_data.drop(['Soil Type', 'Crop Type'], axis=1, inplace=True)

X_fertilizer = fertilizer_data.drop(['Fertilizer Name'], axis=1)
Y_fertilizer = fertilizer_data['Fertilizer Name']

# Train-Test Split
X_train_f, X_test_f, Y_train_f, Y_test_f = train_test_split(X_fertilizer, Y_fertilizer, test_size=0.2, random_state=42)

# Train Decision Tree Model
fertilizer_classifier = DecisionTreeClassifier()
fertilizer_classifier.fit(X_train_f, Y_train_f)

# Save Fertilizer Model
with open('Fertclassifier.pkl', 'wb') as f:
    pickle.dump(fertilizer_classifier, f)

# Define Image Paths
dataset_dir = "dataset"
train_dir = os.path.join(dataset_dir, "Train", "Train")
test_dir = os.path.join(dataset_dir, "Test", "Test")

# Image Preprocessing
img_width, img_height, batch_size = 128, 128, 32
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode="categorical"
)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode="categorical"
)

# CNN Model for Plant Disease Detection
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation="softmax"),
])

# Compile and Train Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    epochs=3,
)

# Save the Model
model.save("models/plant_disease_model.h5")
print("All models saved successfully!")
