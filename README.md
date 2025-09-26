# AI Farm Assist 🌾🤖

AI Farm Assist is a web application designed to empower farmers with data-driven insights for better agricultural practices. This tool leverages the power of machine learning to provide crucial recommendations on crop selection, fertiliser usage, and early detection of plant diseases.

---

## ✨ Key Features

* **Crop Recommendation:** Suggests the most suitable crop to plant based on key soil and environmental factors, including Nitrogen, Phosphorus, Potassium levels, temperature, humidity, pH, and rainfall.

* **Fertiliser Recommendation:** Provides tailored fertiliser suggestions to optimize soil health and maximize crop yield based on soil composition and the selected crop.

* **Plant Disease Prediction:** Identifies potential diseases from an uploaded image of a plant leaf, allowing for early detection and timely intervention to protect crops.

---

## 🛠️ Tech Stack & Architecture

This project is built using a combination of a lightweight web framework and powerful machine learning models.

* **Backend:** Flask (Python)
* **Frontend:** HTML, CSS, JavaScript
* **Machine Learning Models:**
    * **Crop Recommendation:** `CatBoost`
    * **Fertiliser Recommendation:** `XGBoost`
    * **Disease Prediction:** `MobileNetV2` (Deep Learning Model)
* **Core Libraries:** scikit-learn, pandas, numpy, TensorFlow/Keras, Pillow (for image processing)
* **Model Deployment:** The trained models are saved as `.pkl` files and loaded into the Flask application for efficient, real-time predictions without needing to retrain on every request.

---

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed on your system:
* **Python:** Version 3.8 or higher.
* **pip (Python package installer):** Usually comes with Python.

---

## ⚙️ Installation and Running

To get a local copy up and running, follow these simple steps.

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/prohansai/Ai-Farm-Assist.git](https://github.com/prohansai/Ai-Farm-Assist.git)
    ```
2.  **Navigate to the project directory:**
    ```sh
    cd Ai-Farm-Assist
    ```
3.  **Create and activate a virtual environment (Recommended):**
    * **Windows:**
        ```sh
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```sh
        python3 -m venv venv
        source venv/bin/activate
        ```
4.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file by running `pip freeze > requirements.txt` in your project's terminal.)*

5.  **Run the Flask application:**
    ```sh
    python app.py
    ```

---

## 🚀 Usage

1.  Once the server is running, open your web browser and go to: `http://127.0.0.1:5000`
2.  Navigate to one of the three main features using the navigation bar.
3.  **For Crop/Fertiliser Recommendation:** Fill in the required soil and environmental data into the input form and submit.
4.  **For Disease Prediction:** Upload an image of a plant leaf and submit.
5.  The application will display the model's prediction or recommendation on the same page.

---

## 📸 Screenshots

| Homepage                                         | Crop Recommendation                                      |
| ------------------------------------------------ | -------------------------------------------------------- |
| ![Homepage](https://github.com/prohansai/Ai-Farm-Assist/blob/main/screenshots/Screenshot%202025-04-07%20215435.png?raw=true)         | ![Crop Recommendation Form](https://github.com/prohansai/Ai-Farm-Assist/blob/main/screenshots/Screenshot%202025-04-07%20215532.png?raw=true) |
| **Fertiliser Recommendation** | **Disease Prediction** |
| ![Fertiliser Recommendation Form](https://github.com/prohansai/Ai-Farm-Assist/blob/main/screenshots/Screenshot%202025-04-07%20215644.png?raw=true) | ![Disease Prediction Upload](https://github.com/prohansai/Ai-Farm-Assist/blob/main/screenshots/Screenshot%202025-04-07%20215740.png?raw=true)  |


---

## 💡 Future Improvements

* **User Authentication:** Add user accounts to allow farmers to save their farm data and prediction history.
* **Weather API Integration:** Integrate a live weather API to automatically fetch real-time temperature and humidity data for more accurate recommendations.
* **Expanded Disease Database:** Train the model on a wider variety of plant diseases and crop types.
* **Deployment:** Deploy the application to a cloud service like Heroku, AWS, or Azure for public access.
