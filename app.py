import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Load the trained model
# -----------------------------
with open("Classifier.pkl", "rb") as file:
    model = pickle.load(file)

# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="Iris Flower Classifier", layout="centered")
st.title("🌸 Iris Flower Classifier")
st.write("Enter the flower measurements and let the model predict the species.")

# Sidebar for inputs
st.sidebar.header("Input Features")

def user_input_features():
    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
    sepal_width  = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
    petal_width  = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    return features

features = user_input_features()

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(features)
prediction_proba = model.predict_proba(features)

# -----------------------------
# Output
# -----------------------------
species = ["Setosa", "Versicolor", "Virginica"]

st.subheader("Prediction")
st.success(f"The predicted species is: **{species[prediction[0]]}**")

st.subheader("Prediction Probabilities")
st.write(
    {species[i]: f"{prediction_proba[0][i]*100:.2f}%" for i in range(len(species))}
)