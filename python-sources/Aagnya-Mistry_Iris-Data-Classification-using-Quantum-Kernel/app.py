import streamlit as st
import numpy as np
import joblib

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from qiskit_machine_learning.algorithms.classifiers import QSVC

# -------- Load model and scaler --------
model = joblib.load("qsvc_custom.pkl")
scaler = joblib.load("scaler.pkl")

# -------- UI Layout --------
st.set_page_config(page_title="Quantum Iris Classifier", layout="centered")
st.title("🌸 Quantum SVM Iris Classifier")
st.markdown("Classifies an Iris flower as **Setosa**, **Versicolor**, or **Virginica** using a Custom Quantum Kernel.")

# -------- User Inputs --------
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# -------- Predict Button --------
if st.button("Classify"):
    # Scale the input
    scaled_input = scaler.transform(input_data)

    # Predict using QSVM
    prediction = model.predict(scaled_input)[0]

    # Map prediction to labels
    labels = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"🌿 Prediction: **{labels[prediction]}**")
