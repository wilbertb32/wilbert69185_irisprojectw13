import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load the KNN model
model_path = os.path.join(os.path.dirname(__file__), 'knn_model.sav')
model = joblib.load(model_path)

st.title("Classifying Iris Flowers")
st.markdown("Toy model to play to classify iris flowers into (sentosa, versicolor, virginica) based on their sepal/petal and length/width.")

st.header("Plant Features")
col1, col2 = st.columns(2)

with col1:
    st.text("Sepal characteristics")
    sepal_l = st.slider("Sepal length (cm)", 1.0, 8.0, 0.5)
    sepal_w = st.slider("Sepal width (cm)", 2.0, 4.4, 0.5)

with col2:
    st.text("Petal characteristics")
    petal_l = st.slider("Petal length (cm)", 1.0, 7.0, 0.5)
    petal_w = st.slider("Petal width (cm)", 0.1, 2.5, 0.5)

st.text('')
if st.button("Predict type of Iris"):
    # Use the loaded KNN model directly
    input_data = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    result = model.predict(input_data)
    st.text(result[0])

st.text('')
st.text('')