import streamlit as st
import requests

st.title("🌸 Iris Classifier (via FastAPI)")
st.markdown("Enter values to get predictions from the MLflow model hosted via FastAPI.")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict"):
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict/", json=payload)
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"Predicted Class: `{prediction}`")
        else:
            st.error(f"API Error: {response.status_code} — {response.text}")
    except Exception as e:
        st.error(f"Connection Failed: {str(e)}")
