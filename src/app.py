import streamlit as st
import requests

def format_confidence(confidence):
    return f"{confidence:.2%}"

st.title("Name Origin Checker")

name = st.text_input("Enter a name:")

if st.button("Predict"):
    response = requests.post("http://127.0.0.1:8000/predict", json={"name": name})
    data = response.json()

    st.header(f"Predictions for {data['name']}:")

    sorted_predictions = sorted(zip(data['categories'], data['confidences'][0]), key=lambda x: x[1], reverse=True)

    for category, confidence in sorted_predictions:
        st.markdown(f"**{category}:** {format_confidence(confidence)}")
