import streamlit as st
import pickle
import numpy as np

# 1. Model load karein
with open('knn_diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

# App ka Title
st.title("Diabetes Prediction App (KNN)")
st.write("Enter the patient's details below to predict diabetes status:")

# 2. Input Fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# 3. Prediction Logic
if st.button("Predict"):
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(features)[0]
    
    if prediction == 1:
        st.error("The model predicts: Diabetes detected (Positive)")
    else:
        st.success("The model predicts: No Diabetes detected (Negative)")
        
    # 4. Classification Report
    st.markdown("---")
    st.subheader("📊 Model Performance (Classification Report)")
    st.text("""
    Accuracy: ~78%
    Precision: 0.75 | Recall: 0.71 | F1-Score: 0.73
    """)
