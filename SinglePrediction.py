import streamlit as st
import joblib
import numpy as np

def show_single():
    st.title("Mental Health Prediction")

    # Input sesuai fitur model
    gender = float(st.selectbox("Gender (0=Male, 1=Female, 2=Others)", [0.0, 1.0, 2.0]))
    country = float(st.number_input("Country (encoded as number)"))
    occupation = float(st.number_input("Occupation (encoded as number)"))
    self_employed = float(st.selectbox("Self-employed (0=No, 1=Yes)", [0.0, 1.0]))
    family_history = float(st.selectbox("Family History of Mental Illness? (0=No, 1=Yes)", [0.0, 1.0]))
    treatment = float(st.selectbox("Have you sought treatment before? (0=No, 1=Yes)", [0.0, 1.0]))
    growing_stress = float(st.slider("Growing Stress Level (0-10)", 0.0, 10.0, 5.0))
    mental_health_history = float(st.slider("Mental Health History Score (0-10)", 0.0, 10.0, 5.0))
    coping_struggles = float(st.selectbox("Coping Struggles (0=No, 1=Yes)", [0.0, 1.0]))
    interview = float(st.slider("Mental Health Interview Score (0-10)", 0.0, 10.0, 5.0))

    # Pilih model
    use_knn = st.checkbox("Use KNN")
    use_svm = st.checkbox("Use SVM")
    use_nn = st.checkbox("Use Neural Network")
    use_dt = st.checkbox("Use Decision Tree")

    btn = st.button("Predict")

    if btn:
        input_data = np.array([
            gender, country, occupation, self_employed,
            family_history, treatment, growing_stress,
            mental_health_history, coping_struggles, interview
        ]).reshape(1, -1)

        def show_prediction(model_name, model_file):
            model = joblib.load(model_file)
            pred = model.predict(input_data)
            label_map = {0: "Yes", 1: "No", 2: "Not Sure"}
            label = label_map.get(pred[0], "Unknown")
            st.subheader(f"{model_name} Prediction: {pred[0]} → {label}")

        if use_knn:
            show_prediction("K-Nearest Neighbors", "modelJb_KNN.joblib")
        if use_svm:
            show_prediction("Support Vector Machine", "modelJb_SVM.joblib")
        if use_nn:
            show_prediction("Neural Network", "modelJb_NN.joblib")
        if use_dt:
            show_prediction("Decision Tree", "modelJb_DecisionTree.joblib")
