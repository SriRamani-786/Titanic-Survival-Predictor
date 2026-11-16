
import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Application title
st.title("🚢 Titanic Survival Predictor")

st.write(
    "Enter passenger details below to estimate "
    "the probability of survival."
)

# Feature order must match Project 04
FEATURES = ["pclass", "sex", "age", "fare"]

# Load model only once
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


# Passenger input section
st.subheader("Passenger Information")

pclass = st.selectbox(
    "Ticket Class",
    [1, 2, 3]
)

sex = st.radio(
    "Sex",
    ["Female", "Male"],
    horizontal=True
)

age = st.slider(
    "Age",
    min_value=1,
    max_value=80,
    value=30
)

fare = st.slider(
    "Fare Paid",
    min_value=0.0,
    max_value=500.0,
    value=30.0,
    step=0.5
)

# Prediction button
if st.button("Predict Survival"):

    try:
        # Check whether model file exists
        if not os.path.exists("model.joblib"):
            st.error(
                "Model file is missing. "
                "Please provide model.joblib."
            )
            st.stop()

        # Load trained model
        model = load_model()

        # Encode sex using Project 04 encoding
        sex_encoded = 1 if sex == "Female" else 0

        # Create input DataFrame
        input_data = pd.DataFrame(
            [[pclass, sex_encoded, age, fare]],
            columns=FEATURES
        )

        # Generate prediction
        prediction = model.predict(input_data)[0]

        # Generate survival probability
        probabilities = model.predict_proba(input_data)[0]

        # Find probability corresponding to class 1
        survival_index = list(model.classes_).index(1)

        survival_probability = probabilities[survival_index]

        # Display result
        st.subheader("Prediction Result")

        st.metric(
            "Estimated Chance of Survival",
            f"{survival_probability:.2%}"
        )

        if prediction == 1:
            st.success(
                "The model predicts that the passenger may survive."
            )
        else:
            st.warning(
                "The model predicts that the passenger may not survive."
            )

    except FileNotFoundError:
        st.error(
            "The model file is missing. Please upload model.joblib."
        )

    except Exception as e:
        st.error(
            f"Unable to generate prediction: {e}"
        )

# Educational explanation
st.caption(
    "This educational application uses a Decision Tree model "
    "trained on the historical Titanic dataset. "
    "Predictions are estimates and not guarantees."
)
