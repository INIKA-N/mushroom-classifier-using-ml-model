import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the data
@st.cache_data()
def load_data():
    return pd.read_csv("mushrooms.csv")

df = load_data()

# Load the trained RandomForestClassifier model
@st.cache_data()
def load_model():
    with open("random_forest_model.pkl", "rb") as model_file:
        return pickle.load(model_file)

model = load_model()

# Streamlit app
st.title("Mushroom Classifier App")
st.subheader("Predict whether a mushroom is edible or poisonous")

# Create a form for user input
with st.sidebar.form("user_input_form"):
    st.header("Input Features:")
    user_input = {}
    for col in df.columns:
        if col not in ['class', 'veil-type', 'gill-attachment', 'ring-type', 'gill-color', 'bruises']:
            user_input[col] = st.selectbox(f"{col.capitalize()}", df[col].unique())

    # Button to trigger prediction
    predict_button = st.form_submit_button("Predict")

# Display prediction and results
if predict_button:
    st.success("Prediction successful!")

    # Make predictions using the loaded model based on user input
    input_data = pd.DataFrame(user_input, index=[0])

    # Label encode user input
    def label_encoded(feat):
        le = LabelEncoder()
        le.fit(df[feat])
        return le.transform(input_data[feat])

    for col in input_data.columns:
        input_data[col] = label_encoded(col)

    # Make predictions using the loaded model
    prediction = model.predict(input_data)

    # Display the prediction
    st.subheader("Prediction:")
    predicted_class = "Edible" if prediction[0] == 0 else "Poisonous"
    st.write(predicted_class)

# Add some additional effects and information
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app predicts whether a mushroom is edible or poisonous based on user input."
    "\n\n**Disclaimer:** This is a simple demo app and should not be used for actual decision-making."
)

# Add some styling
st.markdown(
    """
    <style>
        .css-17yyfsi {
            background-color: #f0f8ff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
        .css-1yqgjjj {
            color: #191970;
        }
        .css-1t42kcs {
            color: #008080;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
