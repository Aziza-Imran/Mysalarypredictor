import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Change background color to light pink using custom CSS (direct approach)


# Button to clear the cache
if st.button("🌸"):
    st.cache_data.clear()  # Clears @st.cache_data cache
    st.cache_resource.clear()  # Clears @st.cache_resource cache (if used)
    st.success("Cache cleared!")

# Set page title
st.title("𝓕𝓵𝓸𝔀𝓮𝓻𝓢𝓮𝓷𝓼𝓮🌸")

# Display the welcome message
st.markdown("""
    ### Welcome to Flower Sense! 
    𝙀𝙣𝙩𝙚𝙧 𝙩𝙝𝙚 𝙨𝙚𝙥𝙖𝙡 𝙖𝙣𝙙 𝙥𝙚𝙩𝙖𝙡 𝙢𝙚𝙖𝙨𝙪𝙧𝙚𝙢𝙚𝙣𝙩𝙨, 𝙖𝙣𝙙 𝙡𝙚𝙩 𝙩𝙝𝙚 𝙖𝙥𝙥 𝙥𝙧𝙚𝙙𝙞𝙘𝙩 𝙩𝙝𝙚 𝙨𝙥𝙚𝙘𝙞𝙚𝙨 𝙤𝙛 𝙩𝙝𝙚 𝙄𝙧𝙞𝙨 𝙛𝙡𝙤𝙬𝙚𝙧 𝙗𝙖𝙨𝙚𝙙 𝙤𝙣 𝙮𝙤𝙪𝙧 𝙞𝙣𝙥𝙪𝙩. 𝙀𝙖𝙨𝙮 𝙖𝙣𝙙 𝙦𝙪𝙞𝙘𝙠 𝙞𝙙𝙚𝙣𝙩𝙞𝙛𝙞𝙘𝙖𝙩𝙞𝙤𝙣 𝙖𝙩 𝙮𝙤𝙪𝙧 𝙛𝙞𝙣𝙜𝙚𝙧𝙩𝙞𝙥𝙨!
""")

# Load the data
@st.cache_data  # Update cache decorator
def load_data():
    # Replace with your data path (can be from GitHub or local directory)
    df = pd.read_csv("Iris.csv")
    return df

df = load_data()

# Prepare data for training
train, test = train_test_split(df, test_size=0.3, random_state=42)
train_X = train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
train_y = train.Species
test_X = test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
test_y = test.Species

# Encode target labels
label_encoder = LabelEncoder()
train_y_encoded = label_encoder.fit_transform(train_y)
test_y_encoded = label_encoder.transform(test_y)

# Train a Decision Tree model
@st.cache_resource  # Update cache decorator for the model
def train_model():
    model = DecisionTreeClassifier()
    model.fit(train_X, train_y_encoded)
    return model

model = train_model()

# Allow users to make predictions with the trained model
st.write("### Make Predictions on New Data")

# User inputs for prediction
sepal_length = st.number_input("Enter Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Enter Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Enter Petal Length (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_width = st.number_input("Enter Petal Width (cm)", min_value=0.0, max_value=10.0, value=1.0)

# Create a DataFrame for the input values
input_data = pd.DataFrame({
    'SepalLengthCm': [sepal_length],
    'SepalWidthCm': [sepal_width],
    'PetalLengthCm': [petal_length],
    'PetalWidthCm': [petal_width]
})

# Make a prediction for the user input
predicted_class = model.predict(input_data)
predicted_species = label_encoder.inverse_transform(predicted_class)

# Display prediction result
st.write(f"The predicted species for the given input is: {predicted_species[0]}")
