import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('drug30.pkl')
scaler = joblib.load('scaler_30.pkl')

# Load the dataset
df = pd.read_csv('test.csv')

# Take up to 10 rows with all 30 features
sample_size = min(10, len(df))
sampled_df = df.head(sample_size)

# Streamlit app
st.title('Drug induced Autoimmunity Prediction')

# Display the sample data
st.write('Sample Data:')
st.dataframe(sampled_df)

# User selection
selected_row = st.selectbox('Select a row for prediction:', sampled_df.index)

# Prepare input data
input_data = sampled_df.loc[selected_row].values.reshape(1, -1)
input_data_scaled = scaler.transform(input_data)

# Predict button
if st.button('Predict'):
    # Make prediction
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)

    # Display prediction
    st.write(f'Prediction Probabilities: {prediction_proba}')
    st.write(f'Prediction: {"Positive" if prediction[0] == 1 else "Negative"}')
