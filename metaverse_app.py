import streamlit as st
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os

import json

def load_json():
    with open('sending_add.json', 'r') as f:
        sending_add = json.load(f)

    with open('receiving_add.json', 'r') as f:
        receiving_add = json.load(f)

    return sending_add,receiving_add
sending_add,receiving_add = load_json()

# Initialize session state for storing rows
if "data" not in st.session_state:
    st.session_state.data = []

# Function to save accumulated data to a CSV file
#def save_to_csv(file_path):
    # Check if the file exists
    #if os.path.exists(file_path):
        # Load existing data and append new rows
        #existing_data = pd.read_csv(file_path)
        #new_data = pd.DataFrame(st.session_state.data)
        #updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    #else:
        # If the file doesn't exist, just save the new data
        #updated_data = pd.DataFrame(st.session_state.data)

    # Save the updated data back to the file
    #updated_data.to_csv(file_path, index=False)
    #st.success(f"Data saved to {file_path}!")

    # Clear session state data
    #st.session_state.data = []

# Title and description
st.title("*Metaverse Transaction Anomaly Prediction*")
st.write("This application predicts the anomaly of the metaverse transaction")
st.header("*Enter your details below*")

# Transaction details input
def transac_details():
    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input('*Transaction amount*')
        login_frequency = st.number_input("*Log in frequency*", 1, 10, 3)
        session_duration = st.number_input("*Session duration*")
        receiving_address = st.text_input('*Destination address*')
        st.session_state.receiving_address_input = receiving_address
        sending_address = st.text_input('*Your address*')
        st.session_state.sending_address_input = sending_address

    with c2:
        transaction_type = st.selectbox('*What type of transaction do you want to carry out?*',
                                        (['purchase', 'transfer', 'sale']))
        location_region = st.selectbox('*Which location are you making this transaction from*',
                                       (['Europe', 'Asia', 'South America', 'North America', 'Africa']))
        purchase_pattern = st.selectbox('*Select your typical purchasing behavior in the metaverse?*',
                                        (['focused', 'high_value', 'random']))
        age_group = st.selectbox('*Select your metaverse experience level*', (['established', 'veteran', 'new']))
        risk_score = st.number_input('*Risk score*')

    sending_address = sending_add.get(sending_address, 0)
    receiving_address = receiving_add.get(receiving_address, 0)

    feat = np.array([amount, login_frequency, session_duration, risk_score, sending_address, receiving_address,
                     location_region, purchase_pattern, age_group, transaction_type]).reshape(1, -1)
    cols = ['amount', 'login_frequency', 'session_duration', 'risk_score', 'sending_address_encoded',
            'receiving_address_encoded', 'location_region', 'purchase_pattern', 'age_group', 'transaction_type']

    feat1 = pd.DataFrame(feat, columns=cols)
    return feat1

df = transac_details()

## Encode categorical features

def load_encoder():

    with open("encoder.pkl","rb") as f:
        encoder = pickle.load(f)
    return encoder
encoder = load_encoder()

def encode(df):
    cat_columns = ['transaction_type', 'location_region', 'purchase_pattern', 'age_group']
    encoded_values = encoder.transform(df[cat_columns])
    encoded_values = pd.DataFrame(encoded_values, columns=encoder.get_feature_names_out(cat_columns))

    df = pd.concat([df.drop(columns=cat_columns), encoded_values], axis=1)
    return df

df = encode(df)

# Load model
def load_model():
    with open("model.pkl","rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Predict transaction risk
prediction = model.predict(df)[0]

import time

# Prediction logic
if st.button('*Click here*'):
    time.sleep(10)
    with st.spinner('Predicting... Please wait...'):
        if prediction == 0:
            st.success("This is a low risk transaction")
        else:
            st.success("This is a high risk transaction")

    # Add data to session state for storage
    st.session_state.data.append({
        'amount': df['amount'][0],
        'login_frequency': df['login_frequency'][0],
        'session_duration': df['session_duration'][0],
        'risk_score': df['risk_score'][0],
        'sending_address': st.session_state.sending_address_input,
        'receiving_address': st.session_state.receiving_address_input,
        'prediction': 'Low Risk' if prediction == 0 else 'High Risk'
    })
    #st.success("Transaction details added to memory.")

# Save to CSV button
#if st.button('Save to CSV'):
    #save_to_csv('transactions.csv')

# Display accumulated rows
#if st.session_state.data:
    #st.write("Accumulated Transactions:")
    #st.dataframe(pd.DataFrame(st.session_state.data))


import io

# Allow user to upload an existing transactions.csv file
uploaded_file = st.file_uploader("Upload your existing transactions.csv file", type=['csv'])

existing_data = pd.DataFrame()  # Initialize empty DataFrame for existing data

if uploaded_file is not None:
    existing_data = pd.read_csv(uploaded_file)
    st.success("Existing file loaded.")
    st.write("Loaded data:")
    st.dataframe(existing_data)

# Combine existing data with new session data
if st.session_state.data:
    session_df = pd.DataFrame(st.session_state.data)
    if not existing_data.empty:
        combined_data = pd.concat([existing_data, session_df], ignore_index=True)
    else:
        combined_data = session_df
else:
    combined_data = existing_data

# Provide a download button for the updated file
if not combined_data.empty:
    csv = combined_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Updated Transactions CSV",
        data=csv,
        file_name="transactions.csv",
        mime="text/csv"
    )
else:
    st.info("No data to download yet.")



st.sidebar.write("After you have input your values and have gotten your prediction,"
                 " please ensure you download the CSV file and send it to my mail")

st.sidebar.write("Email: junaidyakub28@gmail.com")