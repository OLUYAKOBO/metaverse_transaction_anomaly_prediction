import streamlit as st
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
import os
import io
import pickle

def load_json():
    with open('sending_add.json', 'r') as f:
        sending_add = json.load(f)

    with open('receiving_add.json', 'r') as f:
        receiving_add = json.load(f)

    return sending_add, receiving_add

sending_add, receiving_add = load_json()

# Initialize session state for storing rows
if "data" not in st.session_state:
    st.session_state.data = []

# Function to save accumulated data to a DataFrame
def get_data_as_dataframe():
    if st.session_state.data:
        return pd.DataFrame(st.session_state.data)
    else:
        return pd.DataFrame(columns=['amount', 'login_frequency', 'session_duration', 'risk_score', 
                                     'sending_address', 'receiving_address', 'prediction'])

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

# Load encoder
def load_encoder():
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return encoder

encoder = load_encoder()

# Encode categorical features
def encode(df):
    cat_columns = ['transaction_type', 'location_region', 'purchase_pattern', 'age_group']
    encoded_values = encoder.transform(df[cat_columns])
    encoded_values = pd.DataFrame(encoded_values, columns=encoder.get_feature_names_out(cat_columns))

    df = pd.concat([df.drop(columns=cat_columns), encoded_values], axis=1)
    return df

df = encode(df)

# Load model
def load_model():
    with open("rand_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Predict transaction risk
prediction = model.predict(df)[0]

# Prediction logic
if st.button('*Click here*'):
    with st.spinner('Predicting... Please wait...'):
        if prediction == 0:
            st.success("This is a low-risk transaction")
        else:
            st.success("This is a high-risk transaction")

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

# Generate CSV for download
if st.button('Save and Download CSV'):
    data_df = get_data_as_dataframe()
    if not data_df.empty:
        csv_buffer = io.StringIO()
        data_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="Download Transactions CSV",
            data=csv_data,
            file_name="transactions.csv",
            mime="text/csv"
        )
        st.success("CSV is ready for download!")
    else:
        st.warning("No data available to save.")