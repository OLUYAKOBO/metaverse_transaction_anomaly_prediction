import streamlit as st
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import StandardScaler
#import datetime as dt
import numpy as np

encoder = pkl.load(open('encoder.pkl','rb'))


import json
with open('sending_add.json','r') as f:
    sending_add = json.load(f)


with open('receiving_add.json','r') as f:
    receiving_add = json.load(f)


st.title(" *Metaverse Transaction Anomaly Prediction* ")
st.write("This application predicts the anomaly of the metaverse transaction")

st.header("*Enter your details below*")

def transac_details():
    c1,c2 =  st.columns(2)
    with c1:
        amount = st.number_input('*Transaction amount*')
        login_frequency = st.number_input("*Log in frequency*", 1,10,3)
        session_duration = st.number_input("*Session duration*", )
        receiving_address = st.text_input('*Destination address*')
        sending_address = st.text_input('*Your address*')
        
    with c2:
        transaction_type = st.selectbox('*What type of transaction do you want to carry out?*',
                                        (['purchase','transfer','sale']))
        location_region = st.selectbox('*Which location are you making this transaction from*',
                                       (['Europe','Asia','South America','North America','Africa']))
        purchase_pattern = st.selectbox('*Select your typical purchasing behavior in the metaverse?*',
                                        (['focused','high_value','random']))
        age_group = st.selectbox('*Select your metaverse experience level*',(['established','veteran','new']))
        risk_score = st.number_input('*Risk score*')

    sending_address = sending_add.get(sending_address,0)
    receiving_address = receiving_add.get(receiving_address,0)
        
        
    feat = np.array([amount,login_frequency,session_duration,risk_score,sending_address,receiving_address,
                     location_region,purchase_pattern,age_group,transaction_type]).reshape(1,-1)
    cols = ['amount','login_frequency','session_duration','risk_score','sending_address_encoded',
            'receiving_address_encoded','location_region','purchase_pattern','age_group','transaction_type']   
        
    feat1 = pd.DataFrame(feat, columns=cols)
    return feat1    
df = transac_details()

def encode(df):
    cat_columns = ['transaction_type', 'location_region', 'purchase_pattern', 'age_group']
    encoded_values = encoder.transform(df[cat_columns])
    encoded_values = pd.DataFrame(encoded_values,columns=encoder.get_feature_names_out(cat_columns))

    df = pd.concat([df.drop(columns = cat_columns),encoded_values],axis=1)
    return df
df = encode(df)

#st.write(df)

import pickle
model = pickle.load(open('rand_model.pkl','rb'))

prediction = model.predict(df)[0]
#st.write(prediction)

import time

if st.button('*Click here*'):
    time.sleep(10)
    with st.spinner('Predicting... Please wait...'):
        if prediction == 0:
            st.success("This is a low risk transaction")
        else:
            st.success("This is a high risk transaction")
    

