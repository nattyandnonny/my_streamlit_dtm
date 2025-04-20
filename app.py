#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 10:44:43 2025

@author: tanchanokkomonnak
"""

#import library
import streamlit as st
import numpy as np
import pickle

#lo
with open('dtm_trained_model.pkl', 'rb') as f:
    dtm_model = pickle.load(f)
    
#application 
st.title("Iris flower Classification")
st.write("Enter the feature of the irirs flower")

#input fields
sepal_lenght = st.slider("Sepal Length (cm)",4.0,8.0,5.1)
sepal_width = st.slider("Sepal Width (cm)",2.0,4.5,3.5)
petal_lenght = st.slider("Petal Length (cm)",1.0,7.0,1.4)
petal_width = st.slider("Petal Width (cm)",0.1,2.5,0.2)

#perdict button
if st.button("Predict"):
    input_data = np.array([[sepal_lenght, sepal_width, sepal_lenght, sepal_width]])
    prediction = dtm_model.predict(input_data)
    species = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"The Predicted Species is: **{Species[prediction[0]]}**")
