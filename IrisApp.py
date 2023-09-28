# -*- coding: utf-8 -*-


import numpy as np
import pickle
import streamlit as st
loaded_model = pickle.load(open('trained_modelKNN.sav','rb'))

#creating function
def irisplant_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    return prediction
    

    
    
def main():
    
    st.title('Iris Plant Prediction Using KNN')

    SepalLengthCm=st.slider('Set the Sepal Lenght Cm',min_value=0.000, step=0.1, max_value=10.0, format="%1f")
    SepalWidthCm=st.slider('Set the Sepal Width Cm',min_value=0.000, step=0.1, max_value=10.0, format="%1f")
    PetalLengthCm=st.slider('Set the Petal Lenght Cm',min_value=0.000, step=0.1, max_value=10.0, format="%1f")
    PetalWidthCm=st.slider('Set the Petal Width cm',min_value=0.000, step=0.1, max_value=10.0, format="%1f")  
    
   
    species = ''
    if st.button('Species result'):
        species = irisplant_prediction([SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm])
        
    st.success(species)    
    


if __name__ == '__main__':
     main()    
    
    
