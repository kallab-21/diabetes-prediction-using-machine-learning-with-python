# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:18:27 2022

@author: kal ab
"""

import numpy as np
import pickle
import streamlit as st


loaded_model=pickle.load(open('E:/deploying folder new/trained_model.sav','rb')) 

def diabetes_prediction(input_data):
    
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'

    else:
      return 'The Person is diabetic'
  
def main():
   
    st.title('diabtes prediction web app')
    
    # getting input from users
    
    
    HighBP =st.text_input('high blood pressure value')
    HighChol=st.text_input('high cholestrol value')
    CholCheck=st.text_input('cholestrol check value')
    BMI=st.text_input('BMI value')
    Smoker=st.text_input('smoker or not')
    Stroke=st.text_input('stroke value')
    HeartDiseaseorAttack=st.text_input('HeartDiseaseorAttack value')
    PhysActivity=st.text_input('PhysActivity value')
    Fruits=st.text_input('Fruits value')
    Veggies=st.text_input('Veggies value')
    HvyAlcoholConsump=st.text_input('HvyAlcoholConsump value')
    AnyHealthcare=st.text_input('AnyHealthcare value')
    NoDocbcCost=st.text_input('NoDocbcCost value')
    GenHlth=st.text_input('General Health value')
    MentHlth=st.text_input('Mental Health value')
    PhysHlth=st.text_input('Physical Health value')
    DiffWalk=st.text_input('DiffWalk value')
    Sex=st.text_input('Sex of the person value')
    Age=st.text_input('Age of the person')
    Education=st.text_input('Education value')
    Income=st.text_input('Income value')
    
    diagnosis=''
    
    if st.button('diabetes test result'):
        diagnosis=diabetes_prediction([HighBP,HighChol,CholCheck,BMI,Smoker,Stroke,HeartDiseaseorAttack,PhysActivity,Fruits,
        Veggies,HvyAlcoholConsump,AnyHealthcare,NoDocbcCost,GenHlth,MentHlth,PhysHlth,
        DiffWalk,Sex,Age,Education,Income])
    st.success(diagnosis)

if __name__=='__main__':
    main()

           
        