import streamlit
import streamlit as st
import joblib
import pandas as pd
import numpy as np

df=pd.read_csv('Training.csv')
description=pd.read_csv('description.csv')
precautions=pd.read_csv('precautions_df.csv')
medicine=pd.read_csv('medications.csv')
workout=pd.read_csv('workout_df.csv')
diets=pd.read_csv('diets.csv')
symp=pd.read_csv('symtoms_df.csv')
symp.drop(columns=['Unnamed: 0'],inplace=True)
symp.dropna(inplace=True)
symp.drop_duplicates(inplace=True)
symp_weights=pd.read_csv('Symptom-severity.csv')





encoder=joblib.load('encoder.joblib')
model=joblib.load('model.joblib')



st.title('Medicine Recommendation System')

symptoms=df.columns

symptoms=symptoms[0:]

symptoms=symptoms[:131]

symptoms=st.multiselect('choose your symptoms',symptoms)

clicked=st.button('Predict')

listed_symptoms=df.columns

symptoms_dict={}
for i in range(len(listed_symptoms)):
    symptoms_dict[listed_symptoms[i]]=i

def set_symptoms_arr(symptoms_dict,listed_sympotms,symptoms):
    symptoms_arr=np.zeros(len(listed_sympotms)-2)
    for symp in symptoms:
        if symp in symptoms_dict:
            symptoms_arr[symptoms_dict[symp]]=1
    return symptoms_arr

def preprocess(temp):
    my_str=''
    for i in temp:
        if i=="'" or i==']' or i=='[':
            continue
        my_str+=i
    medicines=my_str.split(',')
    medicines=[i.strip() for i in medicines]
    return medicines

def get_symptoms(symp,dis):
    symptoms_set=[]
    temp=symp[symp['Disease']==dis][['Symptom_1','Symptom_2','Symptom_3','Symptom_4']]
    for col in temp.columns:
        for i in temp[col]:
            if(type(i)!=float):
                if i.strip()=='dischromic _patches':
                    symptoms_set.append('dischromic_patches')
                    continue
                symptoms_set.append(i.strip())
    symptoms_set=set(symptoms_set)
    symptoms_set=list(symptoms_set)
    return symptoms_set
def get_score(symp_weights,symps):
    score=0;
    for i in symps:
        score+=symp_weights[symp_weights['Symptom']==i]['weight'].values[0]
    return score


input_for_model=set_symptoms_arr(symptoms_dict,listed_symptoms,symptoms)

input_for_model=input_for_model.reshape(1,131)

if clicked:
    prediction=model.predict(input_for_model)
    disease=encoder.inverse_transform(prediction)[0]
    st.write(disease)
    st.header(f'Main Symptoms of {disease}')
    my_symp=get_symptoms(symp,disease)
    for item in my_symp:
        st.write(item)
    total_score=get_score(symp_weights,my_symp)
    my_score=get_score(symp_weights,symptoms)
    # st.header(f'Your Severity Score is {int((my_score/total_score)*100)}%')

    st.header('Description')
    desc=description[description['Disease']==disease]['Description'].values
    st.write(desc[0])
    st.header('Precautions')
    prec=precautions[precautions['Disease']==disease][['Precaution_1','Precaution_2','Precaution_3','Precaution_4']]
    for item in prec.values[0]:
        st.write(item)
    st.header('Medicines')
    med=medicine[medicine['Disease']==disease]['Medication']
    medicines=preprocess(med.values[0])
    for item in medicines:
        st.write(item)
    st.header('Workouts')
    workout_list=workout[workout['disease']==disease]['workout'].values
    for item in workout_list:
        st.write(item)
    st.header('Diet')
    diet=diets[diets['Disease']==disease]['Diet'].values[0]
    diet=preprocess(diet)
    for item in diet:
        st.write(item)



