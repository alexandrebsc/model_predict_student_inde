import joblib
import numpy as np
import streamlit as st

st.write('### Idade')
input_age = float(st.slider('Selecione a idade do aluno',1,30,12))

st.write('### Fase')
input_level = float(st.slider('Selecione a fase do aluno',0,7,0))

st.write('### Ingressante')
input_first_year = st.radio('O aluno é ingressante?',['Sim','Não'], index=0)
ingressante_dict = {'Sim': 1., 'Não':0.}

st.write('### IAA')
input_iaa = float(st.number_input('Preencha o IAA do aluno',0.,10.,0.,0.01,"%.3f",key='iaa'))

st.write('### IAN')
input_ian = float(st.number_input('Preencha o IAN do aluno',0.,10.,0.,0.05,"%.2f",key='ian'))

st.write('### IEG')
input_ieg = float(st.number_input('Preencha o IEG do aluno',0.,10.,0.,0.01,"%.3f",key='ieg'))

tranformed_input = np.array([[
    input_age/30.,
    input_level/7.,
    input_ian/10.,
    input_iaa/10.,
    input_ieg/10.,
    1. if input_first_year else 0.
]])

if st.button('Enviar'):
    model = joblib.load('random_forest_regressor_predict_student_inde.pkl')
    pred_inde = model.predict(tranformed_input)[0]*10
    if pred_inde <= 5.506:
        st.error(f'### Quartzo! INDE previsto: {pred_inde}')
    if pred_inde > 5.506 and pred_inde <= 6.868:
        st.success(f'### Ágata! INDE previsto: {pred_inde}')
    if pred_inde > 6.868 and pred_inde <= 8.230:
        st.success(f'### Ametista! INDE previsto: {pred_inde}')
    if pred_inde > 8.230:
        st.success(f'### Topázio! INDE previsto: {pred_inde}')
        st.balloons()