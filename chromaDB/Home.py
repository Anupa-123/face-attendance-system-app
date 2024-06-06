# st Home.py
import streamlit as st

st.set_page_config(page_title='Attendance System',layout='wide')

st.header('Attendance System using Face Recognition')

with st.spinner("Loading Models and Conneting to Redis db ..."):
    import Face_Rec

st.success('Model loaded sucesfully')
st.success('Redis db sucessfully connected')