# st real time attendance
import streamlit as st
from Home import Face_Rec
from streamlit_webrtc import webrtc_streamer
import av
import time

st.set_page_config(page_title='Predictions')
st.subheader('Real-Time Attendance System')

# Use the fixed collection name for retrieval
name = 'academy-register'

# Retrieve the data from ChromaDB
with st.spinner('Retrieving Data from ChromaDB ...'):
    chromadb_face_db = Face_Rec.retrieve_data(name='academy-register')
    st.dataframe(chromadb_face_db)

st.success("Data successfully retrieved from ChromaDB")

# Time settings
waitTime = 30  # time in seconds
setTime = time.time()
realtimepred = Face_Rec.RealTimePred()  # real-time prediction class

# Real-Time Prediction with Streamlit WebRTC
def video_frame_callback(frame):
    global setTime
    img = frame.to_ndarray(format="bgr24")  # 3D numpy array
    pred_img = realtimepred.face_prediction(img, chromadb_face_db, 'facial_features', ['Name', 'Role'], thresh=0.5)
    timenow = time.time()
    difftime = timenow - setTime
    if difftime >= waitTime:
        realtimepred.saveLogs_chromadb()
        setTime = time.time()  # reset time
        print('Save Data to ChromaDB')
    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

webrtc_streamer(key="realtimePrediction", video_frame_callback=video_frame_callback)
