from Home import st

st.set_page_config(page_title='Predictions', layout='wide')
st.subheader('Real-Time Attendance System')

# Retrieve the data from Redis Database
import face_rec
redis_face_db = face_rec.retrive_data(name='academy:register')
st.dataframe(redis_face_db)





# Real Time Prediction