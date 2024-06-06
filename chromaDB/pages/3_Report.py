#st report.py
import streamlit as st
import pandas as pd
from Home import Face_Rec
from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection

st.set_page_config(page_title='Reporting', layout='wide')
st.subheader('Reporting')

# Retrieve logs data and show in Report.py
# Extract data from ChromaDB
name = 'academy-register'
def load_logs(name):
    logs_list = Face_Rec.retrieve_data(name)
    #logs_list = Face_Rec.collection.get(name)
    return logs_list

# Tabs to show the info
tab1, tab2, tab3 = st.tabs(['Registered Data', 'Logs', 'Attendance Report'])

with tab1:
    if st.button('Refresh Data'):
        # Retrieve the data from ChromaDB
        with st.spinner('Retrieving Data from ChromaDB ...'):
            chromadb_face_db = Face_Rec.retrieve_data('academy-register')
            configuration = {
                    "client": "HttpClient",
                    "host":"localhost",
                    "port": 8000,
                    }

            collection_name = "academy-register"

            conn = st.connection(name="http_connection",
                     type=ChromadbConnection,
                     **configuration)

            documents = conn.get_collection_data(collection_name)
            st.dataframe(documents)
            #st.dataframe(chromadb_face_db[['name', 'role']])

with tab2:
    if st.button('Refresh Logs'):
        st.write(load_logs(name=name))

with tab3:
    st.subheader('Attendance Report')

    # Load logs into attribute logs_list
    logs_list = load_logs(name=name)

    # Step 1: Convert the logs that are in list of bytes into list of string
    convert_byte_to_string = lambda x: x.encode('utf-8')
    logs_list_string = list(map(convert_byte_to_string, logs_list))

    st.write(logs_list_string)

    # Step 2: Split string by @ and create nested list
    split_string = lambda x: x.split('@')
    #logs_nested_list = list(map(split_string, logs_list_string))
    logs_nested_list = [log.split('@') for log in logs_list]

    # Ensure that logs_nested_list has the correct number of columns
    # Filter out any malformed entries that do not have exactly 3 elements
    logs_nested_list = [log for log in logs_nested_list if len(log) == 3]

    # Convert nested list info into DataFrame
    logs_df = pd.DataFrame(logs_nested_list, columns=['Name', 'Role', 'Timestamp'])

    # Step 3: Time-based analysis or Report
    logs_df['Timestamp'] = pd.to_datetime(logs_df['Timestamp'])
    logs_df['Date'] = logs_df['Timestamp'].dt.date

    # Step 3.1: Calculate In time and Out time
    report_df = logs_df.groupby(['Name', 'Role', 'Date']).agg(
        In_time=pd.NamedAgg(column='Timestamp', aggfunc='min'),
        Out_time=pd.NamedAgg(column='Timestamp', aggfunc='max')
    ).reset_index()

    report_df['In_time'] = pd.to_datetime(report_df['In_time'])
    report_df['Out_time'] = pd.to_datetime(report_df['Out_time'])
    report_df['Duration'] = report_df['Out_time'] - report_df['In_time']

    # Step 4: Marking person Present or Absent
    all_dates = report_df['Date'].unique()
    name_role = report_df[['Name', 'Role']].drop_duplicates().values.tolist()

    date_name_role_zip = []
    for dt in all_dates:
        for name, role in name_role:
            date_name_role_zip.append([dt, name, role])

    date_name_role_zip_df = pd.DataFrame(date_name_role_zip, columns=['Date', 'Name', 'Role'])
    date_name_role_zip_df = pd.merge(date_name_role_zip_df, report_df, on=['Date', 'Name', 'Role'], how='left')

    # Duration in Hours
    date_name_role_zip_df['Duration_seconds'] = date_name_role_zip_df['Duration'].dt.seconds
    date_name_role_zip_df['Duration_hours'] = date_name_role_zip_df['Duration_seconds'] / (60*60)

    def status_marker(x):
        if pd.Series(x).isnull().all():
            return 'Absent'
        elif x > 0 and x < 1:
            return 'Absent (Less than 1 hr)'
        elif x >= 1 and x < 4:
            return 'Half Day (less than 4 hr)'
        elif x >= 4 and x < 6:
            return 'Half Day'
        elif x >= 6:
            return 'Present'

    date_name_role_zip_df['Status'] = date_name_role_zip_df['Duration_hours'].apply(status_marker)

    st.dataframe(date_name_role_zip_df)
