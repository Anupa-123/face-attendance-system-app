# adding all info in streamlit 4
# st face rec 3
import numpy as np
import pandas as pd
import cv2
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
from datetime import datetime
import chromadb
import os
import uuid
from chromadb import HttpClient
import streamlit as st
from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection
from chromadb.utils import embedding_functions

'''
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                model_name="text-embedding-ada-002"
                )
'''

 # Configuration for embedding
embedding_config = {
    "api_key": "sk-proj-RUxzx4gx2DUtyDY8jqn0T3BlbkFJFN2Xhio4ZVIGOEcqFsyF",  # Replace with your actual OpenAI API key
    "model_name": "text-davinci-003",       # Replace with your chosen model
}


#metadata = {"hnsw:space": "cosine"}  # Using cosine similarity
collection_name = "academy-register"
embedding_function_name = "DefaultEmbeddingFunction"

# Connect to ChromaDB Client
hostname = 'localhost'
portnumber = 8000

configuration = {
    "client": "HttpClient",
    "host": hostname,
    "port": portnumber,
}
client = HttpClient(host=hostname, port=portnumber)
collection = client.get_or_create_collection(name=collection_name)

# Establish the connection using Streamlit
conn = st.connection(name="http_connection",
                     type=ChromadbConnection,
                     **configuration)


# Create the collection with specified metadata
metadata = {"hnsw:space": "cosine"}  # Using cosine similarity
collection_name = "academy-register"
embedding_function_name = "DefaultEmbeddingFunction"

#collection = HttpClient.get_or_create_collection(name='academy_register')
document_id = str(uuid.uuid4())
conn.get_collection_data(collection_name=collection_name,
                       attributes= ["metadatas", "embeddings", "documents"])

try:
    collection = client.create_collection(
        collection_name=collection_name,
        embedding_function_name=embedding_function_name,
        embedding_config=embedding_config,
        metadata=metadata
    )
    print("Collection created successfully.")
except Exception as e:
    print(f"Error while creating collection `{collection_name}`: {e}")


### get collection data
collection_name = "academy-register"
conn.get_collection_data(collection_name=collection_name,
                         attributes= ["metadata", "embeddings", "documents"])


####
def store_face_embedding(name, role, embedding):
    client = HttpClient(host=hostname, port=portnumber)
    collection = client.get_or_create_collection(name=collection_name)
    document_id = str(uuid.uuid4())  # Ensure a unique ID is generated
    metadata = {
        "name_role": f"{name}@{role}"
    }
    try:
        collection.add(ids=[document_id], metadatas=[metadata], embeddings=[embedding.tolist()])
        print("Document added successfully.")
    except Exception as e:
        print(f"Failed to add document: {e}")


import json  # Ensure to import json at the beginning of your script
def retrieve_data(name):
    try:
        collection = client.get_or_create_collection(name=name)
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


        #documents = collection.get()
        records = []
        print("Type of documents:", type(documents))  # Check the type of the returned object
        print("Content of documents:", documents['metadatas'])     # Log the actual content

        # If documents is a dictionary containing actual data:
        if isinstance(documents, dict):
            # Assuming the data is nested under some key, e.g., 'data'
            for doc in documents.get('data', []):
                try:
                    # Assuming each doc under 'data' is a properly JSON-encoded string
                    doc_data = json.loads(doc)
                    metadata = doc_data['metadata']
                    embedding = np.array(doc_data['embedding'], dtype=np.float32)
                    name_role = metadata['name_role']
                    name, role = name_role.split('@')
                    records.append({'Name': name, 'Role': role, 'facial_features': embedding})
                except json.JSONDecodeError:
                    print("Failed to decode document:", doc)
                except KeyError as e:
                    print("Key error:", e, "in document:", doc)
        return pd.DataFrame(records)
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return pd.DataFrame(columns=['Name', 'Role', 'facial_features'])

# Usage
dataframe = retrieve_data('academy_register')
print(dataframe)

'''
client = HttpClient(host=hostname, port=portnumber)
collection = client.get_or_create_collection("academy-register")

documents = collection.get()
#### Update collection
conn.update_collection_data(collection_name=collection_name,
                            ids=document_id,
                            documents=documents,
                            metadatas=metadata
                            )
'''

# Configure face analysis
faceapp = FaceAnalysis(name='buffalo_sc', root='insightface_model', providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

# ML Search Algorithm
def ml_search_algorithm(dataframe, feature_column, test_vector, name_role=['Name', 'Role'], thresh=0.5):
    dataframe = dataframe.copy()
    
    X_list = dataframe[feature_column].tolist()
    x = np.asarray(X_list)
    
    similar = pairwise.cosine_similarity(x, test_vector.reshape(1, -1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr
    
    data_filter = dataframe.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:
        data_filter.reset_index(drop=True, inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'
    return person_name, person_role

class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[], role=[], current_time=[])
    
    def reset_dict(self):
        self.logs = dict(name=[], role=[], current_time=[])
    
    def saveLogs_chromadb(self):
        dataframe = pd.DataFrame(self.logs)
        dataframe.drop_duplicates('name', inplace=True)
        for index, row in dataframe.iterrows():
            document_id = str(uuid.uuid4())
            name, role, ctime = row['name'], row['role'], row['current_time']
            if name != 'Unknown':
                metadata = {
                    "name_role": f"{name}@{role}@{ctime}",
                    "timestamp": ctime
                }
                collection.add(ids=[document_id], metadatas=[metadata])
        self.reset_dict()
    
    def face_prediction(self, test_image, dataframe, feature_column, name_role=['Name', 'Role'], thresh=0.5):
        current_time = str(datetime.now())
        results = faceapp.get(test_image)
        test_copy = test_image.copy()
        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, person_role = ml_search_algorithm(dataframe, feature_column, test_vector=embeddings, name_role=name_role, thresh=thresh)
            color = (0, 0, 255) if person_name == 'Unknown' else (0, 255, 0)
            cv2.rectangle(test_copy, (x1, y1), (x2, y2), color)
            text_gen = person_name
            cv2.putText(test_copy, text_gen, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            cv2.putText(test_copy, current_time, (x1, y2+10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)
        return test_copy

class RegistrationForm:
    def __init__(self):
        self.sample = 0
    def reset(self):
        self.sample = 0
    def get_embedding(self, frame):
        results = faceapp.get(frame, max_num=1)
        embeddings = None
        for res in results:
            self.sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            text = f"samples = {self.sample}"
            cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 2)
            embeddings = res['embedding']
        return frame, embeddings
    def save_data_in_chromadb(self, name, role):
        if name and name.strip() != '':
            key = f'{name}@{role}'
        else:
            return 'name_false'
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'
        x_array = np.loadtxt('face_embedding.txt', dtype=np.float32)
        received_samples = int(x_array.size / 512)
        x_array = x_array.reshape(received_samples, 512)
        x_mean = x_array.mean(axis=0).astype(np.float32).tobytes()
        document_id = str(uuid.uuid4())
        document = {"id": document_id, "name_role": key, "embedding": x_mean}

        #collection.add(ids=["document_id"], documents = ["document"])
        client = HttpClient(host=hostname, port=portnumber)
        collection = client.get_or_create_collection(name=collection_name)
        document_id = str(uuid.uuid4())  # Ensure a unique ID is generated
        metadata = {
        "name_role": f"{name}@{role}"
        }
        collection.add(ids=[document_id], metadatas=metadata, documents=["document"])

        os.remove('face_embedding.txt')
        self.reset()
        return True

