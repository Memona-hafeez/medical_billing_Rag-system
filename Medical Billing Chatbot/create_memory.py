import pandas as pd
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
import os

data_path = r"D:\VS CODE\Medical Billing Chatbot\data\healthcare_dataset.csv"
df = pd.read_csv(data_path)
df.drop_duplicates(inplace=True)
def row_to_text(row):
    #"""Convert a table row into a natural language description."""
    text = (
        f"Patient {row['Name']} is a {row['Age']} year old {row['Gender']} with blood type {row['Blood Type']}. "
        f"They were diagnosed with {row['Medical Condition']} and admitted on {row['Date of Admission']} at {row['Hospital']}. "
        f"The attending doctor was {row['Doctor']}, and the patient was assigned to room {row['Room Number']} under a(n) {row['Admission Type']} admission. "
        f"They were discharged on {row['Discharge Date']}. "
        f"Insurance coverage was provided by {row['Insurance Provider']}, with a billing amount of ${row['Billing Amount']:.2f}. "
        f"Medication administered was {row['Medication']}, and the test results were reported as {row['Test Results']}."
    )
    return text
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
df["Length of Stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days
df['text_representation'] = df.apply(row_to_text, axis=1)
documents = df['text_representation'].tolist()

groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_texts(documents, embedding_model)
db.save_local(DB_FAISS_PATH)
