import streamlit as st
import requests
import pandas as pd
import json
import os
from dotenv import load_dotenv
import time

load_dotenv()

st.set_page_config(
    page_title="Medical Billing Chatbot",
    page_icon="🏥",
    layout="wide"
)

API_URL = "http://localhost:8000/query" 

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0083B8;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
    }
    .chat-message.user {
        background-color: #E8F4F8;
    }
    .chat-message.assistant {
        background-color: #F0F2F6;
    }
    .chat-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-content {
        flex: 1;
    }
    .stTextInput {
        margin-top: 1rem;
    }
    .info-box {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Healthcare Data Assistant</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.title("Dataset Information")
    st.markdown("""
    ### Healthcare Dataset
    This chatbot can answer questions about a dataset with 50,000 patient records containing:
    
    - Patient demographics
    - Medical conditions
    - Admission details
    - Billing information
    - Insurance data
    - Treatment information
    
    ### Example Questions
    Try asking:
    - "What is the average billing amount for heart disease?"
    - "How many patients were admitted in July 2023?"
    - "Which insurance provider covers the most patients?"
    - "What medications are commonly prescribed for diabetes?"
    - "What's the total billing amount for emergency admissions?"
    """)
    
    if st.checkbox("Show Dataset Statistics"):
       try:
           response = requests.get("http://localhost:8000/dataset/stats", timeout=30)
           if response.status_code==200:
               stats = response.json()
               st.metric("Total Patients" , stats["total_patients"])
               st.metric("Hospitals", stats["hospitals"])
               st.metric("Average Billing", f"${stats['average_billing']:.2f}")
               st.metric("Most Common Condition", stats["most_common_condition"])

               st.subheader("Admission Types")
               st.write(stats["admission_types"]) 
                
               st.subheader("Gender Distribution")
               st.write(stats["gender_distribution"])

           else:
                st.error(f"Error fetching stats: {response.status_code}")

       except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {str(e)}\n\nMake sure your FastAPI backend is running.")
            

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask a question about the healthcare dataset...")

if query:

    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            response = requests.post(
                API_URL,
                json={"query": query},
                timeout=120
            )
            
            if response.status_code == 200:
                answer = response.json().get("response", "Sorry, I couldn't process your request.")
                
                full_response = ""
                for chunk in answer.split(" "):
                    full_response += chunk + " "
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)
                
                message_placeholder.markdown(answer)
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                error_msg = f"Error: {response.status_code} - {response.text}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        except requests.exceptions.RequestException as e:
            error_msg = f"Connection error: {str(e)}\n\nMake sure your FastAPI backend is running at {API_URL}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8rem;">
    Medical Billing Chatbot • Powered by LangChain and Groq
</div>
""", unsafe_allow_html=True)