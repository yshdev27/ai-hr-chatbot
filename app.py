import streamlit as st
import requests

st.title("HR Resource Query Chatbot 🤖")
user_query = st.text_input("Ask me to find an employee:")

if user_query:
    
    api_url = "http://localhost:8000/chat"
    
    with st.spinner("Searching for candidates..."):
        try:
            response = requests.post(api_url, json={"query": user_query}, timeout=125)
            response.raise_for_status()
            api_response = response.json()
            st.markdown(api_response['response'])
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the backend. Please ensure it's running. Error: {e}")