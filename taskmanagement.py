import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up Streamlit page
st.set_page_config(page_title="Task Management Chatbot", layout="centered")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize task list
if "tasks" not in st.session_state:
    st.session_state.tasks = []

# Load the dataset
csv_url = "large_task_data.csv"
try:
    df = pd.read_csv(csv_url)
except Exception as e:
    st.error(f"Failed to load the CSV file. Error: {e}")
    st.stop()

# Preprocess the data
df = df.fillna("")
df['user_input'] = df['user_input'].str.lower()
df['response'] = df['response'].str.lower()

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
user_input_vectors = vectorizer.fit_transform(df['user_input'])

# Configure Gemini API 
API_KEY = "AIzaSyB3uidq20tP_lUTFxoN9Mvq4mgRLDSQ3Bk"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Function to find the closest user input using cosine similarity
def find_closest_input(user_query, vectorizer, user_input_vectors, df):
    query_vector = vectorizer.transform([user_query.lower()])
    similarities = cosine_similarity(query_vector, user_input_vectors).flatten()
    best_match_index = similarities.argmax()
    best_match_score = similarities[best_match_index]
    if best_match_score > 0.3:  # Threshold for similarity
        return df.iloc[best_match_index]['response']
    else:
        return None

# Function to replace placeholders with actual data
def replace_placeholders(response):
    if "{task_list}" in response:
        if st.session_state.tasks:
            task_list_str = "\n".join([f"- {task}" for task in st.session_state.tasks])
            return response.replace("{task_list}", task_list_str)
        else:
            return response.replace("{task_list}", "No tasks found.")
    return response

# Streamlit app
st.title("Task Management Chatbot 📝")
st.write("Welcome to the Task Management Chatbot! How can I assist you with your tasks, reminders, or events?")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your query here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Find the closest match from the dataset
    closest_response = find_closest_input(prompt, vectorizer, user_input_vectors, df)
    
    if closest_response:
        # Replace placeholders in the response
        closest_response = replace_placeholders(closest_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": closest_response})
        with st.chat_message("assistant"):
            st.markdown(closest_response)
    else:
        # If no relevant response is found, use Gemini to generate a response
        try:
            response = model.generate_content(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            with st.chat_message("assistant"):
                st.markdown(response.text)
        except Exception as e:
            st.error(f"Sorry, I couldn't generate a response. Error: {e}")
