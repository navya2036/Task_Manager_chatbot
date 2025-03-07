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

# Initialize event list
if "events" not in st.session_state:
    st.session_state.events = []

# Initialize reminder list
if "reminders" not in st.session_state:
    st.session_state.reminders = []

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

# Configure Gemini API (replace with your actual API key)
API_KEY = "AIzaSyB3uidq20tP_lUTFxoN9Mvq4mgRLDSQ3Bk"  # Replace with your Gemini API key
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

    if "{event_list}" in response:
        if st.session_state.events:
            event_list_str = "\n".join([f"- {event}" for event in st.session_state.events])
            return response.replace("{event_list}", event_list_str)
        else:
            return response.replace("{event_list}", "No events found.")

    if "{reminder_list}" in response:
        if st.session_state.reminders:
            reminder_list_str = "\n".join([f"- {reminder}" for reminder in st.session_state.reminders])
            return response.replace("{reminder_list}", reminder_list_str)
        else:
            return response.replace("{reminder_list}", "No reminders found.")

    return response

# Function to handle task addition
def add_task(task_description):
    st.session_state.tasks.append(task_description)
    return f"Task added successfully: {task_description}"

# Function to handle event addition
def add_event(event_description):
    st.session_state.events.append(event_description)
    return f"Event added successfully: {event_description}"

# Function to handle reminder addition
def add_reminder(reminder_description):
    st.session_state.reminders.append(reminder_description)
    return f"Reminder added successfully: {reminder_description}"

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

    # Check if the user is adding a task, event, or reminder
    if "add task" in prompt.lower():
        task_description = prompt.lower().replace("add task", "").strip()
        if task_description:
            response = add_task(task_description)
        else:
            response = "Please provide a task description."

    elif "add event" in prompt.lower():
        event_description = prompt.lower().replace("add event", "").strip()
        if event_description:
            response = add_event(event_description)
        else:
            response = "Please provide an event description."

    elif "add reminder" in prompt.lower():
        reminder_description = prompt.lower().replace("add reminder", "").strip()
        if reminder_description:
            response = add_reminder(reminder_description)
        else:
            response = "Please provide a reminder description."
    elif "show reminders" in prompt.lower():
        response = replace_placeholders("Your reminders: {reminder_list}")
    elif "show tasks" in prompt.lower():
        response = replace_placeholders("Your tasks: {task_list}")
    elif "show events" in prompt.lower():
        response = replace_placeholders("Your events: {event_list}")

    else:
        # Find the closest match from the dataset
        closest_response = find_closest_input(prompt, vectorizer, user_input_vectors, df)

        if closest_response:
            # Replace placeholders in the response
            response = replace_placeholders(closest_response)
        else:
            # If no relevant response is found, use Gemini to generate a response
            try:
                response = model.generate_content(prompt).text
            except Exception as e:
                response = f"Sorry, I couldn't generate a response. Error: {e}"

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
