# For creating a new environment
# python -m venv env

from itertools import zip_longest
from openai import OpenAI
import json
import streamlit as st
from streamlit_chat import message
import speech_recognition as sr
from gtts import gTTS
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage


# Define a function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text)
    tts.save("output.mp3")
    st.audio("output.mp3")


# Define a function to convert speech to text
def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = r.listen(source)
        st.write("Generating...")
    try:
        user_input = r.recognize_google(audio)
        st.text(f'YOU: {user_input}')
        return user_input
    except sr.UnknownValueError:
        st.write("Oops, I didn't get your audio, Please try again.")
        return None


# Initialize session state variable
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []  # Store past user inputs


# Access the ChatGPT API key
chatgpt_api_key = "sk-*****************************************************"

chat = ChatOpenAI(
    temperature=0.5,
    model_name="gpt-3.5-turbo",
    openai_api_key=chatgpt_api_key,
    max_tokens=100
)


# Define a function to generate response
def generate_response(user_query):
    # Build the list of messages
    zipped_messages = build_message_list(user_query)

    # Generate response using the chat model
    ai_response = chat(zipped_messages)

    response = ai_response.content

    return response


# Define a function to build a message list
def build_message_list(user_query):
    """
    Build a list of messages including system, human, and AI messages.
    """
    # Start zipped_messages with the SystemMessage
    zipped_messages = [SystemMessage(
        content="""Your name is Voice Text Assistant. You are an AI Technical Expert for Artificial Intelligence, here to guide and assist students with their AI-related questions and concerns. Please provide accurate and helpful information, and always maintain a polite and professional tone.

(Your introductory message here...)"""
    )]

    # Zip together the past and generated messages
    for human_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
        if human_msg is not None:
            zipped_messages.append(HumanMessage(
                content=human_msg))  # Add user messages
        if ai_msg is not None:
            zipped_messages.append(
                AIMessage(content=ai_msg))  # Add AI messages

    zipped_messages.append(HumanMessage(content=user_query))  # Add the latest user message

    return zipped_messages


# Set streamlit page configuration
st.set_page_config(page_title="Multilingual Speech Recognizer & AI Assistant")
st.title("💬 Multilingual Speech Recognizer & Artificial Intelligence Assistant")
st.caption("🔥 A streamlit Multilingual Speech Recognizer & Artificial Intelligence Assistant using GPT API 🔥")

MODEL_LIST = ["gpt-3.5-turbo", "gpt-4"]
selected_model = st.selectbox("Select a model", MODEL_LIST)
user_input = ""


# Define a function to display the conversation history for text input in newest to oldest order
def display_text_conversation_history():
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        if i < len(st.session_state["past"]):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '-text-user')
        message(st.session_state["generated"][i], key=str(i) + '-text')


# Define a function to display the conversation history for audio input in newest to oldest order
def display_audio_conversation_history():
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        if i < len(st.session_state["past"]):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '-audio-user')
        message(st.session_state["generated"][i], key=str(i) + '-audio')


a = st.sidebar.selectbox('Select one:', ["Intro", "Text", "Audio"])

if a == "Intro":
    st.image("https://media.giphy.com/media/doXBzUFJRxpaUbuaqz/giphy.gif")

elif a == "Text":
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if not chatgpt_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        client = OpenAI(api_key=chatgpt_api_key)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = client.chat.completions.create(model=selected_model, messages=st.session_state.messages)
        msg = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)

elif a == "Audio":
    if st.button('Ask me!'):
        user_input = speech_to_text()

        if user_input:
            # Append user query to past queries
            st.session_state.past.append(user_input)

            # Generate response
            output = generate_response(user_input)

            # Append AI response to generated responses
            # st.session_state.generated.append(output)

            # Display user and AI messages
            # st.text(f'YOU: {user_input}')
            text_to_speech(output)

            # Display the conversation history for audio input
            display_audio_conversation_history()
