import openai
import json
import os
import streamlit as st
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
import cred
import time

os.environ['OPENAI_API_KEY'] = cred.api_key

documents = SimpleDirectoryReader('/Users/trishanu/Downloads/Phycology_LLM').load_data()
index = GPTVectorStoreIndex(documents)

class Chatbot:
    def __init__(self, api_key, index):
        self.index = index
        openai.api_key = api_key
        self.chat_history = []

    def generate_response(self, user_input):
        prompt = "\n".join([f"{message['role']}: {message['content']}" for message in self.chat_history[-5:]])
        prompt += f"\nUser: {user_input}"
        query_engine = index.as_query_engine()
        response=query_engine.query(user_input)

        message = {"role": "assistant", "content": response.response}
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append(message)
        return message

    def load_chat_history(self, filename):
        try:
            with open(filename, 'r') as f:
                self.chat_history = json.load(f)
        except FileNotFoundError:
            pass

    def save_chat_history(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.chat_history, f)

bot = Chatbot(cred.api_key, index=index)
bot.load_chat_history("chat_history.json")

st.title('Psychology Chat Bot')

user_input = st.text_input("You: ", "")
if st.button("Send"):
    while True:
        try:
            response = bot.generate_response(user_input)
            break
        except openai.RateLimitError:
            print("Rate limit exceeded, waiting for 1 minute...")
            time.sleep(60)
    st.write(f"Bot: {response['content']}")