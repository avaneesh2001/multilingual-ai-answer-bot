from dotenv import load_dotenv
import os
from src.config import MODELS
import requests
import time

import streamlit as st

OPENROUTER_KEY = st.secrets["OPENROUTER_KEY"]

class LLMCall:
    def __init__(self):
        API_KEY = OPENROUTER_KEY

        self.MODELS = MODELS
        self.URL = "https://openrouter.ai/api/v1/chat/completions"

        self.HEADERS = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
    
    def call_llm(self,prompt):
        for model in self.MODELS:
            try:
                data = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}]
                }

                response = requests.post(self.URL, headers=self.HEADERS, json=data, timeout=30)

                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]

                # Handle rate limit (429)
                elif response.status_code == 429:
                    print(f"Rate limited on {model}, trying next...")
                    time.sleep(1)
                    continue

                else:
                    print(f"Error from {model}: {response.text}")
                    continue

            except Exception as e:
                print(f"Exception on {model}: {e}")
                continue