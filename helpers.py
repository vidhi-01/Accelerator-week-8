import requests
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(
    api_key=os.environ.get('GROQ_API_KEY'),
)

def fetch_models():
    exclude_models = ['distil-whisper-large-v3-en', 'whisper-large-v3','whisper-large-v3-turbo', "llama3-groq-8b-8192-tool-use-preview", "llama3-groq-70b-8192-tool-use-preview"]
    url = "https://api.groq.com/openai/v1/models"
    api_key = os.environ.get('GROQ_API_KEY')
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers).json()['data']
    if len(response) != 0:
         return [model['id'] for model in response if model['id'] not in exclude_models]
    else:
        print("Error in fetching models!")
        return []
    

def generate_content(selected_model, prompt):
    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content":prompt }],
                        model=selected_model,
                        )
    return chat_completion.choices[0].message.content