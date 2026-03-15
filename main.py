import os
from dotenv import load_dotenv
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# print(HF_API_KEY)
from huggingface_hub import login
login(HF_API_KEY)