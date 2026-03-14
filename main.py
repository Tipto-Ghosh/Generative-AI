import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature = 0.7
)

model = ChatHuggingFace(llm = llm)
result = model.invoke("Who is the CEO of OpenAI?")
print(result.content)