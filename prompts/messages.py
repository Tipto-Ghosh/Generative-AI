from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate 

from dotenv import load_dotenv
load_dotenv()


# define the model
llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation",
    temperature = 0.8,
    max_new_tokens = 2000
)
model = ChatHuggingFace(llm = llm)

messages = [
    SystemMessage(content = "You are a helpful assistant."),
    HumanMessage(content = "Tell me about langchain")
]

result = model.invoke(messages)
messages.append(AIMessage(content = result.content))
print(messages)