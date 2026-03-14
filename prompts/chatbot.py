from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate 
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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

chat_history = [
    SystemMessage(content = "You are a helpful AI assistant.")
]

# run loop for chat
while True:
    user_input = input("You: ")
    if user_input == 'exit':
        break
    
    # store the chat
    chat_history.append(HumanMessage(content = user_input))
    
    # result = model.invoke(user_input)
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content))
    print(f"Model:" , result.content)
    
print("= " * 50)
print(chat_history)