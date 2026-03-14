from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()


# define the model
llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation",
    temperature = 0.8,
    max_new_tokens = 2000
)
# model = ChatHuggingFace(llm = llm)


chat_template = ChatPromptTemplate([
    ('system' , "You are a helpful {domain} expert"),
    ("human"  ,"Explain in simple terms, what is {topic}?")
    # SystemMessage(content = "You are a helpful {domain} expert"),
    # HumanMessage(content = "Explain in simple terms, what is {topic}?")
])

prompt = chat_template.invoke({
    "domain": "Machine Learning",
    "topic": "Random Forest"
})

print(prompt)