from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser



llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation",
    temperature = 0.8,
    max_new_tokens = 2000
)
model = ChatHuggingFace(llm = llm)

# user_input = input("Enter prompt: ")
# result = model.invoke(user_input)
# print('-' * 30)
# print(result.content)


st.header("Research Paper summerizer")
paper_input = st.selectbox(
    "Select Paper name",
    ['Attention is all you need' , 'BERT: Pre-training of Deep Bidirectional Transformers']
)
style_input = st.selectbox(
    "Select explaination type",
    ["Beginner-Friendly" , "Technical" , "Code-oriented" , "Mathmatical"]
)
length_input = st.selectbox(
    "Selec explaination length",
    ["Short (1-2 paragraphs)" , "Medium (3-5 paragraphs)" , "Long(detailed explaination)"]
)

template = """ 
   Please summarize the research paper titled "{paper_input}" with the following specifications:
   Explanation Style: {style_input}  Explanation Length: {length_input}  
    1. Mathematical Details:  
      - Include relevant mathematical equations if present in the paper.  
      - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
    2. Analogies:  
      - Use relatable analogies to simplify complex ideas.  
   If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
   Ensure the summary is clear, accurate, and aligned with the provided style and length.
"""
# create prompt
prompt_template = PromptTemplate(
    template = template,
    input_variables = ['paper_input' , 'style_input' , 'length_input'],
    validate_template = True # validate all placeholder is filled or not
)


# format the prompt with the placeholders
formated_prompt = prompt_template.format(
    paper_input = paper_input,
    style_input = style_input,
    length_input = length_input
)


if st.button("Summarize"):
    # result = model.invoke(formated_prompt)
    # st.write(result.content)
    
    chain = prompt_template | model | StrOutputParser()
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(result)