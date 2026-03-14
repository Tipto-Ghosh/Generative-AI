import json
from pydantic import BaseModel, create_model
from typing import Optional, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Step 1: Load the schema from file
with open("review_schema.json", "r") as f:
    json_schema = json.load(f)

# Step 2: Dynamically create a Pydantic model from the JSON schema
def json_schema_to_pydantic(schema: dict) -> type[BaseModel]:
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])
    
    field_definitions = {}
    
    for field_name, field_info in properties.items():
        field_type = field_info.get("type")
        
        # Handle type mapping
        if isinstance(field_type, list):  # e.g. ["string", "null"] → Optional
            base_type = next(t for t in field_type if t != "null")
            py_type = {"string": str, "array": list, "integer": int, "number": float}[base_type]
            py_type = Optional[py_type]
            default = None
        elif field_type == "array":
            py_type = Optional[list[str]]
            default = []
        elif field_type == "string":
            # Check for enum
            if "enum" in field_info:
                py_type = Literal[tuple(field_info["enum"])]
            else:
                py_type = str
            default = ... if field_name in required_fields else None
        else:
            py_type = str
            default = ... if field_name in required_fields else None

        if field_name in required_fields:
            field_definitions[field_name] = (py_type, ...)
        else:
            field_definitions[field_name] = (py_type, default)

    return create_model(schema.get("title", "DynamicModel"), **field_definitions)

# define the model
llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-7B-Instruct",
    task = "text-generation",
    temperature = 0.1,
    max_new_tokens = 512
)
model = ChatHuggingFace(llm = llm)


# Step 3: Create the Pydantic model
Review = json_schema_to_pydantic(json_schema)

# Step 4: Build prompt with the schema injected
prompt = ChatPromptTemplate.from_messages([
    ("system", """Extract structured information from the review.
Return ONLY a valid JSON object that strictly follows this schema:

{schema}

No explanation, no markdown, no code blocks. Just raw JSON."""),
    ("human", "{review_text}")
])

# Step 5: Extraction function
def parse_review(review_text: str) -> BaseModel:
    messages = prompt.format_messages(
        schema=json.dumps(json_schema, indent=2),
        review_text=review_text
    )
    raw_output = model.invoke(messages)
    text = raw_output.content.strip()
    
    clean = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(clean)
    
    # Validate against the dynamically created Pydantic model
    return Review.model_validate(data)



review_text = """
Sarah here. Absolutely love this coffee maker! Brews perfectly every time.
Super easy to clean and looks great on the counter. 
The only downside is it's quite loud and the carafe lid is flimsy.
"""

result = parse_review(review_text)
print(result.model_dump())