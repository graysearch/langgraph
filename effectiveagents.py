import os
import openai
import json
from pydantic import BaseModel, Field
llm = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Prepare the prompt and messages for the chat completion
response = llm.chat.completions.create(
    model="gpt-4",  # You can change this to the desired model like 'gpt-3.5-turbo'
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How does Calcium CT score relate to high cholesterol?"},
    ],
)
#print(response.choices[0].message.content.strip())

class SearchQuery(BaseModel):
    search_query: str = Field(..., description="Query that is optimized for web search.")
    justification: str = Field(..., description="Why this query is relevant to the user's request.")

# Prepare a prompt instructing the assistant to return JSON matching our schema exactly
structured_prompt = """
Please provide your answer in JSON format exactly as follows:

{
  "search_query": "<optimized search query>",
  "justification": "<explanation of why this query is relevant>"
}

Question: How does Calcium CT score relate to high cholesterol?
"""

structured_response = llm.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": structured_prompt},
    ],
)

# Extract and clean the generated response content
structured_content = structured_response.choices[0].message.content.strip()

# Attempt to parse the assistant's JSON output and validate it with Pydantic
try:
    structured_json = json.loads(structured_content)
    validated_output = SearchQuery.parse_obj(structured_json)
    print("Structured Output:")
    print("Search Query:", validated_output.search_query)
    print("Justification:", validated_output.justification)
except Exception as e:
    print("Error parsing or validating structured output:", e)
    print("Received response:", structured_content)