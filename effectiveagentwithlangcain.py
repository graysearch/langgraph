import os
import json
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

#source https://www.youtube.com/watch?v=aHCDrAbH_go&t=668s&ab_channel=LangChain
#Source code - https://mirror-feeling-d80.notion.site/Workflow-And-Agents-17e808527b1780d792a0d934ce62bee6
# Define the structured output schema with Pydantic.
class SearchQuery(BaseModel):
    search_query: str = Field(..., description="Query that is optimized for web search.")
    justification: str = Field(..., description="Why this query is relevant to the user's request.")

# Custom wrapper to simulate structured output.
class StructuredLLM:
    def __init__(self, llm, schema):
        self.llm = llm
        self.schema = schema

    def invoke(self, prompt: str):
        # Construct a prompt that instructs the assistant to return JSON in a specific format.
        structured_prompt = f"""
Please provide your answer in JSON format exactly as follows:

{{
  "search_query": "<optimized search query>",
  "justification": "<explanation of why this query is relevant>"
}}

Question: {prompt}
"""
        # Query the LLM using LangChain's predict_messages.
        response = self.llm.predict_messages([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=structured_prompt)
        ])
        response_content = response.content.strip()
        try:
            # Attempt to parse the response as JSON.
            response_json = json.loads(response_content)
        except Exception as e:
            raise ValueError(f"Response is not valid JSON: {response_content}") from e
        # Validate and return the structured output using the Pydantic model.
        return self.schema.parse_obj(response_json)

# Initialize the LangChain OpenAI Chat model.
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Create an instance of the custom structured LLM.
structured_llm = StructuredLLM(llm, SearchQuery)

# Invoke the structured LLM.
output = structured_llm.invoke("How does Calcium CT score relate to high cholesterol?")

# Print the structured output.
print(output.search_query)
print(output.justification)

# Define a tool
def multiply(a: int, b: int) -> int:
    return a * b

# Augment the LLM with tools
llm_with_tools = llm.bind_tools([multiply])

# Invoke the LLM with input that triggers the tool call
msg = llm_with_tools.invoke("What is 2 times 3?")

# Get the tool call
msg.tool_calls