#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph Agent Example

This script demonstrates how to build an agent using LangGraph.
The agent can both converse with the user and use tools to perform actions.
"""

import os
import json
import datetime
import re
import math
from typing import Dict, Any, List, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# Load environment variables from .env file (for API keys)
load_dotenv()

# Initialize OpenAI client
import openai
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Define tools that our agent can use
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: A string containing a mathematical expression
        
    Returns:
        The result of the calculation
    """
    try:
        # Use safer eval with math functions
        allowed_names = {
            k: v for k, v in math.__dict__.items() 
            if not k.startswith('__')
        }
        
        # Evaluate the expression
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result of {expression} = {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


def get_current_time() -> str:
    """
    Get the current date and time.
    
    Returns:
        A string with the current date and time
    """
    now = datetime.datetime.now()
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


def get_weather(location: str) -> str:
    """
    Get the weather for a location (simulated).
    
    Args:
        location: City or location to get weather for
        
    Returns:
        Weather information for the specified location
    """
    # This is a mock implementation - in a real app, you would call a weather API
    weathers = ["sunny", "cloudy", "rainy", "snowy", "windy"]
    temps = range(0, 35)
    
    import random
    weather = random.choice(weathers)
    temp = random.choice(temps)
    
    return f"Weather in {location}: {weather} with a temperature of {temp}°C"


# Define states and functions for our agent
def process_input(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process user input and add it to conversation history.
    """
    # Initialize history if it doesn't exist
    if "history" not in state:
        state["history"] = []
    
    # Get user input
    user_input = state.get("input", "")
    
    # Add user message to history
    state["history"].append({"role": "user", "content": user_input})
    
    return state


def decide_action(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze user request and decide whether to use a tool or respond directly.
    """
    # Access the latest user message
    latest_message = state["history"][-1]["content"]
    
    # Create a prompt for the LLM to decide the action
    messages = [
        {"role": "system", "content": """You are an assistant that can use tools or respond directly. 
         Analyze the user's request and decide what to do.
         
         Available tools:
         - calculator: For mathematical calculations. Example: "What is 25 * 16?"
         - get_current_time: To get the current date and time. Example: "What time is it?"
         - get_weather: To check weather for a location. Example: "What's the weather in Paris?"
         
         If a tool is needed, respond in this format:
         TOOL: tool_name
         ARGS: {"arg_name": "value"}
         
         For example:
         TOOL: calculator
         ARGS: {"expression": "25 * 16"}
         
         Or:
         TOOL: get_weather
         ARGS: {"location": "Paris"}
         
         If no tool is needed, respond with:
         DIRECT_RESPONSE
         """},
    ]
    
    # Add relevant history for context
    history_to_include = state["history"][-3:] if len(state["history"]) > 3 else state["history"]
    messages.extend(history_to_include)
    
    # Get LLM decision
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
    )
    decision = response.choices[0].message.content.strip()
    
    # Parse the decision
    if "TOOL:" in decision:
        # Extract tool name and arguments
        tool_match = re.search(r"TOOL: (\w+)", decision)
        args_match = re.search(r"ARGS: ({.*})", decision)
        
        if tool_match and args_match:
            tool_name = tool_match.group(1)
            try:
                args = json.loads(args_match.group(1))
                
                # Set the action state
                state["action"] = "use_tool"
                state["tool_name"] = tool_name
                state["tool_args"] = args
            except json.JSONDecodeError:
                # If JSON parsing fails, fall back to direct response
                state["action"] = "respond_directly"
        else:
            state["action"] = "respond_directly"
    else:
        state["action"] = "respond_directly"
    
    return state


def use_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the selected tool and capture the result.
    """
    tool_name = state["tool_name"]
    tool_args = state["tool_args"]
    result = None
    
    # Execute the requested tool
    if tool_name == "calculator" and "expression" in tool_args:
        result = calculator(tool_args["expression"])
    elif tool_name == "get_current_time":
        result = get_current_time()
    elif tool_name == "get_weather" and "location" in tool_args:
        result = get_weather(tool_args["location"])
    else:
        result = f"Error: Invalid tool '{tool_name}' or missing required arguments"
    
    # Store the result
    state["tool_result"] = result
    
    # Add to history as a function message
    state["history"].append({"role": "function", "name": tool_name, "content": result})
    
    return state


def generate_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a final response, either directly or based on tool results.
    """
    # Prepare messages for the API call
    messages = [
        {"role": "system", "content": "You are a helpful, friendly assistant that can use tools to assist the user."}
    ]
    
    # Add conversation history
    messages.extend(state.get("history", []))
    
    # If we used a tool, add a prompt to incorporate the tool results
    if state.get("action") == "use_tool" and "tool_result" in state:
        messages.append({
            "role": "system", 
            "content": "Based on the tool output, provide a helpful and informative response to the user's request."
        })
    
    # Generate the response
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
    )
    
    # Extract and add the response to state
    assistant_message = response.choices[0].message.content.strip()
    state["response"] = assistant_message
    
    # Add assistant response to history
    state["history"].append({"role": "assistant", "content": assistant_message})
    
    return state


def router(state: Dict[str, Any]) -> Literal["use_tool", "generate_response"]:
    """
    Determine which node to call next based on the action decision.
    """
    return state["action"]


def build_graph() -> StateGraph:
    """
    Build the agent graph with tool integration.
    """
    # Create a new graph
    graph = StateGraph(dict)
    
    # Add nodes
    graph.add_node("process_input", process_input)
    graph.add_node("decide_action", decide_action)
    graph.add_node("use_tool", use_tool)
    graph.add_node("generate_response", generate_response)
    
    # Set the entry point
    graph.set_entry_point("process_input")
    
    # Add basic edges
    graph.add_edge("process_input", "decide_action")
    
    # Add conditional edge from decide_action
    graph.add_conditional_edges(
        "decide_action",
        router,
        {
            "use_tool": "use_tool",
            "respond_directly": "generate_response"
        }
    )
    
    # Connect remaining nodes
    graph.add_edge("use_tool", "generate_response")
    graph.add_edge("generate_response", END)
    
    return graph


def save_mermaid_diagram():
    """
    Create a Mermaid diagram definition and save it to an HTML file.
    """
    mermaid_code = """
    graph TD
        START --> process_input
        process_input --> decide_action
        decide_action -->|use_tool| use_tool
        decide_action -->|respond_directly| generate_response
        use_tool --> generate_response
        generate_response --> END
    """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LangGraph Agent Visualization</title>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{ startOnLoad: true }});
        </script>
    </head>
    <body>
        <h1>LangGraph Agent - Flow Visualization</h1>
        <div class="mermaid">
        {mermaid_code}
        </div>
    </body>
    </html>
    """
    
    with open("agent_visualization.html", "w") as f:
        f.write(html_content)
    
    print("Saved Mermaid diagram to 'agent_visualization.html'")


def main():
    """Run the LangGraph agent example."""
    print("\n--- LangGraph Agent Example ---\n")
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in a .env file or export it to your environment.")
        return
    
    # Build and compile the graph
    graph = build_graph()
    compiled_graph = graph.compile()
    
    # Save a visualization
    save_mermaid_diagram()
    
    # Initialize conversation history
    state = {"history": []}
    
    # Interactive conversation loop
    print("Chat with the agent. You can ask for calculations, weather, or the current time.")
    print("Type 'exit' to end the conversation.\n")
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nGoodbye!")
            break
            
        # Prepare state for this turn
        current_state = state.copy()
        current_state["input"] = user_input
        
        # Run the graph
        result = compiled_graph.invoke(current_state)
        
        # Update the persistent state with new history
        state["history"] = result["history"]
        
        # Display the response
        print(f"Agent: {result['response']}")
        
    print("\n--- End of Example ---\n")


if __name__ == "__main__":
    main()