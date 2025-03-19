#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph Agent with DeepSeek-style Thinking Tokens

This script implements a LangGraph agent that uses <Thinking> tokens
to show its reasoning process in a style similar to DeepSeek models.
"""

import os
import json
import datetime
import re
import math
import time
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
    
    # Initialize thinking tokens if it doesn't exist
    if "thinking_tokens" not in state:
        state["thinking_tokens"] = []
    
    # Get user input
    user_input = state.get("input", "")
    
    # Add user message to history
    state["history"].append({"role": "user", "content": user_input})
    
    return state


def thinking_decision(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the request and decide on an action using DeepSeek-style thinking tokens.
    """
    # Access the latest user message
    latest_message = state["history"][-1]["content"]
    
    # Create a prompt for the LLM to decide with thinking tokens
    messages = [
        {"role": "system", "content": """You are an assistant that demonstrates your reasoning with <Thinking> tokens.
        
         For the user's request, use <Thinking> tags to show your detailed reasoning process,
         then decide if the request requires any of these tools:
         
         - calculator: For mathematical calculations. Example: "What is 25 * 16?"
         - get_current_time: To get the current date and time. Example: "What time is it?"
         - get_weather: To check weather for a location. Example: "What's the weather in Paris?"
         
         Your response should follow this format:
         
         <Thinking>
         First, let me analyze what the user is asking for...
         [Your detailed reasoning about the problem]
         
         Based on my analysis, I need to... [explain your decision]
         </Thinking>
         
         TOOL: [tool_name or "none"]
         ARGS: [arguments in JSON format if needed]
         
         Example:
         <Thinking>
         The user is asking "What is 25 * 16?". This is clearly a mathematical calculation.
         I need to multiply 25 by 16, which requires the calculator tool.
         The input to the calculator should be the expression "25 * 16".
         </Thinking>
         
         TOOL: calculator
         ARGS: {"expression": "25 * 16"}
         """},
        {"role": "user", "content": latest_message}
    ]
    
    # Get the thinking and decision
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2,  # Lower temperature for more deterministic reasoning
    )
    
    full_response = response.choices[0].message.content.strip()
    
    # Extract thinking part
    thinking_match = re.search(r"<Thinking>(.*?)</Thinking>", full_response, re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
        state["thinking_tokens"].append({
            "type": "decision",
            "content": thinking
        })
    
    # Extract tool decision
    tool_match = re.search(r"TOOL: (\w+)", full_response)
    args_match = re.search(r"ARGS: ({.*})", full_response)
    
    if tool_match:
        tool_name = tool_match.group(1).strip().lower()
        
        if tool_name == "calculator" and args_match:
            try:
                args = json.loads(args_match.group(1))
                state["action"] = "use_tool"
                state["tool_name"] = "calculator"
                state["tool_args"] = args
            except json.JSONDecodeError:
                state["action"] = "respond_directly"
        
        elif tool_name == "get_current_time":
            state["action"] = "use_tool"
            state["tool_name"] = "get_current_time"
            state["tool_args"] = {}
        
        elif tool_name == "get_weather" and args_match:
            try:
                args = json.loads(args_match.group(1))
                state["action"] = "use_tool"
                state["tool_name"] = "get_weather"
                state["tool_args"] = args
            except json.JSONDecodeError:
                state["action"] = "respond_directly"
        
        elif tool_name == "none":
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
    
    # Add to thinking tokens
    state["thinking_tokens"].append({
        "type": "tool_execution",
        "content": f"Now I'll use the {tool_name} tool with these parameters: {tool_args}"
    })
    
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
    
    # Add to thinking tokens
    state["thinking_tokens"].append({
        "type": "tool_result",
        "content": f"The {tool_name} tool returned: {result}"
    })
    
    # Add to history as a function message
    state["history"].append({"role": "function", "name": tool_name, "content": result})
    
    return state


def thinking_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a final response with DeepSeek-style thinking tokens.
    """
    # Prepare messages for the API call
    if state.get("action") == "use_tool":
        # If we used a tool, include the tool result
        system_message = f"""You are a helpful assistant that demonstrates reasoning with <Thinking> tokens.
        
        The user asked: "{state["history"][0]["content"]}"
        
        You used the {state["tool_name"]} tool, which returned: "{state["tool_result"]}"
        
        First use <Thinking> tags to show your reasoning process, then provide your final answer.
        
        Format:
        <Thinking>
        [Your detailed reasoning about how to interpret the tool result and formulate a response]
        </Thinking>
        
        [Your final, concise response to the user]
        """
    else:
        # For direct responses
        system_message = f"""You are a helpful assistant that demonstrates reasoning with <Thinking> tokens.
        
        The user asked: "{state["history"][0]["content"]}"
        
        First use <Thinking> tags to show your reasoning process, then provide your final answer.
        
        Format:
        <Thinking>
        [Your detailed reasoning about how to answer this question]
        </Thinking>
        
        [Your final, concise response to the user]
        """
    
    messages = [
        {"role": "system", "content": system_message}
    ]
    
    # Add conversation history (without function messages)
    user_messages = [msg for msg in state["history"] if msg["role"] == "user"]
    messages.extend(user_messages)
    
    # Get response with thinking tokens
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,  # Higher temperature for more creative responses
    )
    
    full_response = response.choices[0].message.content.strip()
    
    # Extract thinking part
    thinking_match = re.search(r"<Thinking>(.*?)</Thinking>", full_response, re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
        state["thinking_tokens"].append({
            "type": "response",
            "content": thinking
        })
    
    # Extract the actual response (everything after </Thinking>)
    response_match = re.search(r"</Thinking>(.*)", full_response, re.DOTALL)
    if response_match:
        assistant_message = response_match.group(1).strip()
    else:
        # Fallback if format not followed
        assistant_message = full_response.replace("<Thinking>", "").replace("</Thinking>", "").strip()
    
    # Store the response
    state["response"] = assistant_message
    
    # Add assistant response to history
    state["history"].append({"role": "assistant", "content": assistant_message})
    
    return state


def router(state: Dict[str, Any]) -> Literal["use_tool", "thinking_response"]:
    """
    Determine which node to call next based on the action decision.
    """
    return state["action"]


def build_graph() -> StateGraph:
    """
    Build the agent graph with thinking tokens.
    """
    # Create a new graph
    graph = StateGraph(dict)
    
    # Add nodes
    graph.add_node("process_input", process_input)
    graph.add_node("thinking_decision", thinking_decision)
    graph.add_node("use_tool", use_tool)
    graph.add_node("thinking_response", thinking_response)
    
    # Set the entry point
    graph.set_entry_point("process_input")
    
    # Add basic edges
    graph.add_edge("process_input", "thinking_decision")
    
    # Add conditional edge from thinking_decision
    graph.add_conditional_edges(
        "thinking_decision",
        router,
        {
            "use_tool": "use_tool",
            "respond_directly": "thinking_response"
        }
    )
    
    # Connect remaining nodes
    graph.add_edge("use_tool", "thinking_response")
    graph.add_edge("thinking_response", END)
    
    return graph


def display_thinking_tokens(tokens, typing_speed=0.02, pause_between_tokens=0.5):
    """
    Display thinking tokens with a typing effect.
    """
    for token in tokens:
        # Format based on token type
        if token["type"] == "decision":
            print("\n\033[1;36m<Thinking>\033[0m")  # Cyan, bold
        elif token["type"] == "tool_execution":
            print("\n\033[1;33m<Thinking>\033[0m")  # Yellow, bold
        elif token["type"] == "tool_result":
            print("\n\033[1;32m<Thinking>\033[0m")  # Green, bold
        elif token["type"] == "response":
            print("\n\033[1;35m<Thinking>\033[0m")  # Magenta, bold
        
        # Display the content with typing effect
        for char in token["content"]:
            print(char, end="", flush=True)
            time.sleep(typing_speed)
        
        print("\n\033[1m</Thinking>\033[0m")  # Bold
        time.sleep(pause_between_tokens)


def main():
    """Run the LangGraph agent with DeepSeek-style thinking tokens."""
    print("\n--- LangGraph Agent with DeepSeek-style Thinking Tokens ---\n")
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in a .env file or export it to your environment.")
        return
    
    # Build and compile the graph
    graph = build_graph()
    compiled_graph = graph.compile()
    
    # Initialize conversation history
    state = {"history": [], "thinking_tokens": []}
    
    # Interactive conversation loop
    print("Chat with the agent. You can ask for calculations, weather, or the current time.")
    print("Type 'exit' to end the conversation.")
    print("Type 'thinking on' to see the thinking process.")
    print("Type 'thinking off' to hide the thinking process.")
    print("Type 'thinking fast' for faster thinking display.")
    print("Type 'thinking slow' for slower thinking display.\n")
    
    show_thinking = True
    typing_speed = 0.01  # Default speed (seconds per character)
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nGoodbye!")
            break
        
        # Check for thinking commands
        if user_input.lower() == "thinking on":
            show_thinking = True
            print("Thinking tokens enabled. You will see the agent's reasoning process.")
            continue
        elif user_input.lower() == "thinking off":
            show_thinking = False
            print("Thinking tokens disabled. You will only see the final responses.")
            continue
        elif user_input.lower() == "thinking fast":
            typing_speed = 0.005
            print("Thinking display speed increased.")
            continue
        elif user_input.lower() == "thinking slow":
            typing_speed = 0.03
            print("Thinking display speed decreased for more dramatic effect.")
            continue
            
        # Prepare state for this turn
        current_state = state.copy()
        current_state["input"] = user_input
        
        # Run the graph
        result = compiled_graph.invoke(current_state)
        
        # Update the persistent state with new history
        state["history"] = result["history"]
        
        # Show thinking tokens if enabled
        if show_thinking:
            display_thinking_tokens(result["thinking_tokens"], typing_speed)
        
        # Display the response
        print(f"\nAgent: {result['response']}")


if __name__ == "__main__":
    main()