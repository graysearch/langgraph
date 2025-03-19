#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph Agent with Transparent Reasoning

This script demonstrates how to build an agent using LangGraph that shows
its reasoning process to users, making the decision-making transparent.
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
    
    # Initialize reasoning log if it doesn't exist
    if "reasoning_log" not in state:
        state["reasoning_log"] = []
    
    # Get user input
    user_input = state.get("input", "")
    
    # Add user message to history
    state["history"].append({"role": "user", "content": user_input})
    
    # Add to reasoning log
    state["reasoning_log"].append(f"📝 Received user input: \"{user_input}\"")
    
    return state


def decide_action(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze user request and decide whether to use a tool or respond directly.
    Show reasoning in the process.
    """
    # Access the latest user message
    latest_message = state["history"][-1]["content"]
    
    # Add to reasoning log
    state["reasoning_log"].append(f"🤔 Analyzing what to do with: \"{latest_message}\"")
    
    # Create a prompt for the LLM to decide the action
    messages = [
        {"role": "system", "content": """You are an assistant that can use tools or respond directly. 
         Analyze the user's request and decide what to do.
         
         Available tools:
         - calculator: For mathematical calculations. Example: "What is 25 * 16?"
         - get_current_time: To get the current date and time. Example: "What time is it?"
         - get_weather: To check weather for a location. Example: "What's the weather in Paris?"
         
         First, explain your thinking process in a few sentences. Start with "THINKING:"
         
         Then, if a tool is needed, respond in this format:
         TOOL: tool_name
         ARGS: {"arg_name": "value"}
         
         For example:
         THINKING: This is clearly a mathematical calculation asking for the product of 25 and 16.
         TOOL: calculator
         ARGS: {"expression": "25 * 16"}
         
         Or:
         THINKING: The user is asking about the current weather conditions in Paris.
         TOOL: get_weather
         ARGS: {"location": "Paris"}
         
         If no tool is needed, respond with:
         THINKING: [your reasoning]
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
    
    # Extract thinking process
    thinking_match = re.search(r"THINKING:(.*?)(?=TOOL:|DIRECT_RESPONSE|$)", decision, re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
        state["reasoning_log"].append(f"💭 Agent's thinking: {thinking}")
    
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
                
                # Add to reasoning log
                state["reasoning_log"].append(f"🔧 Decided to use tool: {tool_name} with args: {args}")
            except json.JSONDecodeError:
                # If JSON parsing fails, fall back to direct response
                state["action"] = "respond_directly"
                state["reasoning_log"].append("⚠️ Couldn't parse tool arguments, falling back to direct response")
        else:
            state["action"] = "respond_directly"
            state["reasoning_log"].append("⚠️ Tool format incorrect, falling back to direct response")
    else:
        state["action"] = "respond_directly"
        state["reasoning_log"].append("💬 Decided to respond directly without using a tool")
    
    return state


def use_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the selected tool and capture the result.
    """
    tool_name = state["tool_name"]
    tool_args = state["tool_args"]
    result = None
    
    # Add to reasoning log
    state["reasoning_log"].append(f"🛠️ Executing tool: {tool_name}")
    
    # Execute the requested tool
    if tool_name == "calculator" and "expression" in tool_args:
        state["reasoning_log"].append(f"🧮 Calculating expression: {tool_args['expression']}")
        result = calculator(tool_args["expression"])
    elif tool_name == "get_current_time":
        state["reasoning_log"].append("🕒 Fetching current time")
        result = get_current_time()
    elif tool_name == "get_weather" and "location" in tool_args:
        state["reasoning_log"].append(f"🌤️ Checking weather for: {tool_args['location']}")
        result = get_weather(tool_args["location"])
    else:
        result = f"Error: Invalid tool '{tool_name}' or missing required arguments"
        state["reasoning_log"].append(f"❌ Error with tool: {result}")
    
    # Store the result
    state["tool_result"] = result
    state["reasoning_log"].append(f"✅ Tool result: {result}")
    
    # Add to history as a function message
    state["history"].append({"role": "function", "name": tool_name, "content": result})
    
    return state


def generate_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a final response, either directly or based on tool results.
    """
    # Add to reasoning log
    if state.get("action") == "use_tool":
        state["reasoning_log"].append("📝 Generating response based on tool result")
    else:
        state["reasoning_log"].append("📝 Generating direct response")
    
    # Prepare messages for the API call
    messages = [
        {"role": "system", "content": """You are a helpful, friendly assistant that can use tools to assist the user.
        Start your response with a brief explanation of how you arrived at your answer, prefixed with 'REASONING:'.
        Then provide your actual response to the user on a new line after 'RESPONSE:'.
        Make your reasoning insightful but concise."""}
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
    full_message = response.choices[0].message.content.strip()
    
    # Extract reasoning and response
    reasoning_match = re.search(r"REASONING:(.*?)(?=RESPONSE:|$)", full_message, re.DOTALL)
    response_match = re.search(r"RESPONSE:(.*)", full_message, re.DOTALL)
    
    if reasoning_match and response_match:
        reasoning = reasoning_match.group(1).strip()
        assistant_message = response_match.group(1).strip()
        
        state["reasoning_log"].append(f"💭 Final reasoning: {reasoning}")
    else:
        # If the format wasn't followed, use the whole message
        assistant_message = full_message
    
    state["response"] = assistant_message
    state["reasoning_log"].append(f"🤖 Final response: {assistant_message}")
    
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


def main():
    """Run the LangGraph agent example with transparent reasoning."""
    print("\n--- LangGraph Agent with Transparent Reasoning ---\n")
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in a .env file or export it to your environment.")
        return
    
    # Build and compile the graph
    graph = build_graph()
    compiled_graph = graph.compile()
    
    # Initialize conversation history
    state = {"history": [], "reasoning_log": []}
    
    # Interactive conversation loop
    print("Chat with the agent. You can ask for calculations, weather, or the current time.")
    print("Type 'exit' to end the conversation.")
    print("Type 'debug on' to see the agent's reasoning.")
    print("Type 'debug off' to hide the agent's reasoning.\n")
    
    show_reasoning = False
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nGoodbye!")
            break
        
        # Check for debug commands
        if user_input.lower() == "debug on":
            show_reasoning = True
            print("Debug mode enabled. You will see the agent's reasoning.")
            continue
        elif user_input.lower() == "debug off":
            show_reasoning = False
            print("Debug mode disabled. The agent's reasoning will be hidden.")
            continue
            
        # Prepare state for this turn
        current_state = state.copy()
        current_state["input"] = user_input
        
        # Run the graph
        result = compiled_graph.invoke(current_state)
        
        # Update the persistent state with new history
        state["history"] = result["history"]
        
        # Show reasoning if debug mode is on
        if show_reasoning:
            print("\n🔍 Agent Reasoning:")
            for log in result["reasoning_log"]:
                print(f"  {log}")
            print()
        
        # Display the response
        print(f"Agent: {result['response']}\n")


if __name__ == "__main__":
    main()