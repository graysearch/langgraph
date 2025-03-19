#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph Agent with DeepThink Reasoning

This script demonstrates how to build an agent using LangGraph that shows
detailed, step-by-step reasoning inspired by chain-of-thought approaches.
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
    
    # Initialize reasoning log if it doesn't exist
    if "reasoning_log" not in state:
        state["reasoning_log"] = []
        
    # Initialize deepthink log if it doesn't exist
    if "deepthink_log" not in state:
        state["deepthink_log"] = []
    
    # Get user input
    user_input = state.get("input", "")
    
    # Add user message to history
    state["history"].append({"role": "user", "content": user_input})
    
    # Add to reasoning log
    state["reasoning_log"].append(f"📝 Received user input: \"{user_input}\"")
    state["deepthink_log"].append({
        "phase": "input_processing",
        "thought": f"User has provided the input: \"{user_input}\". Now I need to understand what they're asking for."
    })
    
    return state


def deepthink_decision(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform detailed, token-by-token reasoning to decide which action to take.
    """
    # Access the latest user message
    latest_message = state["history"][-1]["content"]
    
    # Add to reasoning log
    state["reasoning_log"].append(f"🧠 DeepThink analysis of: \"{latest_message}\"")
    
    # Create a prompt for the LLM to show token-by-token reasoning
    messages = [
        {"role": "system", "content": """You are an assistant that demonstrates extremely detailed thinking.
        
         Think through this problem step-by-step, showing your thought process token by token. 
         Your goal is to analyze whether the user's request requires any of these tools:
         
         - calculator: For mathematical calculations. Example: "What is 25 * 16?"
         - get_current_time: To get the current date and time. Example: "What time is it?"
         - get_weather: To check weather for a location. Example: "What's the weather in Paris?"
         
         First, explain your thinking process in extreme detail, considering all aspects of the request.
         Break down the request into components, identify key terms, and consider possible interpretations.
         Think about whether this requires factual knowledge, calculation, or external data.
         
         FORMAT YOUR RESPONSE AS FOLLOWS:
         STEP 1: [First thinking step]
         STEP 2: [Second thinking step]
         ...
         STEP N: [Final thinking step]
         DECISION: [Your final decision: calculator, get_current_time, get_weather, or none]
         
         If a tool is needed, also include:
         ARGS: {"param_name": "value"}
         
         Example thinking process for "What is 25 * 16?":
         STEP 1: The user is asking "What is 25 * 16?". This appears to be a mathematical question.
         STEP 2: Breaking it down, I can identify "25" and "16" as numbers, and "*" as the multiplication operator.
         STEP 3: This is clearly a request to perform multiplication between these two numbers.
         STEP 4: To perform mathematical calculations accurately, I should use the calculator tool.
         STEP 5: The calculator tool needs an expression to evaluate, which in this case is "25 * 16".
         DECISION: calculator
         ARGS: {"expression": "25 * 16"}
         """},
        {"role": "user", "content": latest_message}
    ]
    
    # Get detailed reasoning
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2,  # Lower temperature for more deterministic reasoning
    )
    
    reasoning = response.choices[0].message.content.strip()
    
    # Extract steps and decision
    steps = re.findall(r"STEP \d+: (.*?)(?=STEP \d+:|DECISION:|$)", reasoning, re.DOTALL)
    decision_match = re.search(r"DECISION: (.*?)(?=ARGS:|$)", reasoning, re.DOTALL)
    args_match = re.search(r"ARGS: ({.*})", reasoning)
    
    # Record each reasoning step with a brief pause to simulate thinking
    for i, step in enumerate(steps):
        step_content = step.strip()
        state["deepthink_log"].append({
            "phase": "decision_making",
            "step": i+1,
            "thought": step_content
        })
        # Simulate thinking time
        time.sleep(0.1)
    
    # Extract the decision
    if decision_match:
        decision = decision_match.group(1).strip().lower()
        state["deepthink_log"].append({
            "phase": "decision_conclusion",
            "thought": f"Final decision: {decision}"
        })
        
        # Determine action based on decision
        if "calculator" in decision and args_match:
            try:
                args = json.loads(args_match.group(1))
                state["action"] = "use_tool"
                state["tool_name"] = "calculator"
                state["tool_args"] = args
                state["reasoning_log"].append(f"🧮 DeepThink decided to use calculator with args: {args}")
            except json.JSONDecodeError:
                state["action"] = "respond_directly"
                state["reasoning_log"].append("⚠️ Couldn't parse calculator arguments, falling back to direct response")
        
        elif "get_current_time" in decision:
            state["action"] = "use_tool"
            state["tool_name"] = "get_current_time"
            state["tool_args"] = {}
            state["reasoning_log"].append("🕒 DeepThink decided to get the current time")
        
        elif "get_weather" in decision and args_match:
            try:
                args = json.loads(args_match.group(1))
                state["action"] = "use_tool"
                state["tool_name"] = "get_weather"
                state["tool_args"] = args
                state["reasoning_log"].append(f"🌤️ DeepThink decided to get weather with args: {args}")
            except json.JSONDecodeError:
                state["action"] = "respond_directly"
                state["reasoning_log"].append("⚠️ Couldn't parse weather arguments, falling back to direct response")
        
        else:
            state["action"] = "respond_directly"
            state["reasoning_log"].append("💬 DeepThink decided to respond directly without using a tool")
    else:
        state["action"] = "respond_directly"
        state["reasoning_log"].append("⚠️ No clear decision found, falling back to direct response")
    
    # Store complete reasoning for reference
    state["full_reasoning"] = reasoning
    
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
    state["deepthink_log"].append({
        "phase": "tool_execution",
        "thought": f"Now executing the {tool_name} tool with parameters: {tool_args}"
    })
    
    # Execute the requested tool
    if tool_name == "calculator" and "expression" in tool_args:
        state["deepthink_log"].append({
            "phase": "tool_execution",
            "thought": f"Calculating expression: {tool_args['expression']}"
        })
        result = calculator(tool_args["expression"])
    elif tool_name == "get_current_time":
        state["deepthink_log"].append({
            "phase": "tool_execution",
            "thought": "Retrieving the current system time"
        })
        result = get_current_time()
    elif tool_name == "get_weather" and "location" in tool_args:
        state["deepthink_log"].append({
            "phase": "tool_execution", 
            "thought": f"Checking weather information for location: {tool_args['location']}"
        })
        result = get_weather(tool_args["location"])
    else:
        result = f"Error: Invalid tool '{tool_name}' or missing required arguments"
        state["deepthink_log"].append({
            "phase": "tool_execution",
            "thought": f"Error executing tool: {result}"
        })
    
    # Store the result
    state["tool_result"] = result
    state["reasoning_log"].append(f"✅ Tool result: {result}")
    state["deepthink_log"].append({
        "phase": "tool_result",
        "thought": f"Tool execution complete. Result: {result}"
    })
    
    # Add to history as a function message
    state["history"].append({"role": "function", "name": tool_name, "content": result})
    
    return state


def deepthink_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a final response with detailed reasoning.
    """
    # Add to reasoning log
    state["reasoning_log"].append("🧠 DeepThink generating final response")
    state["deepthink_log"].append({
        "phase": "response_generation",
        "thought": "Now I need to formulate a complete response based on all available information"
    })
    
    # Prepare messages for the API call
    if state.get("action") == "use_tool":
        # If we used a tool, include a special system message
        system_message = f"""You are a helpful assistant with strong reasoning abilities.
        
        The user asked: "{state["history"][0]["content"]}"
        
        You used the {state["tool_name"]} tool, which returned: "{state["tool_result"]}"
        
        Now explain your thought process step by step for formulating a response based on this tool result.
        Think about how to present this information clearly and helpfully.
        
        FORMAT YOUR RESPONSE AS FOLLOWS:
        STEP 1: [First reasoning step]
        STEP 2: [Second reasoning step]
        ...
        FINAL RESPONSE: [The actual response to give to the user]
        """
    else:
        # For direct responses
        system_message = f"""You are a helpful assistant with strong reasoning abilities.
        
        The user asked: "{state["history"][0]["content"]}"
        
        Think step by step about how to respond to this request without using any external tools.
        Consider what information you already know that can help answer this question.
        
        FORMAT YOUR RESPONSE AS FOLLOWS:
        STEP 1: [First reasoning step]
        STEP 2: [Second reasoning step]
        ...
        FINAL RESPONSE: [The actual response to give to the user]
        """
    
    messages = [
        {"role": "system", "content": system_message}
    ]
    
    # Get response with reasoning
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,  # Higher temperature for more creative responses
    )
    
    full_response = response.choices[0].message.content.strip()
    
    # Extract steps and final response
    steps = re.findall(r"STEP \d+: (.*?)(?=STEP \d+:|FINAL RESPONSE:|$)", full_response, re.DOTALL)
    final_match = re.search(r"FINAL RESPONSE: (.*)", full_response, re.DOTALL)
    
    # Record each response reasoning step
    for i, step in enumerate(steps):
        step_content = step.strip()
        state["deepthink_log"].append({
            "phase": "response_formulation",
            "step": i+1,
            "thought": step_content
        })
        # Simulate thinking time
        time.sleep(0.1)
    
    # Extract the final response
    if final_match:
        assistant_message = final_match.group(1).strip()
    else:
        # Fallback if format not followed
        assistant_message = "I apologize, but I encountered an issue while formulating my response."
    
    state["response"] = assistant_message
    state["reasoning_log"].append(f"🤖 Final response: {assistant_message}")
    state["deepthink_log"].append({
        "phase": "final_output",
        "thought": f"Delivering final response to user: {assistant_message}"
    })
    
    # Add assistant response to history
    state["history"].append({"role": "assistant", "content": assistant_message})
    
    return state


def router(state: Dict[str, Any]) -> Literal["use_tool", "deepthink_response"]:
    """
    Determine which node to call next based on the action decision.
    """
    return state["action"]


def build_graph() -> StateGraph:
    """
    Build the agent graph with DeepThink reasoning.
    """
    # Create a new graph
    graph = StateGraph(dict)
    
    # Add nodes
    graph.add_node("process_input", process_input)
    graph.add_node("deepthink_decision", deepthink_decision)
    graph.add_node("use_tool", use_tool)
    graph.add_node("deepthink_response", deepthink_response)
    
    # Set the entry point
    graph.set_entry_point("process_input")
    
    # Add basic edges
    graph.add_edge("process_input", "deepthink_decision")
    
    # Add conditional edge from deepthink_decision
    graph.add_conditional_edges(
        "deepthink_decision",
        router,
        {
            "use_tool": "use_tool",
            "respond_directly": "deepthink_response"
        }
    )
    
    # Connect remaining nodes
    graph.add_edge("use_tool", "deepthink_response")
    graph.add_edge("deepthink_response", END)
    
    return graph


def display_deepthink(log, thinking_speed=0.05):
    """
    Display the DeepThink process with a typing effect.
    """
    for entry in log:
        phase = entry["phase"]
        
        # Format the phase header
        if phase == "input_processing":
            print("\n🔍 UNDERSTANDING INPUT:")
        elif phase == "decision_making":
            if entry.get("step") == 1:
                print("\n🧠 ANALYZING REQUEST:")
            print(f"  Step {entry.get('step', '•')}:", end=" ")
        elif phase == "decision_conclusion":
            print("\n✅ DECISION:", end=" ")
        elif phase == "tool_execution":
            print("\n⚙️ EXECUTING TOOL:", end=" ")
        elif phase == "tool_result":
            print("\n📊 TOOL RESULT:", end=" ")
        elif phase == "response_formulation":
            if entry.get("step") == 1:
                print("\n💭 FORMULATING RESPONSE:")
            print(f"  Step {entry.get('step', '•')}:", end=" ")
        elif phase == "final_output":
            print("\n📝 FINAL RESPONSE:", end=" ")
        
        # Display the thought with a typing effect
        thought = entry["thought"]
        for char in thought:
            print(char, end="", flush=True)
            time.sleep(thinking_speed)
        print()


def main():
    """Run the LangGraph agent with DeepThink reasoning."""
    print("\n--- LangGraph Agent with DeepThink Reasoning ---\n")
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in a .env file or export it to your environment.")
        return
    
    # Build and compile the graph
    graph = build_graph()
    compiled_graph = graph.compile()
    
    # Initialize conversation history
    state = {"history": [], "reasoning_log": [], "deepthink_log": []}
    
    # Interactive conversation loop
    print("Chat with the agent. You can ask for calculations, weather, or the current time.")
    print("Type 'exit' to end the conversation.")
    print("Type 'deepthink on' to see detailed step-by-step reasoning.")
    print("Type 'deepthink off' to hide the detailed reasoning.")
    print("Type 'deepthink fast' for faster thinking animation.")
    print("Type 'deepthink slow' for slower thinking animation.\n")
    
    show_deepthink = False
    thinking_speed = 0.01  # Default speed (seconds per character)
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nGoodbye!")
            break
        
        # Check for deepthink commands
        if user_input.lower() == "deepthink on":
            show_deepthink = True
            print("DeepThink mode enabled. You will see detailed step-by-step reasoning.")
            continue
        elif user_input.lower() == "deepthink off":
            show_deepthink = False
            print("DeepThink mode disabled. Detailed reasoning will be hidden.")
            continue
        elif user_input.lower() == "deepthink fast":
            thinking_speed = 0.005
            print("DeepThink speed increased.")
            continue
        elif user_input.lower() == "deepthink slow":
            thinking_speed = 0.03
            print("DeepThink speed decreased for more dramatic effect.")
            continue
            
        # Prepare state for this turn
        current_state = state.copy()
        current_state["input"] = user_input
        
        # Run the graph
        result = compiled_graph.invoke(current_state)
        
        # Update the persistent state with new history
        state["history"] = result["history"]
        
        # Show DeepThink reasoning if enabled
        if show_deepthink:
            display_deepthink(result["deepthink_log"], thinking_speed)
            print("\n" + "-" * 80)
        
        # Display the response
        print(f"\nAgent: {result['response']}")


if __name__ == "__main__":
    main()