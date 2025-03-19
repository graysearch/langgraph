#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph Memory Example

This script demonstrates a clean implementation of memory in LangGraph,
allowing the AI to maintain context across multiple conversation turns.
"""

import os
import openai
from typing import Dict, Any, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# Load environment variables from .env file (for API keys)
load_dotenv()

# Initialize the OpenAI client using the API key from the environment variable
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def process_input_and_memory(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process the user's input and update the conversation memory.
    
    Args:
        state: Dictionary containing the current graph state
    
    Returns:
        Updated state with user message added to history
    """
    # Initialize history if it doesn't exist
    if "history" not in state:
        state["history"] = []
    
    # Get user input
    user_input = state.get("input", "")
    
    # Add user message to history
    state["history"].append({"role": "user", "content": user_input})
    
    return state


def generate_ai_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a response using OpenAI API that takes conversation history into account.
    
    Args:
        state: Dictionary containing the current graph state and history
    
    Returns:
        Updated state with OpenAI-generated content and updated history
    """
    try:
        # Prepare messages for the API call
        messages = []
        
        # Add system message to set context
        messages.append({
            "role": "system", 
            "content": ("You are a helpful, friendly assistant that maintains context throughout "
                        "the conversation. You remember details the user has shared previously "
                        "and refer back to them when relevant.")
        })
        
        # Add conversation history
        messages.extend(state.get("history", []))
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
        )
        
        # Extract and add the response to state
        assistant_message = response.choices[0].message.content.strip()
        state["response"] = assistant_message
        
        # Add assistant response to history for future context
        state["history"].append({"role": "assistant", "content": assistant_message})
        
    except Exception as e:
        # Handle errors gracefully
        error_message = f"I encountered an issue while generating a response: {str(e)}"
        state["response"] = error_message
        state["history"].append({"role": "assistant", "content": error_message})
    
    return state


def build_graph() -> StateGraph:
    """
    Build a simple graph with memory implementation.
    
    Returns:
        StateGraph: A graph object that can be compiled
    """
    # Create a new graph with dictionary state
    graph = StateGraph(dict)
    
    # Add our nodes
    graph.add_node("process_input", process_input_and_memory)
    graph.add_node("generate_response", generate_ai_response)
    
    # Set the entry point
    graph.set_entry_point("process_input")
    
    # Add edges to define the flow
    graph.add_edge("process_input", "generate_response")
    graph.add_edge("generate_response", END)
    
    return graph


def save_mermaid_diagram():
    """
    Create a Mermaid diagram definition and save it to an HTML file.
    """
    mermaid_code = """
    graph TD
        START --> process_input
        process_input --> generate_response
        generate_response --> END
    """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LangGraph Visualization</title>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{ startOnLoad: true }});
        </script>
    </head>
    <body>
        <h1>LangGraph with Memory - Flow Visualization</h1>
        <div class="mermaid">
        {mermaid_code}
        </div>
    </body>
    </html>
    """
    
    with open("graph_visualization.html", "w") as f:
        f.write(html_content)
    
    print("Saved Mermaid diagram to 'graph_visualization.html'")


def main():
    """Run the LangGraph with memory example."""
    print("\n--- LangGraph with Memory Example ---\n")
    
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
    print("Chat with the AI assistant. Type 'exit' to end the conversation.\n")
    
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
        print(f"Assistant: {result['response']}")
        
    print("\n--- End of Example ---\n")


if __name__ == "__main__":
    main()