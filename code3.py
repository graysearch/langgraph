#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph Example with Memory Implementation

This script builds on the previous example by adding a memory system
that keeps track of conversation history, allowing the LLM to maintain
context across multiple interactions.
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


def memory_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process user input and add it to conversation history.
    
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


def greeting_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a greeting based on user input.
    
    Args:
        state: Dictionary containing the current graph state
    
    Returns:
        Updated state with a greeting added
    """
    # Get the name from the most recent user message
    name = state.get("name", "World")
    
    # Add greeting to state
    state["greeting"] = f"Hello, {name}!"
    state["date"] = "Today's date is 2025-03-17"
    
    return state


def openai_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a response using OpenAI API that takes conversation history into account.
    
    Args:
        state: Dictionary containing the current graph state and history
    
    Returns:
        Updated state with OpenAI-generated content
    """
    try:
        # Prepare messages for the API call
        messages = []
        
        # Add system message to set context
        messages.append({
            "role": "system", 
            "content": "You are a helpful, friendly assistant that maintains context throughout the conversation."
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
        state["assistant_response"] = assistant_message
        
        # Add assistant response to history for future context
        state["history"].append({"role": "assistant", "content": assistant_message})
        
    except Exception as e:
        # Handle errors gracefully
        error_message = f"I encountered an issue while generating a response: {str(e)}"
        state["assistant_response"] = error_message
        state["history"].append({"role": "assistant", "content": error_message})
    
    return state


def build_graph() -> StateGraph:
    """
    Build the graph with memory implementation.
    
    Returns:
        StateGraph: A graph object that can be visualized and compiled
    """
    # Create a new graph with dictionary state
    graph = StateGraph(dict)
    
    # Add our nodes
    graph.add_node("memory", memory_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("openai_response", openai_response_node)
    
    # Set the entry point to the memory node
    graph.set_entry_point("memory")
    
    # Add edges to define the flow
    graph.add_edge("memory", "greeting")
    graph.add_edge("greeting", "openai_response")
    graph.add_edge("openai_response", END)
    
    return graph


def save_mermaid_diagram():
    """
    Create a Mermaid diagram definition and save it to an HTML file.
    """
    mermaid_code = """
    graph TD
        START --> memory
        memory --> greeting
        greeting --> openai_response
        openai_response --> END
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
        
        # For the first message, extract name for greeting
        if not state["history"]:
            # Try to extract a name from the first message
            words = user_input.split()
            if len(words) > 1 and "name" in user_input.lower():
                # Very simple name extraction, could be improved
                for word in words:
                    if word != "my" and word != "name" and word != "is" and len(word) > 2:
                        state["name"] = word.strip(".,!?")
                        break
        
        # Prepare state for this turn
        current_state = state.copy()
        current_state["input"] = user_input
        
        # Run the graph
        result = compiled_graph.invoke(current_state)
        
        # Update the persistent state with new history
        state["history"] = result["history"]
        
        # Display the response
        print(f"Assistant: {result['assistant_response']}")
        
    print("\n--- End of Example ---\n")


if __name__ == "__main__":
    main()