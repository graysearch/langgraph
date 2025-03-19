#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple LangGraph Example with OpenAI API and Visualization

This script demonstrates a basic LangGraph with just two nodes:
1. A greeting node that adds a greeting and date
2. An OpenAI node that generates a creative response

It also includes visualization using your approach.
"""

import os
import openai
from typing import Dict, Any
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# Load environment variables from .env file (for API keys)
load_dotenv()

# Initialize the OpenAI client using the API key from the environment variable
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def greeting_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    First node that creates a simple greeting.
    
    Args:
        state: Dictionary containing the current graph state
    
    Returns:
        Updated state with a greeting added
    """
    # Get the name from state
    name = state.get("name", "World")
    
    # Add greeting to state
    state["greeting"] = f"Hello, {name}!"
    state["date"] = "Today's date is 2025-03-17"
    
    return state


def openai_creative_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Second node that uses OpenAI API directly to generate a creative response.
    
    Args:
        state: Dictionary containing the current graph state
    
    Returns:
        Updated state with OpenAI-generated content
    """
    # Get the name from state
    name = state.get("name", "World")
    
    try:
        # Call OpenAI API directly using the client
        response = client.chat.completions.create(
            model="gpt-4",  # You can change this to a different model as needed
            messages=[
                {"role": "system", "content": "You are a helpful, creative assistant."},
                {"role": "user", "content": f"Write a short, creative 2-3 sentence message welcoming {name} to the world of AI. Include a fun fact about language models."},
            ],
        )
        
        # Extract and add the response to state
        state["creative_message"] = response.choices[0].message.content.strip()
        
    except Exception as e:
        # Handle errors gracefully
        state["creative_message"] = f"I wanted to generate a creative message, but encountered an issue: {str(e)}"
    
    return state


def build_graph() -> StateGraph:
    """
    Build and compile the graph with two nodes.
    
    Returns:
        StateGraph: A graph object that can be visualized and compiled
    """
    # Create a new graph with dictionary state
    graph = StateGraph(dict)
    
    # Add our nodes
    graph.add_node("greeting", greeting_node)
    graph.add_node("creative", openai_creative_node)
    
    # Set the entry point - which node to start with
    graph.set_entry_point("greeting")
    
    # Add edges to define the flow
    graph.add_edge("greeting", "creative")  # After greeting, go to creative
    graph.add_edge("creative", END)         # After creative, end the graph
    
    return graph


def visualize_graph(graph):
    """
    Visualize the graph using the Mermaid approach from your sample.
    """
    try:
        # Using the approach from the 'vizualize graph' sample
        from IPython.display import Image, display
        
        # Check if we're in an IPython/Jupyter environment
        try:
            get_ipython  # This will raise a NameError if we're not in IPython
            
            # Use the visualization code from the sample
            try:
                print("Attempting to visualize the graph using Mermaid...")
                display(Image(graph.get_graph().draw_mermaid_png()))
                print("Graph visualization displayed successfully!")
            except Exception as e:
                print(f"Visualization error: {str(e)}")
                print("This may require additional dependencies or may not work in all environments.")
        except NameError:
            # Not in IPython/Jupyter
            print("\nVisualization not available in this environment.")
            print("To visualize the graph, run this code in a Jupyter notebook.")
            print("The graph structure is:")
            print("""
            [START] --> [greeting] --> [creative] --> [END]
            """)
    except ImportError:
        print("\nIPython display module not available.")
        print("To use this visualization, install IPython: pip install ipython")


def save_mermaid_diagram():
    """
    Alternative: Create a Mermaid diagram definition and save it to an HTML file.
    This works in any environment, not just Jupyter.
    """
    mermaid_code = """
    graph TD
        START --> greeting
        greeting --> creative
        creative --> END
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
        <h1>LangGraph Flow Visualization</h1>
        <div class="mermaid">
        {mermaid_code}
        </div>
    </body>
    </html>
    """
    
    with open("graph_visualization.html", "w") as f:
        f.write(html_content)
    
    print("Saved Mermaid diagram to 'graph_visualization.html'")
    print("Open this file in a web browser to view the visualization.")


def main():
    """Run the LangGraph with OpenAI API example."""
    print("\n--- LangGraph with OpenAI API Example ---\n")
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in a .env file or export it to your environment.")
        return
    
    # Get user input
    user_name = input("Enter your name: ")
    
    # Build the graph (but don't compile yet for visualization)
    graph = build_graph()
    
    # Visualize the graph
    visualize_graph(graph)
    
    # Alternative: Save a Mermaid diagram that works in any environment
    save_mermaid_diagram()
    
    # Now compile the graph for execution
    compiled_graph = graph.compile()
    
    # Create initial state with user name
    state = {"name": user_name} if user_name else {"name": "World"}
    
    # Run the graph
    result = compiled_graph.invoke(state)
    
    # Print the results
    print("\nBasic greeting:")
    print(f"{result['greeting']} {result['date']}")
    
    print("\nCreative message from OpenAI:")
    print(result['creative_message'])
    
    print("\n--- End of Example ---\n")


if __name__ == "__main__":
    main()