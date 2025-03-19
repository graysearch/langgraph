from langgraph.graph import StateGraph, END
from langgraph.graph.viz import visualize

# Define a node function that adds both greeting and date
def hello_world(state):
    """
    A node function that adds a greeting and date to the state.
    """
    # Get the name from state with "World" as default
    name = state.get("name", "World")
    
    # Add greeting and date to the state
    state["greeting"] = f"Hello, {name}!"
    state["date"] = "Today's date is 2025-03-17"
    
    return state

# Create and set up the graph
builder = StateGraph(dict)
builder.add_node("hello", hello_world)
builder.set_entry_point("hello")
builder.add_edge("hello", END)

# Before compiling, visualize the graph
# This function creates a visual representation of your graph

# Method 1: Display the graph directly in a Jupyter notebook
# If you're running this in a Jupyter notebook, this will render it inline
# visualize(builder)

# Method 2: Save the visualization to an HTML file
# This creates an HTML file that you can open in any web browser
visualize(builder).save("my_graph.html")

# Method 3: Print the graph as a DOT string (Graphviz format)
# You can use this with Graphviz tools to create custom visualizations
dot_string = visualize(builder).source
print("Graph DOT string:")
print(dot_string)

# Now compile and run the graph as before
graph = builder.compile()
result = graph.invoke({"name": "LangGraph"})

# Print both the greeting and date from the result state
print("\nExecution result:")
print(f"{result['greeting']} {result['date']}")