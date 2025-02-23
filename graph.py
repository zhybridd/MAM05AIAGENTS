from dataclasses import dataclass  # Used to define structured data objects

import yaml  # Used for loading AI prompts from YAML files
import streamlit as st  # Used for UI elements, such as the progress bar

@dataclass(frozen=True)
class Prompt:
    text: str  # The actual text of the AI prompt
    parameters: list  # List of parameters that need to be replaced in the prompt

    # Applies the parameters to the prompt by replacing placeholders with actual values
    def apply(self, args: dict) -> str:
        text = self.text
        for p in self.parameters:
            text = text.replace(f"{{{{{p}}}}}", args[p])  # Replaces placeholders with actual values
        return text

    # Loads a prompt from a YAML file
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(data["text"], data["parameters"])

    # Ensures that prompts can be used as dictionary keys (hashable)
    def __hash__(self):
        return hash(self.text)

# Stores execution context data during the agent run
class ExecutionContext:
    def __init__(self):
        self.context = {}  # Dictionary to store key-value pairs of execution data

    def set(self, key, value):
        self.context[key] = value  # Stores a value in the execution context

    def get(self, key):
        return self.context[key]  # Retrieves a value from the execution context

    def __str__(self):
        return str(self.context)  # String representation of the execution context

    def __repr__(self):
        return str(self.context)

@dataclass(frozen=True)
class Agent:
    name: str  # Name of the agent (e.g., "biomarkers", "oncologist")
    prompt: Prompt  # The AI prompt associated with the agent

    # Prepares the prompt by replacing placeholders with actual context values
    def _prepare_prompt(self, context: ExecutionContext) -> Prompt:
        params = {v: context.get(v) for v in self.prompt.parameters}  # Retrieves parameters from context
        return self.prompt.apply(params)  # Applies parameters to the prompt

# Defines a directed graph structure to manage AI agent execution
class Graph:

    # Creates a graph from a set of agents and their dependencies
    @classmethod
    def from_nodes(cls, nodes: set[Agent], edges: set[str]):
        graph = cls()
        for node in nodes:
            graph.add_node(node)  # Adds each agent as a node

        node_id_to_node = {node.name: node for node in nodes}  # Maps agent names to their objects

        for edge in edges:
            from_node, to_node = edge.split("->")  # Splits edge definition (e.g., "biomarkers -> oncologist")
            from_node = from_node.strip()
            to_node = to_node.strip()

            graph.add_edge(node_id_to_node[from_node], node_id_to_node[to_node])  # Connects agents
        return graph

    def __init__(self):
        self.nodes = set()  # Stores the set of agents (nodes)
        self.edges = dict()  # Stores connections between agents (edges)

    def add_node(self, value: Agent):
        self.nodes.add(value)  # Adds an agent to the graph
        self.edges[value] = []  # Initializes an empty list of connections for the new agent

    def add_edge(self, from_node: Agent, to_node: Agent):
        self.edges[from_node].append(to_node)  # Adds a directed edge between two agents

    def run(self, context: ExecutionContext):
        # Sorts the graph nodes using topological sorting (Depth First Search)
        visited: set[Agent] = set()  # Keeps track of visited nodes
        stack: list[Agent] = []  # Stores nodes in execution order

        def dfs(node):
            if node in visited:
                return  # If the node is already visited, do nothing
            visited.add(node)
            for neighbor in self.edges[node]:
                dfs(neighbor)  # Recursively visit all connected nodes
            stack.append(node)  # Adds the node to the execution stack

        for node in self.nodes:
            dfs(node)  # Runs DFS for all nodes in the graph
        stack = stack[::-1]  # Reverses the stack to get the correct execution order
        
        # Executes the graph in order
        bar = st.sidebar.progress(0)  # Initializes a progress bar in the UI
        for i, node in enumerate(stack):
            context = node.run(context)  # Runs each agent in the execution sequence
            bar.progress((i + 1) / len(stack))  # Updates the progress bar
        return context  # Returns the final execution context
