import sys
import csv
from matplotlib import pyplot as plt
import numpy as np
import math
from typing import List


class Vertex:
    """
    Vertex class defined by x and y coordinate.
    """
    # constructor or initializer of vertex class

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def dist(self, p: "Vertex"):
        """
        Return distance between vertices
         Parameters:
            p: input vertex to calculate distance to.
         Returns:
            Distance to vertex from this vertex object
        """

        return math.sqrt((self.x - p.x)**2 + (self.y - p.y)**2)

    # method to define print() function of object vertex
    def __str__(self):
        return "({}, {})".format(np.round(self.x, 2), np.round(self.y, 2))

    # method to define print() function of list[] of object vertex
    def __repr__(self):
        return "({}, {})".format(np.round(self.x, 2), np.round(self.y, 2))


def plot(vertices, edges):

    for v in vertices:
        plt.plot(v.x, v.y, 'r+')

    for e in edges:
        plt.plot([vertices[e[0]].x, vertices[e[1]].x],
                 [vertices[e[0]].y, vertices[e[1]].y],
                 "g--")

    for i, v in enumerate(vertices):
        plt.text(v.x + 0.2, v.y, str(i))
    plt.axis('equal')
    


def load_vertices_from_file(filename: str):
    # list of vertices
    vertices: List[Vertex] = []
    with open(filename, newline='\n') as csvfile:
        v_data = csv.reader(csvfile, delimiter=",")
        next(v_data)
        for row in v_data:
            vertex = Vertex(float(row[1]), float(row[2]))
            vertices.append(vertex)
    return vertices


def load_edges_from_file(filename: str):
    edges = []
    with open(filename, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader)
        for row in reader:
            edges.append((int(row[0]), int(row[1])))
    return edges


 #OUR CODE STARTS HERE

# function to find the path cost

def calculate_past_cost(node, parent, edges, costs):
    if node == 0:
        return 0

    cost = 0
    for edge in edges:
        if node in edge and parent[node] in edge:
            try:
                cost = costs[edge] + calculate_past_cost(parent[node], parent, edges, costs)
                break
            except KeyError:
                return 0

    return cost


# function to add a node either in open dictionary or closed list
def update_open_closed(node, edge_costs, closed_set, open_set, parents, path_costs, heuristic_costs, edges):
    node_costs = edge_costs[node, :]
    for neighbor in range(len(node_costs)):
        if node_costs[neighbor] > 0 and neighbor not in closed_set:
            temp_parents = parents.copy()
            parents[neighbor] = node
            updated_cost = calculate_past_cost(neighbor, parents, edges, edge_costs)
            if path_costs[neighbor] > updated_cost:
                path_costs[neighbor] = updated_cost
            else:
                parents = temp_parents

            estimated_total_cost = path_costs[neighbor] + heuristic_costs[neighbor]
            open_set[neighbor] = estimated_total_cost

            # Sorting the open set by costs
            sorted_open_set = sorted(open_set.items(), key=lambda item: item[1])
            open_set = dict(sorted_open_set)

    open_set.pop(node, None)  # removes node from open_set if it exists
    closed_set.append(node)

    return open_set, closed_set, parents, path_costs


def Astar(vertices_path, edges_path):
    # Load data
    nodes = load_vertices_from_file(vertices_path)
    edges = load_edges_from_file(edges_path)

    # Calculate heuristic (optimization cost) for each node to the goal
    goal_node = nodes[-1]
    heuristic_costs = [Vertex.dist(node, goal_node) for node in nodes]

    # Initialize costs and path costs
    num_nodes = len(nodes)
    costs_matrix = np.full((num_nodes, num_nodes), -1.0, dtype=float)
    for edge in edges:
        v1, v2 = edge
        costs_matrix[v1, v2] = costs_matrix[v2, v1] = Vertex.dist(nodes[v1], nodes[v2])

    path_costs = {node_index: (0 if node_index == 0 else float('inf')) for node_index in range(num_nodes)}

    # Initialize search structures
    parent_nodes = {0: None}
    open_nodes = {0: heuristic_costs[0]}
    closed_nodes = []

    # Perform A* search
    while open_nodes:
        current_node = min(open_nodes, key=open_nodes.get)
        open_nodes, closed_nodes, parent_nodes, path_costs = update_open_closed(
            current_node, costs_matrix, closed_nodes, open_nodes, parent_nodes, path_costs, heuristic_costs, edges
        )

    # Reconstruct path
    path = []
    current = num_nodes - 1
    while current is not None:
        path.insert(0, current)
        current = parent_nodes.get(current)

    # Output results
    if path[0] != 0:
        print("There is no solution")
    else:
        print("Path:", path)
        total_distance = path_costs[num_nodes - 1]
        print(f"Distance is: {total_distance}")

        # Extract path edges
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

        # Plot the graph
        plt.figure(figsize=(8, 5))
        plot(nodes, edges)  

        # Plot nodes
        for node in nodes:
            plt.plot(node.x, node.y, 'ro')

        # Plot edges in the path
        for start, end in path_edges:
            plt.plot([nodes[start].x, nodes[end].x], [nodes[start].y, nodes[end].y], 'r-')

        # Annotate nodes with their indices
        for index, node in enumerate(nodes):
            plt.text(node.x + 0.2, node.y, str(index))

        plt.axis('equal')
        plt.show()

    return path, total_distance, path_edges


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: ./Astar_Lisa_Sazid.py vertices.csv edges.csv"
        )
    else:
        Astar(sys.argv[1], sys.argv[2])

Astar('env_2.csv', 'visibility_graph_env_2.csv')