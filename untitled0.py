# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lZ5v5c-PZNkrh6CNrnhs2jUljaOYw_LK
"""

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

# function to find the past cost
def past_cost(n, parent_node, edges, costs):
  if n==0:
    pc = 0
  else:
    try:
      for edge in edges:
        if (edge[0] == n or edge[0] == parent_node[n]) and (edge[1] == n or edge[1] == parent_node[n]):
          diff = costs[edge]
          pc = past_cost(parent_node[n], parent_node, edges, costs) + diff
    except KeyError:
      pc = 0
  return pc

# function to add a node either in open dictionary or closed list
def addToOpenAndClosed(n, cost, closed, openn, parent_node, pc, OC, edges):
    p = cost[n,:]
    for i in range(len(p)):
        pn = parent_node.copy()
        if p[i] > 0 and i not in closed:
            a = i
            parent_node[a] = n
            if pc[i] > past_cost(i, parent_node, edges, cost):
                pc[i] = past_cost(i, parent_node, edges, cost)
            else:
                pc[i] = pc[i]
                parent_node = pn
            ETC = pc[i] + OC[i]
            openn[a] = ETC
            keys = list(openn.keys())
            values = list(openn.values())
            sorted_values = np.argsort(values)
            openn = {keys[i]: values[i] for i in sorted_values}
        else:
            parent_node = parent_node
    try:
        del openn[n]
    except KeyError:
        openn = openn
    closed.append(n)
    return openn, closed, parent_node, pc

# function that performs Astar algorithm and plots the output
def Astar(vertices,edges_from_csv):
  nodes = load_vertices_from_file(vertices)    #loading vertices
  edges = load_edges_from_file(edges_from_csv)  #loading edges

  #optimization cost from each vertex to the goal vertex
  OC = []
  for node in nodes:
    oc = Vertex.dist(node, nodes[-1])
    OC.append(oc)


  N = len(nodes)  # Number of nodes (PS: NODES = VERTICES!)


  # initializing the cost between nodes as -1
  cost = np.full((N,N), -1)
  cost = cost.astype(float)

  # putting actual costs for the nodes that form edges, for nodes not connected it stays as -1
  for i in range(len(edges)):
      v,w = edges[i]
      cost[v,w] = Vertex.dist(nodes[v], nodes[w])
      cost[w,v] = Vertex.dist(nodes[v], nodes[w])

  px = np.full(N, float('inf'))
  px[0] = 0
  pc = {}
  for x in range(px.shape[0]):
      pc[x] = px[x]


  parent_node = {0:''}     # saves each node with its parent node
  openn = {0:OC[0]}        # open list of nodes and their optimization costs
  closed = []              # list of nodes taken to the closed after being gone through in open

  for nod in range(N):
      if len(list(openn.keys())) != 0:
          nod = list(openn.keys())[0]
          vals = addToOpenAndClosed(nod, cost, closed, openn, parent_node, pc, OC, edges)
          openn = vals[0]
          closed = vals[1]
          parent_node = vals[2]
          pc = vals[3]
  # putting the nodes showing path into a list, if there is a path. And then sorting the list into the correct order of the path

  path = []                # saves the shortest paths between nodes
  x = N-1
  path.append(x)
  try:
      while x > 0:
          path.append(parent_node[x])
          x = parent_node[x]
      real_path = []      # path in the correct order
      for v in range(len(path)):
          real_path.append(path[-(v+1)])
      print(real_path)
  except:
      print("There is no solution")

  # finding distance
  last_node = list(pc)[-1]
  distance = pc[last_node]
  print("distance is {}".format(distance))


  path_edges = []
  for i in real_path:
    if i != real_path[-1]:
      index = real_path.index(i)
      path_edges.append((real_path[index], real_path[index+1]))


  # plotting the output
  plot(nodes, edges)

  for v in nodes:
        plt.plot(v.x, v.y, 'r+')

  for e in path_edges:
        plt.plot([nodes[e[0]].x, nodes[e[1]].x],
                 [nodes[e[0]].y, nodes[e[1]].y],
                 "r-")

  for i, v in enumerate(nodes):
        plt.text(v.x + 0.2, v.y, str(i))
  plt.axis('equal')


# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print(
#             "Usage: ./Astar_Lisa_Sazid.py visibility_graph.csv edges.csv"
#         )
#     else:
#         Astar(sys.argv[1], sys.argv[2])

Astar('env_2.csv', 'visibility_graph_env_2.csv')