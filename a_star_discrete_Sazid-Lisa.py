import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from collections import defaultdict
from heapq import heappush, heappop

def position(position, grid):
    rows, cols = grid.shape
    return 0 <= position[0] < rows and 0 <= position[1] < cols

def check_obstacle(position, grid):
    return grid[position] != 1

def is_valid_node(node, grid):
    return position(node, grid) and check_obstacle(node, grid)

def get_distance(pos1, pos2):
    return np.linalg.norm(np.subtract(pos1, pos2))

def a_star_search(start, goal, grid):
    movements = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    open_set = []
    heappush(open_set, (0 + get_distance(start, goal), start))
    g_scores = defaultdict(lambda: float('inf'))
    g_scores[start] = 0
    f_scores = defaultdict(lambda: float('inf'))
    f_scores[start] = get_distance(start, goal)
    parent_set = {start: None}

    while open_set:
        current_f_score, current_position = heappop(open_set)

        if current_position == goal:
            path = []
            while current_position:
                path.append(current_position)
                current_position = parent_set[current_position]
            return path[::-1], f_scores[goal]

        for dx, dy in movements:
            next_position = (current_position[0] + dx, current_position[1] + dy)
            tentative_g_score = g_scores[current_position] + get_distance(current_position, next_position)

            if is_valid_node(next_position, grid) and tentative_g_score < g_scores[next_position]:
                parent_set[next_position] = current_position
                g_scores[next_position] = tentative_g_score
                f_scores[next_position] = tentative_g_score + get_distance(next_position, goal)
                heappush(open_set, (f_scores[next_position], next_position))

    return [], float('inf')

# Load grid map
image_path = 'map3.png'
image = Image.open(image_path).convert('L')
grid = np.array(image).astype(np.float32)
grid /= 255.0
grid = 1 - grid
grid = np.round(grid).astype(np.int8)

start_node = (50,90)
goal_node = (375,375)

path, path_cost = a_star_search(start_node, goal_node, grid)

if path:
    print("Path found:", path)
    print("Path cost:", path_cost)

    # Visualization
    for node in path:
        grid[node] = 2  # Path
    grid[start_node] = 3  # Start
    grid[goal_node] = 4  # Goal

    plt.matshow(grid)
    plt.colorbar()
    plt.show()
else:
    print("No path found")
