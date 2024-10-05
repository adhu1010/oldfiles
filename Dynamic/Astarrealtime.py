import heapq
import random
import time

import matplotlib.pyplot as plt
import numpy as np


# A* Algorithm Node Class
class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic (estimated cost from current to goal)
        self.f = 0  # Total cost (g + h)

    def __lt__(self, other):
        return self.f < other.f

# Heuristic Functions (Manhattan Distance)
def heuristic(current, goal):
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

# A* Search Algorithm
def astar(grid, start, goal):
    open_list = []
    closed_list = set()
    open_dict = {}  # To keep track of nodes in open list (for faster lookup)
    
    # Create start and goal nodes
    start_node = Node(start)
    goal_node = Node(goal)
    
    heapq.heappush(open_list, start_node)
    open_dict[start_node.position] = start_node
    
    while open_list:
        current_node = heapq.heappop(open_list)
        del open_dict[current_node.position]
        
        # Add current node to closed list
        closed_list.add(current_node.position)

        # Check if goal is reached
        if current_node.position == goal_node.position:
            return reconstruct_path(current_node)
        
        # Explore neighbors
        for neighbor_position in get_neighbors(current_node.position, grid):
            if neighbor_position in closed_list:
                continue
            
            # Create neighbor node
            neighbor_node = Node(neighbor_position, current_node)
            neighbor_node.g = current_node.g + 1
            neighbor_node.h = heuristic(neighbor_node.position, goal_node.position)
            neighbor_node.f = neighbor_node.g + neighbor_node.h
            
            # Check if it's in open list and compare f value
            if neighbor_position in open_dict:
                existing_node = open_dict[neighbor_position]
                if neighbor_node.g < existing_node.g:  # Better path found
                    existing_node.g = neighbor_node.g
                    existing_node.f = neighbor_node.f
                    existing_node.parent = current_node
            else:
                heapq.heappush(open_list, neighbor_node)
                open_dict[neighbor_position] = neighbor_node
    
    return None  # No path found

# Reconstruct the path from goal to start
def reconstruct_path(current_node):
    path = []
    while current_node:
        path.append(current_node.position)
        current_node = current_node.parent
    return path[::-1]

# Get valid neighbors for a given node
def get_neighbors(position, grid):
    neighbors = []
    rows, cols = len(grid), len(grid[0])
    for new_position in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # Right, Left, Down, Up
        node_position = (position[0] + new_position[0], position[1] + new_position[1])
        if 0 <= node_position[0] < rows and 0 <= node_position[1] < cols and grid[node_position[0]][node_position[1]] == 0:
            neighbors.append(node_position)
    return neighbors

# Create a grid environment
def create_grid(rows, cols, static_obstacles, dynamic_obstacles):
    grid = np.zeros((rows, cols))
    for obstacle in static_obstacles:
        grid[obstacle] = 1  # Mark static obstacles as 1
    for obstacle in dynamic_obstacles:
        grid[obstacle] = 1  # Mark dynamic obstacles as 1
    return grid

# Move dynamic obstacles
def move_dynamic_obstacles(grid, dynamic_obstacles, static_obstacles):
    rows, cols = grid.shape
    new_dynamic_obstacles = []
    grid[:, :] = 0  # Clear the grid

    # Place static obstacles back on the grid
    for obstacle in static_obstacles:
        grid[obstacle] = 1

    for obstacle in dynamic_obstacles:
        # Randomly move the obstacle up, down, left, or right
        direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        new_position = (obstacle[0] + direction[0], obstacle[1] + direction[1])

        # Ensure the new position is within grid bounds and not a static obstacle
        if (0 <= new_position[0] < rows and 0 <= new_position[1] < cols 
            and grid[new_position[0]][new_position[1]] == 0):
            new_dynamic_obstacles.append(new_position)
            grid[new_position] = 1  # Mark new dynamic obstacle position as 1
        else:
            new_dynamic_obstacles.append(obstacle)  # If out of bounds, obstacle stays in place
    
    return new_dynamic_obstacles

# Plot the grid, path, and start/goal
def plot_grid(grid, path, start, goal, static_obstacles, dynamic_obstacles, robot_position):
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='Greys', origin='upper')

    # Plot start and goal
    ax.scatter(start[1], start[0], marker='o', color='green', label='Start')  # Start (green)
    ax.scatter(goal[1], goal[0], marker='x', color='red', label='Goal')  # Goal (red)

    # Plot path
    if path:
        path = np.array(path)
        ax.plot(path[:, 1], path[:, 0], color='blue', label='Path')

    # Plot static obstacles
    static_obstacles = np.array(static_obstacles)
    if len(static_obstacles) > 0:
        ax.scatter(static_obstacles[:, 1], static_obstacles[:, 0], color='black', label='Static Obstacles')

    # Plot dynamic obstacles
    dynamic_obstacles = np.array(dynamic_obstacles)
    if len(dynamic_obstacles) > 0:
        ax.scatter(dynamic_obstacles[:, 1], dynamic_obstacles[:, 0], color='orange', label='Dynamic Obstacles')

    # Plot robot position
    ax.scatter(robot_position[1], robot_position[0], marker='s', color='blue', label='Robot Position')

    plt.legend()
    plt.show()

# Test the A* Algorithm with continuous path re-planning and visualization
def test_astar_with_replanning():
    rows, cols = 10, 10  # Grid size
    static_obstacles = [(1, 1), (2, 1), (3, 1), (5, 5),(7,7),(8,7),(9,5)]  # Static Obstacles
    dynamic_obstacles = [(1, 3), (2, 3), (3, 3),(5, 9),(6, 6)]  # Initial Dynamic Obstacles
    
    start = (0, 0)  # Start position
    goal = (10, 10)   # Goal position
    
    path = []  # Initialize an empty path
    current_position = start
    grid = create_grid(rows, cols, static_obstacles, dynamic_obstacles)

    while current_position != goal:
        # Calculate the path from current position to goal
        path = astar(grid, current_position, goal)
        
        if path:
            print("Current path:", path)
            # Plot the grid and path
            plot_grid(grid, path, start, goal, static_obstacles, dynamic_obstacles, current_position)
            
            # Move the robot step by step along the path
            for step in path:
                if current_position == goal:
                    break
                
                # Move robot to next step in the path
                current_position = step
                print(f"Robot moved to: {current_position}")

                # Move dynamic obstacles
                dynamic_obstacles = move_dynamic_obstacles(grid, dynamic_obstacles, static_obstacles)

                # Update grid with new dynamic obstacles
                grid = create_grid(rows, cols, static_obstacles, dynamic_obstacles)

                # Plot the grid and path
                plot_grid(grid, path, start, goal, static_obstacles, dynamic_obstacles, current_position)

                # Pause for visualization
                time.sleep(1)
        else:
            print("No valid path found")
            break

test_astar_with_replanning()
