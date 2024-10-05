import heapq
import math

# Node class for representing each cell in the grid
class Node:
    def __init__(self, x, y, walkable=True):
        self.x = x
        self.y = y
        self.walkable = walkable  # True if the node is not an obstacle
        self.g = float('inf')  # Distance from start node
        self.h = 0  # Heuristic distance to the goal node
        self.f = 0  # f = g + h
        self.parent = None  # To reconstruct the path later

    def __lt__(self, other):
        return self.f < other.f


# A* Algorithm
def a_star_search(grid, start, goal):
    open_list = []
    closed_list = set()
    heapq.heappush(open_list, start)

    start.g = 0
    start.f = start.h = manhattan_distance(start, goal)

    while open_list:
        current_node = heapq.heappop(open_list)

        if current_node == goal:
            return reconstruct_path(current_node)

        closed_list.add((current_node.x, current_node.y))

        # Explore neighbors
        neighbors = get_neighbors(grid, current_node)
        for neighbor in neighbors:
            if (neighbor.x, neighbor.y) in closed_list or not neighbor.walkable:
                continue

            tentative_g = current_node.g + 1  # Assume uniform cost of 1 to move to neighbor

            if tentative_g < neighbor.g:
                neighbor.parent = current_node
                neighbor.g = tentative_g
                neighbor.h = manhattan_distance(neighbor, goal)
                neighbor.f = neighbor.g + neighbor.h

                if neighbor not in open_list:
                    heapq.heappush(open_list, neighbor)

    return None  # No path found


# Heuristic: Manhattan distance (you can replace with Euclidean for another option)
def manhattan_distance(node1, node2):
    return abs(node1.x - node2.x) + abs(node1.y - node2.y)


# Get neighboring nodes (up, down, left, right)
def get_neighbors(grid, node):
    neighbors = []
    if node.x > 0:
        neighbors.append(grid[node.x - 1][node.y])  # Left
    if node.x < len(grid) - 1:
        neighbors.append(grid[node.x + 1][node.y])  # Right
    if node.y > 0:
        neighbors.append(grid[node.x][node.y - 1])  # Up
    if node.y < len(grid[0]) - 1:
        neighbors.append(grid[node.x][node.y + 1])  # Down
    return neighbors


# Reconstruct the path by backtracking from the goal node to the start node
def reconstruct_path(current_node):
    path = []
    while current_node:
        path.append((current_node.x, current_node.y))
        current_node = current_node.parent
    return path[::-1]  # Return reversed path


# Create a grid with obstacles (you can add dynamic obstacles here if needed)
def create_grid(rows, cols, obstacles):
    grid = [[Node(x, y) for y in range(cols)] for x in range(rows)]
    for obs in obstacles:
        grid[obs[0]][obs[1]].walkable = False  # Mark obstacles as non-walkable
    return grid


# Visualize the grid and path
def print_grid(grid, path):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if (i, j) in path:
                print("P", end=" ")  # Path
            elif not grid[i][j].walkable:
                print("X", end=" ")  # Obstacle
            else:
                print(".", end=" ")  # Empty space
        print()


# Example usage
def main():
    rows, cols = 10, 10
    start_pos = (0, 0)
    goal_pos = (9, 9)

    obstacles = [(3, 3), (3, 4), (3, 5), (6, 7), (5, 7), (4, 7)]

    grid = create_grid(rows, cols, obstacles)

    start_node = grid[start_pos[0]][start_pos[1]]
    goal_node = grid[goal_pos[0]][goal_pos[1]]

    path = a_star_search(grid, start_node, goal_node)

    if path:
        print("Path found!")
        print_grid(grid, path)
        print("Path:", path)
    else:
        print("No path found")


if __name__ == "__main__":
    main()
