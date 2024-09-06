import numpy as np
import networkx as nx
from limp.planner.fmt import FMTPlanner
import matplotlib.pyplot as plt
import matplotlib.colors
import random

def get_result_visualization(plt_title: str, map_design: np.ndarray, planner: FMTPlanner, path_info: dict, start, goals,show_color_bar=True) -> matplotlib.figure.Figure:
    cmap = matplotlib.colors.ListedColormap(['black', 'white', 'gold'])
    bounds = [-0.5, 0.5, 1.5, 2.5]  # Boundaries for the colormap
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 10))
    
    cax = ax.imshow(map_design, cmap=cmap, norm=norm, interpolation='nearest')

    nx.draw(planner.graph, [x[::-1] for x in planner.node_list], node_size=0, alpha=.5, ax=ax) #node_size=0 will not show sampled points
    path = path_info["path"]

    ax.plot(path[:, 1], path[:, 0], 'r-', lw=2)
    ax.scatter(start[1], start[0], color='blue', marker='X', label='Start')
    if len(goals) == 1:
        ax.scatter(goals[0][1], goals[0][0], color='chartreuse', marker='X', label='Goal')
    else:
        for i, goal in enumerate(goals):
            if i == 0:
                ax.scatter(goal[1], goal[0], color='chartreuse', marker='X', label='Goal')
            else:
                ax.scatter(goal[1], goal[0], color='chartreuse', marker='X')

    if show_color_bar:
        fig.colorbar(cax, ticks=[0, 1, 2], format=plt.FuncFormatter(lambda x, _: {0: 'Obstacles', 1: 'Free Space', 2: 'Goal Points'}[x]))
    ax.set_title(plt_title)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.legend()
    ax.axis("on")
    ax.grid(False)

    return fig

def plot_map_with_points(map_array):
    clicked_points = []

    def on_click(event):
        ix, iy = int(event.xdata), int(event.ydata)
        print(f"Coordinates: x={ix}, y={iy} || formated: [{iy}, {ix}]")

        # Clear the list and add the new coordinates
        clicked_points.append((iy, ix))

        # Plot an 'X' marker at the clicked coordinates
        plt.plot(ix, iy, marker='x', markersize=10, color='red')
        fig.canvas.draw()  # Update the figure to show the new marker

    def on_key(event):
        if event.key == 'enter':
            plt.close(fig)  # Close the figure window

    global fig, ax
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = matplotlib.colors.ListedColormap(['black', 'white'])
    ax.imshow(map_array, cmap=cmap)

    plt.legend()
    plt.show()

    # Connect the click event
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    # Connect the key press event to stop collecting points
    kid = fig.canvas.mpl_connect('key_press_event', on_key)

    # Return the list reference
    return clicked_points
    
    
#function to get the goal coordinates
def get_goal_cordinates(obstacle_map,goal_value=2):
    x_coords, y_coords = np.where(obstacle_map == goal_value)
    goal_grid_coordinates = np.column_stack((x_coords, y_coords))
    return goal_grid_coordinates


def sample_goals(map_grouped_goals, percentage):
    sampled_map_grouped_goals = {}

    for key, values in map_grouped_goals.items():
        # Calculate the number of elements to sample
        sample_size = max(int(len(values) * percentage / 100), 1)  # Ensure at least 1 element is sampled

        # Randomly sample elements
        sampled_elements = random.sample(values, sample_size)

        # Add the sampled elements to the new dictionary
        sampled_map_grouped_goals[key] = sampled_elements

    return sampled_map_grouped_goals


def sampled_visualize_obstacle_map(obstacle_map, sampled_dict, title,show_color_bar=True):
    # Ensure the obstacle_map is a numpy array for processing
    obstacle_map = np.array(obstacle_map)

    # Create a modified copy of the obstacle map to mark sampled goal points
    modified_map = np.copy(obstacle_map)

    # Mark sampled goal points in the modified map
    for key, points in sampled_dict.items():
        for point in points:
            # Assuming the points are in (row, column) format
            modified_map[point[0], point[1]] = 3  # Use a distinct value (e.g., 3) for sampled goal points

    # Create a color map: obstacles (black), free space (white), original goal points (also white), sampled goal points (gold)
    cmap = matplotlib.colors.ListedColormap(['black', 'white', 'white', 'gold'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]  # Boundaries for the colormap
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 10))  # Adjust size as needed

    # Plotting
    cax = ax.imshow(modified_map, cmap=cmap, norm=norm, interpolation='nearest')
    if show_color_bar:
        fig.colorbar(cax, ticks=[0, 1, 2, 3], format=plt.FuncFormatter(lambda x, _: {0: 'Obstacles', 1: 'Free Space', 2: 'Original Goal Points', 3: 'Sampled Goals'}[x]))
    ax.set_title(title)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(False)

    return fig  # Return the figure object