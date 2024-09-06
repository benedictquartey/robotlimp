# Modified FMT* planner
# Code originally adapted from: https://github.com/yonetaniryo/fmt/blob/main/README.md || Author: Ryo Yonetani

import math

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from pqdict import pqdict

class FMTPlanner():
    def __init__(
        self,
        n_samples: int = 1000,
        r_n: float = 20.0,
        path_resolution: float = 0.1,
        rr: float = 1.0,
        max_search_iter: int = 10000,
        seed: int = 0,
    ):
        """
        Fast Marching Tree Path Planner 

        Args:
            n_samples (int, optional): Number of nodes to sample. Defaults to 1000.
            r_n (float, optional): Range to find neighbor nodes. Defaults to .0.
            path_resolution (float, optional): Resolution of paths to check collisions. Defaults to 0.1.
            rr (float, optional): Distance threshold to check collisions. Defaults to 1.0.
            max_search_iter (int, optional): Number of maximum iterations. Defaults to 10000.
            seed (int, optional): Random seed. Defaults to 0.
        """

        # hyperparameters
        self.path_resolution = path_resolution
        self.rr = rr
        self.n_samples = n_samples
        self.r_n = r_n
        self.max_search_iter = max_search_iter
        self.prng = np.random.RandomState(seed)  # initialize PRNG


    def add_map_design(self, map_design: np.ndarray) -> None:
        """
        Add map design to the planner

        Args:
            map_design (np.ndarray): Obstacle map described by a binary image. 1: free nodes; 0: obstacle nodes
        """
        
        self.map_size = map_design.shape

        # construct obstacle tree
        obstacles = np.argwhere(map_design == 0)
        self.obstacles_tree = cKDTree(obstacles)

        # initialize graph
        self.graph = nx.Graph()
        self.node_list = list()
        i = 0
        while len(self.node_list) < self.n_samples:
            node = self.prng.uniform(0, self.map_size)
            if self.check_collision(node, None):
                self.node_list.append(node)
                self.graph.add_node(i)
                i += 1


    def calculate_heuristic(self, node_list, goal_list):
        """
        Calculate the heuristic for each node as the minimum distance to any goal.

        :param node_list: An array of node coordinates.
        :param goal_list: An array of goal coordinates.
        :return: A numpy array containing the heuristic for each node.
        """
        # Convert lists to numpy arrays if they are not already
        node_array = np.array(node_list)
        goal_array = np.array(goal_list)

        # Calculate the differences between each node and each goal
        diffs = node_array[:, np.newaxis, :] - goal_array[np.newaxis, :, :]

        # Calculate the Euclidean distance for each difference
        distances = np.linalg.norm(diffs, axis=2)

        # Find the minimum distance to a goal for each node
        heuristic = np.min(distances, axis=1)

        return heuristic

    def plan(self,
             start: np.ndarray,
             sampled_goals: dict,
             heuristic_weight: int = 0.0) -> dict:
        """
        Run path planning

        Args:
            start (np.ndarray): Start location
            goal (np.ndarray): Dictionary with goal identifiers (predicate_index) and their corresponding goal points
            heuristic_weight (int, optional): Weight for Euclidean heuristics. Defaults to 0.0.

        Returns:
            dict: Containing path to the first reached goal, number of steps required, and goal flag
        """
        # Flatten the sampled_dict to a list of goals and create a mapping to their identifiers
        goals = []
        goal_identifiers = {}
        for identifier, points in sampled_goals.items():
            for point in points:
                goals.append(point)
                goal_identifiers[tuple(point)] = identifier  # Store the identifier for each goal point

        print(f"Start: {list(start)}, Num Goals: {len(goals)}")
        start = np.asarray(start)
        assert self.check_collision(start, None) # check_collision for start node

        # Add start node to graph
        start_id = len(self.node_list)
        self.graph.add_node(start_id)
        self.node_list.append(start)

        # Add goal nodes to graph and create a set for easy checking
        goal_ids = set()
        goals = np.asarray(goals)
        used_goals = []
        for idx,goal in enumerate(goals):
            goal = np.asarray(goal)  # Ensure each goal is a numpy array
            try :
                assert self.check_collision(goal, None)
                used_goals.append(goal)
                goal_id = len(self.node_list)
                self.graph.add_node(goal_id)
                self.node_list.append(goal)
                goal_ids.add(goal_id)
            except AssertionError:
                continue

        print(f"Num Goals after collision checking: {len(used_goals)}")

        # Construct KDTree with all nodes
        node_tree = cKDTree(self.node_list)

        # Calculate heuristic as the minimum distance to any goal || for each node the heuristic is the distance to the nearest goal.
        heuristic = self.calculate_heuristic(self.node_list, used_goals)
        
        # Initialize
        goal_flag = 0
        z = start_id
        V_open = pqdict({z: 0.})
        V_closed = list()
        V_unvisited = set(range(len(self.node_list)))
        V_unvisited.remove(z)

        # start search
        print("Starting search...")
        for n_steps in range(self.max_search_iter):
            if z in goal_ids:
                print(f"Reached a goal: {self.node_list[z]}")
                goal_flag = 1
                break
            N_z = node_tree.query_ball_point(self.node_list[z], self.r_n)
            X_near = list(set(N_z) & V_unvisited)
            for x in X_near:
                N_x = node_tree.query_ball_point(self.node_list[x], self.r_n)
                Y_near = list(set(N_x) & set(V_open))
                y_min = Y_near[np.argmin([V_open[y] for y in Y_near])]
                if self.check_collision(self.node_list[y_min], self.node_list[x]):
                    edge_cost = np.linalg.norm(self.node_list[y_min] - self.node_list[x]) #edge cost is the euclidien distance between the two nodes
                    self.graph.add_edge(y_min, x, weight=edge_cost)
                    cost = (V_open[y_min] +
                            np.linalg.norm(self.node_list[y_min] - self.node_list[x]) +
                            heuristic_weight * (-heuristic[y_min] + heuristic[x]))
                    if x in V_open:
                        V_open.updateitem(x, cost)
                    else:
                        V_open.additem(x, cost)
                    V_unvisited.remove(x)
            V_open.pop(z)
            V_closed.append(z)
            if len(V_open) == 0:
                print("Search failed")
                break
            z = V_open.top()

        if goal_flag:
            path_nodes = nx.shortest_path(self.graph, start_id, z)
            path = np.vstack([self.node_list[x] for x in path_nodes])
            total_cost = nx.shortest_path_length(self.graph, start_id, z, weight='weight')
            # Record the identifier of the goal reached
            goal_origin_predicate = goal_identifiers.get(tuple(self.node_list[z]), "Unknown")
        else:
            goal_origin_predicate = "Unknown"
            path = np.array([])
            total_cost = 0

        return {
            "path": path,
            "cost": total_cost,
            "n_steps": n_steps,
            "goal_flag": goal_flag,
            "goal": self.node_list[z],
            "goal_origin_predicate": goal_origin_predicate
        }

    def check_collision(self, src: np.ndarray, dst: np.ndarray) -> bool:
        """
        Check collision

        Args:
            src (np.ndarray): Source node
            dst (np.ndarray): Destination node

        Returns:
            bool: True if no collisions were found and False otherwise. (True if all pathpoints are farther than rr from the nearest obstacle). It returns False otherwise.
        """
        pr = self.path_resolution
        if (dst is None) | np.all(src == dst):
            return self.obstacles_tree.query(src)[0] > self.rr #checks if distance to nearest obstacle is greater than robot radius

        dx, dy = dst[0] - src[0], dst[1] - src[1]
        yaw = math.atan2(dy, dx)
        d = math.hypot(dx, dy)
        steps = np.arange(0, d, pr).reshape(-1, 1)
        pts = src + steps * np.array([math.cos(yaw), math.sin(yaw)])
        pts = np.vstack((pts, dst))
        return bool(self.obstacles_tree.query(pts)[0].min() > self.rr)
