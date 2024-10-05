import numpy as np
import random

class RRTstar:
    
    """ Initialize the RRT* algorithm with given parameters. """
    def __init__(self, start, goal, state_validity_checker, max_iterations=10000, delta_q=2, p_goal=0.2, dominion=[-10, 10, -10, 10]):
        self.state_validity_checker = state_validity_checker 
        self.max_iterations = max_iterations # Maximum number of iterations.
        self.delta_q = delta_q # Step size.
        self.p_goal = p_goal # Probability of sampling the goal point.
        self.dominion = dominion # bounds of the state space.
        self.start = start # Start point.
        self.goal_p = goal # Goal point.
        self.nodes = [{'point': np.array(self.start)[:2], 'parent': None, 'cost': 0}] # List of nodes in the tree, starting with the initial node.
        self.invalid_nodes = set() # Set to store invalid nodes.
        self.tree_edges = [] # List to store tree edges.
    
    
    """ Create a random point in the bound space. """
    def create_random_point(self):
        # Extract the dominion boundaries.
        x_min, x_max, y_min, y_max = self.dominion
        # With probability 1 - p_goal, sample randomly in the space.
        if random.random() > self.p_goal:
            # Return a random point within the dominion.
            return np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)])
        # Otherwise with ith probability p_goal, return the goal point.
        return self.goal_p
    
    """ Find the nearest node in the tree to a given random point. """
    def find_nearest_node(self, random_point):
        # Initialize the nearest node as None.
        nearest_node = None
        # Initialize the minimum distance as infinity.
        min_distance = np.inf
        # Iterate over all nodes in the tree.
        for node in self.nodes:
            if node['point'] is not None:
                 # Calculate the Euclidean distance between the node and the random point.
                distance = np.linalg.norm(np.array(node['point'])[:2] - np.array(random_point))
                 # If the calculated distance is less than the minimum distance.
                if distance < min_distance:
                    # Update the nearest node.
                    nearest_node = node
                    # Update the minimum distance.
                    min_distance = distance
        # Return the nearest node.
        return nearest_node
    
    """ Determine a new point in the direction from nearest_point to random_point. """
    def determine_new_point(self, nearest_point, random_point):
        # Calculate the direction vector from nearest_point to random_point.
        direction = np.array(random_point) - np.array(nearest_point)
        # Calculate the distance between nearest_point and random_point.
        distance = np.linalg.norm(direction)
        # Normalize the direction vector.
        direction_vector = direction / distance 
        # If the distance is smaller than step size.
        if self.delta_q > distance:
            # Use the random_point as the new point.
            return random_point
        # Otherwise, step delta_q towards the random_point.
        return np.array(nearest_point) + direction_vector * self.delta_q
    
    """ Add a new node to the tree. """
    def add_new_node(self, new_point, parent_node, new_node_cost):
        #Check if the parent node exists and has a valid point.
        if parent_node and 'point' in parent_node:
            # Get the parent node's point.
            parent_point = parent_node['point']
            # Append the edge between the parent node and the new node to the tree edges.
            self.tree_edges.append((tuple(parent_point), tuple(new_point)))
            
        else:
             # If there is no valid parent, set parent_point to None.
            parent_point = None
             # Set the cost of the new node to 0.
            new_node_cost = 0
            
        # Create the new node with the given point, parent, and cost.
        self.nodes.append(new_node)  
        # Add the new node to the list of nodes.
        new_node = {'point': new_point, 'parent': parent_point, 'cost': new_node_cost}
        # Append the new node to the list of nodes.
        self.nodes.append(new_node)
        # Return the new node.
        return new_node
    
    """ Find a valid random point that is not in an obstacle. """
    def find_valid_random_point(self):
        # Continue sampling points until a valid point is found.
        while True:
             # Create a random point.
            qrand = self.create_random_point()
            # Check if the point is valid.
            if self.state_validity_checker.is_valid(qrand):
                # Return the valid random point.
                return qrand
    
    """ Check if the path segment between qnew and qnear is valid."""
    def check_segment(self, qnew, qnear):
        return self.state_validity_checker.check_path([qnew, qnear])
    
    """ Retrace the path from the goal node to the start node. """
    def retrace_path(self, goal_node):
        # Initialize the path with the goal node's point.
        path = [goal_node['point']]
        # Start retracing from the goal node.
        current_node = goal_node
        # Continue until the start node is reached.
        while current_node['parent'] is not None:
            # Find the parent node.
            parent_node = next((node for node in self.nodes if np.array_equal(node['point'], current_node['parent'])), None)
            # If the parent node is found.
            if parent_node is not None:
                # Append the parent node's point to the path.
                path.append(parent_node['point'])
                # Set the current node to the parent node.
                current_node = parent_node
            else:
                 # If no parent node is found, break the loop.
                break
         # Return the path in reverse order (from start to goal).
        return path[::-1]
    
    """ Check if two points are within a certain distance (tolerance) """
    def is_close(self, point_a, point_b, tolerance=0.01):
         # If either point is None. Return False
        if point_a is None or point_b is None:
            return False
        #Return True if the points are within the tolerance
        return np.linalg.norm(np.array(point_a) - np.array(point_b)) <= tolerance
    
    """ Smooth the path by removing unnecessary points. """
    def smooth_path(self, path, start, goal):
        # Initialize a counter to prevent infinite loops.
        counter = 0
        # Set a maximum number of iterations to prevent infinite loops.
        max_iterations = 100000
        # Initialize the smoothed path with the goal point.
        smoothed_path = [goal]
        
        # Continue until the path is smoothed or the maximum number of iterations is reached.
        while True:
            # Iterate over the path points in reverse order, excluding the last point.
            for point in reversed(path[:-1]):
                # Check if the segment between the current point and the last point in the smoothed path is valid.
                if self.state_validity_checker.check_path([point, smoothed_path[-1]]):
                    # Append the current point to the smoothed path.
                    smoothed_path.append(point)
                    # Break the loop to continue with the next point.
                    break
             # Check if the last point in the smoothed path is close to the start point.
            if self.is_close(smoothed_path[-1], start):
                break  # If close, break the loop.
            # Increment the counter.
            counter += 1
            # If the maximum number of iterations is reached.
            if counter >= max_iterations:
                # Set the smoothed path to the original path.
                smoothed_path = path
                break # Break the loop.
            
        # Reverse the smoothed path to start from the beginning.
        smoothed_path.reverse()
        # Return the smoothed path.
        return smoothed_path
    
    """ Find nodes within a certain distance (max_dist) from a new node. """
    def find_near_nodes(self, new_node, max_dist):
        # Initialize the list of near nodes.
        near_nodes = []
        # Iterate over all nodes in the tree.
        for node in self.nodes:
            # Check if the distance between the new node and the current node is within max_dist.
            if np.linalg.norm(new_node - node['point'][:2]) <= max_dist:
                # Append the current node to the list of near nodes.
                near_nodes.append(node)
        # Return the list of near nodes.
        return near_nodes
    
    """ Rewire the tree to include the new node. """
    def rewire(self, new_node, near_nodes):
        # Iterate over all near nodes.
        for node in near_nodes:
            # Calculate the new cost.
            new_cost = new_node['cost'] + np.linalg.norm(new_node['point'][:2] - node['point'][:2])
             # If the new cost is lower and the path segment is valid.
            if new_cost < node['cost'] and self.check_segment(new_node['point'][:2], node['point'][:2]):
                # Update the parent of the current node to the new node.
                node['parent'] = new_node['point']
                # Update the cost of the current node.
                node['cost'] = new_cost
                 # Update the costs of all descendant nodes.
                self.update_descendant_costs(node)
    
    """ Update the costs of all descendant nodes of a given node."""
    def update_descendant_costs(self, node):
        # Iterate over all children of the given node.
        for child in self.find_children(node):
            # Update the cost of the child node.
            child['cost'] = node['cost'] + np.linalg.norm(child['point'] - node['point'])
            self.update_descendant_costs(child) # Recursively update the costs of the descendants.
    
    
    """ Find all children of a given parent node. """
    def find_children(self, parent_node):
        return [node for node in self.nodes if np.array_equal(node['parent'], parent_node['point'])]

    
    """ Compute the path from start to the goal. """
    def compute_path(self, qgoal):
        # Convert the goal point to a numpy array.
        qgoal = np.array(qgoal)
        # Iterate for a maximum number of iterations.
        for _ in range(self.max_iterations):
            # Sample a valid random point.
            qrand = self.find_valid_random_point()
            # Find the nearest node in the tree.
            qnear = self.find_nearest_node(qrand)
            # Determine the new point to add to the tree.
            qnew = self.determine_new_point(qnear['point'][:2], qrand)
            # Check if the new point is valid.
            if self.check_segment(qnew, qnear['point'][:2]):
                # Find nearby nodes within a distance of 4.
                near_nodes = self.find_near_nodes(qnew, max_dist=4)
                # Calculate the cost to the new node.
                min_cost = qnear['cost'] + np.linalg.norm(qnew - qnear['point'])
                # Set the initial parent of the new node to the nearest node.
                qnew_parent = qnear
                # Set the initial cost of the new node to the minimum cost.
                qnew_cost = min_cost
                 # Iterate over all near nodes.
                for node in near_nodes:
                    # Check if the path to the nearby node is valid.
                    if self.check_segment(qnew, node['point']):
                        # Calculate the potential new cost.
                        potential_new_cost = node['cost'] + np.linalg.norm(qnew - node['point'])
                        # If the potential new cost is lower than the minimum cost.
                        if potential_new_cost < min_cost:
                            # Update the parent of the new node to the nearby node.
                            qnew_parent = node
                            # Update the cost of the new node.
                            qnew_cost = potential_new_cost
                            # Update the minimum cost.
                            min_cost = potential_new_cost
                # Add the new node to the tree.
                new_node = self.add_new_node(qnew, qnew_parent, qnew_cost)
                # Rewire the tree.
                self.rewire(new_node, near_nodes)
                # Check if the goal is reached.
                if np.linalg.norm(new_node['point'] - qgoal) <= self.delta_q and self.check_segment(new_node['point'], qgoal):
                    # Calculate the cost to the goal node.
                    goal_node_cost = new_node['cost'] + np.linalg.norm(qgoal - new_node['point'])
                    # Add the goal node to the tree.
                    goal_node = self.add_new_node(qgoal, new_node, goal_node_cost)
                    # Retrace the path from the goal node to the start node.
                    total_path = self.retrace_path(goal_node)
                     # Check if the entire path is valid.
                    if self.state_validity_checker.check_path(total_path):
                        # Return the valid path, tree edges, and nodes.
                        return total_path, self.tree_edges, self.nodes
                    # Get the start and goal points from the path.
                    start, goal = total_path[0], total_path[-1]
                    # Return the smoothed path, tree edges, and nodes.
                    return self.smooth_path(total_path, start, goal), self.tree_edges, self.nodes
        # Return an empty path, tree edges, and nodes if no valid path is found within the maximum iterations.
        return [], self.tree_edges, self.nodes