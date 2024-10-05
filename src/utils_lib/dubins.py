import numpy as np
import math
from math import sqrt
import dubins
from utils_lib.RRTInformed import *
from utils_lib.RRTstar import *
import time

def dubins_smooth(turning_radius, step_size, path):
    if not path:
        return []

    dubins_path = []

    # Iterate through the waypoints to generate Dubins paths
    for start, end in zip(path[:-1], path[1:]):
        start_array = np.array(start)
        end_array = np.array(end)

        # Generate the Dubins path for the current segment
        segment_path, _ = dubins.path_sample(start_array, end_array, turning_radius, step_size)

        # Wrap the angles and store the path segments
        dubins_path.extend((x, y, wrap_angle(theta)) for x, y, theta in segment_path)

    return dubins_path

def adding_theta(path, initial_angle, final_angle):
    if not path:
        return []

    # Initialize the list of angles with the initial angle
    angles = [wrap_angle(initial_angle)]

    # Calculate the angles for intermediate points
    for current, next_point in zip(path[:-2], path[1:-1]):
        dx = next_point[0] - current[0]
        dy = next_point[1] - current[1]
        angle = wrap_angle(np.arctan2(dy, dx))
        angles.append(angle)

    # Append the final angle
    angles.append(wrap_angle(final_angle))

    # Create the path with theta
    path_with_theta = []
    for point, angle in zip(path, angles):
        path_with_theta.append((point[0], point[1], angle))

    return path_with_theta

def compute_path(start_p, goal_p, state_validity_checker, bounds, algorithm='RRTInformed'):
    # Create instances of the RRT variants with the given parameters
    rrt_informed = RRTInformed(start_p, goal_p, state_validity_checker=state_validity_checker, max_iterations=10000, delta_q=2, p_goal=0.2, dominion=bounds)
    rrt_star = RRTstar(start_p, goal_p, state_validity_checker=state_validity_checker, max_iterations=10000, delta_q=2, p_goal=0.2, dominion=bounds)
    
    # Select the appropriate RRT algorithm instance
    if algorithm == 'RRTInformed':
        rrt_instance = rrt_informed
    
    elif algorithm == 'RRTStar':
        rrt_instance = rrt_star

    else:
        raise ValueError("Unknown algorithm specified. Choose from 'RRTInformed', or 'RRTStar'.")
     # Measure computation time
    start_time = time.time()

    # Compute the path
    if algorithm == 'RRTInformed' or algorithm == 'RRTStar':
        path, tree, node = rrt_instance.compute_path(goal_p)
   
    end_time = time.time()
    "Computational time"
    computation_time = end_time - start_time

    """ Calculate path length """
    path_length = calculate_path_length(path)

    return path, tree, node

""" Calculate path length"""
def calculate_path_length(path):
    if not path or len(path) < 2:
        return 0
    path_length = 0
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        # Calculate the Euclidean distance between consecutive points and add it to the total path length.
        path_length += sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return path_length

def wrap_angle(angle):
    #corrects an angle to be within the range of [-pi, pi]
    return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )

