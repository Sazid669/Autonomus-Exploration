import math

def move_to_point(current, goal, Kv=0.5, Kw=0.5, reverse_threshold=math.pi/2):
    # Calculate the Euclidean distance to the goal.
    distance = math.sqrt((goal[0] - current[0]) ** 2 + (goal[1] - current[1]) ** 2)
    # Desired orientation to reach the goal from the current position.
    desire_oreientation = math.atan2(goal[1] - current[1], goal[0] - current[0])
    # Angular correction needed, wrapped within [-pi, pi].
    angular_correction = wrap_angle(desire_oreientation - current[2])

    # Determine the most efficient way to reach the goal (forward or backward).
    # If the absolute value of the angular correction is less than the reverse threshold,
    # it means moving forward is efficient. Otherwise, consider reversing.
    if abs(angular_correction) <= reverse_threshold:
        # Move forward
        v = Kv * distance
        w = Kw * angular_correction
    else:
        # Reverse movement
        # Adjust orientation for reverse; the goal is essentially "behind" the robot.
        angular_correction = wrap_angle(desire_oreientation - current[2] + math.pi)
        v = -Kv * distance  # Negative velocity for reverse
        w = Kw * angular_correction

    # Ensure robot is oriented correctly before moving forward or backward.
    if abs(angular_correction) > 0.05:
        v = 0  # Prioritize orientation correction

    return v, w

def pure_pursuit_controller(current_pose, goal_pose, lookahead_distance=0.1, vehicle_length=0.257):
    """
    Calculate the linear and angular velocity for a vehicle to reach a goal pose using the Pure Pursuit algorithm.
    
    Args:
    current_pose (tuple): The current pose of the vehicle as (x, y, theta).
    goal_pose (tuple): The goal pose of the vehicle as (x, y).
    lookahead_distance (float): The distance ahead of the current pose to calculate the lookahead point.
    vehicle_length (float): The length of the vehicle (wheelbase).

    Returns:
    tuple: A tuple containing the linear velocity (v) and angular velocity (w).
    """

    x, y, theta = current_pose
    if len(goal_pose) == 3:
        goal_x, goal_y,_ = goal_pose
    else:
        goal_x, goal_y = goal_pose

    # Calculate the Euclidean distance to the goal
    dx = goal_x - x
    dy = goal_y - y
    distance_to_goal = math.sqrt(dx**2 + dy**2)

    # Determine the lookahead point based on the distance to the goal
    if distance_to_goal < lookahead_distance:
        lookahead_x, lookahead_y = goal_x, goal_y
    else:
        scale = lookahead_distance / distance_to_goal
        lookahead_x = x + scale * dx
        lookahead_y = y + scale * dy

    # Compute the angle to the lookahead point
    angle_to_lookahead = math.atan2(lookahead_y - y, lookahead_x - x)

    # Compute the heading error angle (alpha)
    alpha = wrap_angle(angle_to_lookahead - theta)

    # Calculate the steering angle (delta)
    delta = math.atan2(2 * vehicle_length * math.sin(alpha), lookahead_distance)

    # Constant linear velocity
    v = 0.4

    # Calculate angular velocity based on the bicycle model
    w = (v / vehicle_length) * math.tan(delta)

    return v, w

# Helper function to wrap angles to [-pi, pi]
def wrap_angle(angle):
    """ Normalize the angle to be within the range of -pi to pi """
    return (angle + math.pi) % (2 * math.pi) - math.pi