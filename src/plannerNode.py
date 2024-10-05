#!/usr/bin/python3

import numpy as np
import time
import rospy
import tf
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA 
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray 
from std_msgs.msg import Bool
from utils_lib.StateValidityChecker import *
from utils_lib.dubins import *
from utils_lib.controller import *

class OnlinePlanner:

    # OnlinePlanner Constructor
    def __init__(self, gridmap_topic, odom_topic, cmd_vel_topic, bounds, distance_threshold):

       
        # State Validity Checker object                                                 
        self.svc = StateValidityChecker(distance_threshold)

        # Current robot pose [x, y, yaw], None if unknown            
        self.current_pose = None
        # Goal where the robot has to move, None if it is not set                                                                   
        self.goal = None
        # Last time a map was received (to avoid map update too often)                                                
        self.last_map_time = rospy.Time.now()
        # Dominion [min_x_y, max_x_y] in which the path planner will sample configurations                           
        self.bounds = bounds   
        
        # for visualization
        self.path = []
        self.tree = []

        # CONTROLLER PARAMETERS
        # Proportional linear velocity controller gain
        self.Kv = 0.5
        # Proportional angular velocity controller gain                   
        self.Kw = 0.5
        # Maximum linear velocity control action                   
        self.v_max = 0.15
        # Maximum angular velocity control action               
        self.w_max = 0.3                
        self.distance_threshold = 0.2
        # Current speed of the robot
        self.current_v = 0.0
        # current angular velocity
        self.current_w = 0.0
        # wheel encoder parameters
        self.wheel_radius = 0.035 # meters      
        self.wheel_base_distance = 0.257 # meters 
        # TIMERS
        # Timer for velocity controller
        rospy.Timer(rospy.Duration(0.1), self.controller)
        
        # PUBLISHERS
        # Publisher for sending velocity commands to the robot
        self.cmd_pub = rospy.Publisher(cmd_vel_topic, Float64MultiArray, queue_size=1)
        # Publisher for visualizing the path to with rviz
        self.marker_pub = rospy.Publisher('~path_marker', Marker, queue_size=1)
        #robot square publisher
        self.square_pub = rospy.Publisher('square', Marker, queue_size=10)
        # Publisher to indicate if the goal is reached
        self.goal_achieved_pub = rospy.Publisher('goal_reached', Bool, queue_size=10)
        # Publisher to visualize the tree
        self.tree_marker_pub = rospy.Publisher('tree', MarkerArray, queue_size=10)
        
        # SUBSCRIBERS
        # Subscriber to get the gridmap
        self.gridmap_sub = rospy.Subscriber(gridmap_topic, OccupancyGrid, self.get_gridmap)
        # Subscriber to get odometry data
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.get_odom)
        # Subscriber to get the move goal from rviz
        self.move_goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.get_goal)
        
        
        
    
    # Odometry callback: Gets current robot pose and stores it into self.current_pose
    def get_odom(self, odom):

        _, _, yaw = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x, 
                                                              odom.pose.pose.orientation.y,
                                                              odom.pose.pose.orientation.z,
                                                              odom.pose.pose.orientation.w])

      
        self.current_pose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, yaw])
        self.current_v = odom.twist.twist.linear.x
        self.current_w = odom.twist.twist.angular.z
        
    # Callback to get the goal pose
    def get_goal(self, goal):
        if self.svc.there_is_map:
            self.goal = np.array([goal.pose.position.x, goal.pose.position.y]) 
            
            print('New goal set to: ',self.goal)   
            if self.svc.is_valid(self.goal):   # Check if the goal is reachable
                self.path = []                                                   
                self.path = self.plan()
                self.goal_achieved = False
                goal_achieved_msg = Bool()
                goal_achieved_msg.data = self.goal_achieved

                # Publish the message
                self.goal_achieved_pub.publish(goal_achieved_msg)
            else:
                rospy.logwarn("Invalid goal position trying to find a valid position")
                self.goal = self.svc.find_valid_goal_cell(self.goal) # Works 99% of the time
                if self.goal is None:
                    rospy.logwarn("Failed to find a valid position sending the robot in a safe position")
                    self.goal = np.array([3.20514512, -0.58976054]) # Only for stonefish simulation environment 
                print(" New goal set to", self.goal)
                self.path = []                                                   
                self.path = self.plan()
                self.goal_achieved = False
                goal_achieved_msg = Bool()
                goal_achieved_msg.data = self.goal_achieved
                self.clear_tree_marker()

                # Publish the message
                self.goal_achieved_pub.publish(goal_achieved_msg)
    
    # Callback to get the gridmap
    def get_gridmap(self, gridmap):
      
        # To avoid map update too often (change value if necessary)
        if (gridmap.header.stamp - self.last_map_time).to_sec() > 1:            
            self.last_map_time = gridmap.header.stamp

            # Update State Validity Checker
            self.env = np.array(gridmap.data).reshape(gridmap.info.height, gridmap.info.width).T
            origin = [gridmap.info.origin.position.x, gridmap.info.origin.position.y]
            self.svc.set(self.env, gridmap.info.resolution, origin) 

            # If the robot is following a path, check if it is still valid
            if self.path is not None and len(self.path) > 0:
                # Create total_path adding the current position to the rest of waypoints in the path
                total_path = [self.current_pose[:2]] + self.path[:2]
                # Check total_path validity. If total_path is not valid make self.path = None and replan
                if not self.svc.check_path(total_path):
                    self.path = []
                    self.goal_achieved = False
                    goal_achieved_msg = Bool()
                    goal_achieved_msg.data = self.goal_achieved
                    # Publish the message
                    self.goal_achieved_pub.publish(goal_achieved_msg)
                    self.path = self.plan() 
                elif self.svc.is_valid(self.goal) == False:
                    self.goal = self.svc.set_new_goal(total_path)
                    rospy.logwarn("Goal became invalid, setting new goal %s", self.goal)
                    self.goal_achieved = False
                    goal_achieved_msg = Bool()
                    goal_achieved_msg.data = self.goal_achieved
                    # Publish the message
                    self.goal_achieved_pub.publish(goal_achieved_msg)
                    self.path = self.plan()

    # Function to check and handle invalid position
    def check_and_handle_invalid_position(self):
        if not self.svc.is_valid(self.current_pose[0:2]):
            rospy.logwarn("Invalid current position, obstacle ahead")
            # Move backward to avoid the obstacle
            start_time = time.time()
            while time.time() - start_time < 1.0:
                self.__send_command__(-0.5, 0.0)
            self.__send_command__(0.0, 0.0)
            # Clear the existing path and re-plan
            del self.path[:]
            self.plan()
    
    # Function to plan the path
    def plan(self):
        
        count = 0
        # If planning fails, allow replanning for several counts
        while len(self.path) <= 1 and count < 5:
            print("Computing new path") 
            self.path, self.tree, self.nodes = compute_path(self.current_pose, self.goal, self.svc, self.bounds)
            
            self.publish_path(self.path)
            self.tree_marker()
            
            # Calculate the final orientation angle
            delta_x = self.current_pose[0] - self.goal[0]
            delta_y = self.current_pose[1] - self.goal[1]
            angle_rad = wrap_angle(math.atan2(delta_y, delta_x)) 
            final_orientation = angle_rad + math.pi

            # Add orientation angle to each waypoint in the path
            self.path = adding_theta(self.path, self.current_pose[2], final_orientation) 
            turning_radius = 0.2
            step_size = 0.1
            # Smooth the path using Dubins path smoothing algorithm
            self.path = dubins_smooth(turning_radius, step_size, self.path)  
                                      
            count += 1

            if self.path == []:
                # If no path is found, set goal_achieved to True and publish the message
                self.goal_achieved = True
                goal_achieved_msg = Bool()
                goal_achieved_msg.data = self.goal_achieved
                self.goal_achieved_pub.publish(goal_achieved_msg)
                            
        if count == 5:
            # If planning fails after 5 attempts, set goal_achieved to True and publish the message
            print("Path not found!") 
            self.goal_achieved = True
            goal_achieved_msg = Bool()
            goal_achieved_msg.data = self.goal_achieved
            self.goal_achieved_pub.publish(goal_achieved_msg)
        else:
            # If path is found, set goal_achieved to False and publish the message
            print("Robot's Path found")
            self.goal_achieved = False
            goal_achieved_msg = Bool()
            goal_achieved_msg.data = self.goal_achieved
            self.goal_achieved_pub.publish(goal_achieved_msg)
            
            # Publish plan marker to visualize in rviz
            self.publish_path(self.path) 
            # Remove initial waypoint in the path (current pose is already reached)
            del self.path[0]  
            
        return self.path 

    # Velocity controller
    def controller(self, event):

        if self.current_pose is None or not self.svc.there_is_map:
            return

        self.check_and_handle_invalid_position()
        v = 0
        w = 0
        distance_to_waypoint = float('inf') 
        
        if len(self.path) > 0:
            waypoint = self.path[0]
            v, w = pure_pursuit_controller(self.current_pose, waypoint)
            
            distance_to_waypoint = np.sqrt((self.current_pose[0] - waypoint[0])**2 + (self.current_pose[1] - waypoint[1])**2)
        
            if distance_to_waypoint < self.distance_threshold:
                # Move to the next waypoint
                del self.path[0]
                if len(self.path) > 0:
                    waypoint = self.path[0]  # Update to the new current waypoint
                    v, w = pure_pursuit_controller(self.current_pose, waypoint, 0.06)
                    self.goal_achieved = False
                    goal_achieved_msg = Bool()
                    goal_achieved_msg.data = self.goal_achieved
                    self.goal_achieved_pub.publish(goal_achieved_msg)
                else:
                    print("Destination reached")
                    v = 0
                    w = 0
                    self.goal_achieved = True
                    goal_achieved_msg = Bool()
                    goal_achieved_msg.data = self.goal_achieved
                    self.goal_achieved_pub.publish(goal_achieved_msg)
                    self.clear_tree_marker()
             
        # Publish velocity commands
        self.__send_command__(v, w)
        
    # Send velocity commands to the robot
    def __send_command__(self, v, w):
        rate = rospy.Rate(100)   
        move = Float64MultiArray() 
       
        v_l = (2 * v + w * self.wheel_base_distance) / (2 * self.wheel_radius)
        v_r = (2 * v - w * self.wheel_base_distance) / (2 * self.wheel_radius) 
        v_l = v_l/5
        v_r = v_r/5
        move.data = [v_l, v_r]   
        self.cmd_pub.publish(move)

    # Publish a path as a series of line markers
    def publish_path(self, path):
        if len(path) > 1:
            print("Publish path!")
            m = Marker()
            m.header.frame_id = 'world_ned'
            m.header.stamp = rospy.Time.now()
            m.id = 0
            m.type = Marker.LINE_STRIP
            m.ns = 'path'
            m.action = Marker.DELETE
            m.lifetime = rospy.Duration(0)
            self.marker_pub.publish(m)

            m.action = Marker.ADD
            m.scale.x = 0.1
            m.scale.y = 0.0
            m.scale.z = 0.0
            
            m.pose.orientation.x = 0
            m.pose.orientation.y = 0
            m.pose.orientation.z = 0
            m.pose.orientation.w = 1
            
            color_red = ColorRGBA()
            color_red.r = 1
            color_red.g = 0
            color_red.b = 0
            color_red.a = 1
            color_blue = ColorRGBA()
            color_blue.r = 0
            color_blue.g = 0
            color_blue.b = 1
            color_blue.a = 1

            p = Point()
            p.x = self.current_pose[0]
            p.y = self.current_pose[1]
            p.z = 0.0
            m.points.append(p)
            m.colors.append(color_blue)
            
            for n in path:
                p = Point()
                p.x = n[0]
                p.y = n[1]
                p.z = 0.0
                m.points.append(p)
                m.colors.append(color_red)
            
            self.marker_pub.publish(m)

    # Publish tree marker
    def tree_marker(self):
        if not self.tree:
            return

        tree_marker_array = MarkerArray()
        node_marker_array = MarkerArray()

        for i, edge in enumerate(self.tree):
            start, end = edge

            tree_marker = Marker()
            tree_marker.header.frame_id = 'world_ned'
            tree_marker.header.stamp = rospy.Time.now()
            tree_marker.ns = 'rrt_star_tree'
            tree_marker.id = i + 2
            tree_marker.type = Marker.LINE_STRIP
            tree_marker.action = Marker.ADD
            tree_marker.scale.x = 0.02
            tree_marker.pose.orientation.x = 0
            tree_marker.pose.orientation.y = 0
            tree_marker.pose.orientation.z = 0
            tree_marker.pose.orientation.w = 1
            tree_marker.color = ColorRGBA(0, 0, 1, 1)

            start_point = Point()
            start_point.x, start_point.y = start
            end_point = Point()
            end_point.x, end_point.y = end

            tree_marker.points.append(start_point)
            tree_marker.points.append(end_point)

            tree_marker_array.markers.append(tree_marker)
            
            # Adding nodes as points
            node_marker = Marker()
            node_marker.header.frame_id = 'world_ned'
            node_marker.header.stamp = rospy.Time.now()
            node_marker.ns = 'rrt_star_nodes'
            node_marker.id = i + 1002
            node_marker.type = Marker.SPHERE
            node_marker.action = Marker.ADD
            node_marker.scale.x = 0.05
            node_marker.scale.y = 0.05
            node_marker.scale.z = 0.05
            node_marker.pose.orientation.x = 0
            node_marker.pose.orientation.y = 0
            node_marker.pose.orientation.z = 0
            node_marker.pose.orientation.w = 1
            node_marker.color = ColorRGBA(0, 1, 0, 1)  # Red color for nodes

            node_point = Point()
            node_point.x, node_point.y = start
            node_point.x,node_point.y=end
            node_marker.pose.position = node_point

            node_marker_array.markers.append(node_marker)

        self.tree_marker_pub.publish(tree_marker_array)
        self.tree_marker_pub.publish(node_marker_array)
        rospy.loginfo("Published RRTInformed tree with {} edges and {} nodes".format(len(self.tree), len(node_marker_array.markers)))

    # Clear tree markers from RViz
    def clear_tree_marker(self):
        tree_marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        tree_marker_array.markers.append(clear_marker)
        self.tree_marker_pub.publish(tree_marker_array)
            
# MAIN FUNCTION
if __name__ == '__main__':
    rospy.init_node('turtlebot_online_path_planning_node')   
    node = OnlinePlanner('/projected_map', '/turtlebot/kobuki/odom', '/kobuki/commands/wheel_velocities', np.array([-10.0, 10.0, -10.0, 10.0]), 0.2)
    
    # Run forever
    rospy.spin()
