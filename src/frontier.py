#!/usr/bin/env python3
import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header, Bool
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseArray, Pose, Point, PoseStamped
from sklearn.cluster import DBSCAN
import tf
from utils_lib.StateValidityChecker import StateValidityChecker
import scipy.stats as stats

class FrontierExploration:
    def __init__(self):
        self.goal = None
        self.goal_reached = True
        self.resolution = None
        self.distance_threshold = 0.3
        self.svc = StateValidityChecker(self.distance_threshold)
        self.current_pose = None
        self.clusters = {}
        self.sorted_clusters=None

        ########################################### Publishers ###############################################################
        self.frontier_pub = rospy.Publisher('/frontier_points', PoseArray, queue_size=10)
        self.cluster_pub = rospy.Publisher('/frontier_clusters', MarkerArray, queue_size=10)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.cluster_marker = rospy.Publisher('/frontier_clusters_marker', Marker, queue_size=10)
        self.last_map_time = rospy.Time.now()
        
        #################################### Subscribers #################################
        self.gridmap_sub = rospy.Subscriber('/projected_map', OccupancyGrid, self.get_gridmap, queue_size=10)
        self.odom_sub = rospy.Subscriber('/turtlebot/kobuki/odom', Odometry, self.get_odom)
        self.goal_reach=rospy.Subscriber('/goal_reached', Bool, self.reached)
    
    """"" Check Planner reaches to the best viewpooint or not"""
    def reached(self,msg):
        # If the goal message is False that means the planner is not reached to the best viewpoint yet.
        if msg.data == False:
            self.goal_reached = False
        else:
            # If the goal message is True that means the planner is reached to the best viewpoint.
            self.goal_reached = True
            # Call the clear_clusters and frontiers method.
            self.clear_clusters_and_frontiers()
           

    def wrap_angle(self,angle):
        #corrects an angle to be within the range of [-pi, pi]
        return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )
    
    """Clear Cluster and Frontiers when the planner is reached to the best viewpoint"""         
    def clear_clusters_and_frontiers(self):
        self.clear_clusters()
        self.clear_frontiers()
    def get_gridmap(self, gridmap):
        if (gridmap.header.stamp - self.last_map_time).to_sec() > 1:
            self.last_map_time = gridmap.header.stamp
            env = np.array(gridmap.data).reshape(gridmap.info.height, gridmap.info.width).T
            self.map = env
            self.origin = [gridmap.info.origin.position.x, gridmap.info.origin.position.y]
            self.resolution = gridmap.info.resolution
            # Update State Validity Checker
            self.svc.set(env, self.resolution, self.origin)

            # Call the Frontier Function.
            frontiers = self.get_frontiers(self.map)
            """If the planner is reached to the goal (best viewpoint)"""
            if self.goal_reached:
                # Publish the detected frontier points using a ROS publisher
                self.frontier_publish(frontiers)
                # Call the Cluster Function.
                clusters=self.cluster_frontiers(frontiers)
                # Publish the clusters using a ROS publisher.
                self.publish_clusters(clusters)
                # Call the Explore Function.
                self.explore(clusters)

    """" Odometry callback: Gets current robot pose and stores it into self.current_pose"""
    def get_odom(self, odom):

        _, _, yaw = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x, 
                                                              odom.pose.pose.orientation.y,
                                                              odom.pose.pose.orientation.z,
                                                              odom.pose.pose.orientation.w])

      
        self.current_pose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, self.wrap_angle(yaw)])
        self.current_v = odom.twist.twist.linear.x
        self.current_w = odom.twist.twist.angular.z
        
   ########################################### Identifies frontier points #########################################   
    """Frontier points"""     
    def get_frontiers(self, gridmap):
       
        # Initialize a list to store the locations of frontier points.
        frontiers = []
        # Total number of rows in the grid map
        row=gridmap.shape[0]
        # Total number of columns in the grid map
        col=gridmap.shape[1]
        # Iterate over each cell in the grid map
        for i in range(row):
            for j in range(col):
                
                 # Check if the current cell is free
                if gridmap[i, j] == 0:
                    # Define the coordinates of the neighboring cells surrounding the current cell
                    neighbors = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1),
                                (i + 1, j + 1), (i + 1, j - 1), (i - 1, j + 1),(i - 1, j - 1)]
                    # Check each neighbor to determine if at least one is an unknown space
                    for ni_row, nj_col in neighbors:
                        # Ensure the neighbor indices are within the bounds of the grid map and  Check if the neighboring cell is unknown
                        if 0 <= ni_row < row and 0 <= nj_col < col and gridmap[ni_row, nj_col] == -1:
                            # If adjacent to an unknown space, classify the current cell as a frontier point
                            frontiers.append((i,j))  
                            break
        return frontiers
    
    """Publishing the frontiers."""
    def frontier_publish(self, frontier_points):
        # Create a new PoseArray object to store all frontier poses
        pose_array = PoseArray()
        pose_array.header.frame_id = "world_ned"
        pose_array.header.stamp = rospy.Time.now()
         # Iterate over each point in the list of frontier points
        for point in frontier_points:
            # Convert the grid map coordinates (i, j) to real-world coordinates (x, y)
            mapped_point = self.svc.map_to_position(point)
            # Create a new Pose object to store the position
            pose = Pose()
            pose.position.x = mapped_point[0]
            pose.position.y = mapped_point[1]
            pose_array.poses.append(pose)
        # Publish the PoseArray to the ROS topic frontier_pub
        self.frontier_pub.publish(pose_array)
    
    """""Clear the frontiers."""
    def clear_frontiers(self):
        empty_pose_array = PoseArray()
        empty_pose_array.header.frame_id = "world_ned"
        empty_pose_array.header.stamp = rospy.Time.now()
        self.frontier_pub.publish(empty_pose_array)
        

    ##################################################### Clustering #################################################################
   
 
    def cluster_frontiers(self, frontier_points):
        
        "Using DBSCAN (Density-Based Spatial Clustering of Applications with Noise) to cluster the frontier points.  This method used to find high-density areas  and distinguish them from regions of low density (outliers)."
        
        "Parameter Slection"
        "eps: Maximum distance between two frontier points for one to be considered as in the neighborhood of the other"
        "min_samples: Minimum number of points required to form a dense region"
        
        # Return an empty dictionary if no frontier points are provided
        if not frontier_points:
            return {}
        
        frontier_points= np.array(frontier_points)  
        dbscan = DBSCAN(eps=3, min_samples=4)  # Initialize DBSCAN
        dbscan.fit(frontier_points)  # Apply DBSCAN to the frontier points
        
        labels = dbscan.labels_
        clusters = {}
        for label, point in zip(labels, frontier_points):
            if label != -1:  # Exclude noise points
                clusters.setdefault(label, []).append(point)
        return clusters
        
        
    """ Publish Clusters """
    def publish_clusters(self, clusters):
        marker_array = MarkerArray()
        
        for idx, cluster in clusters.items():
            # Initialize a marker for displaying the points in the cluster
            point_marker = Marker()
            point_marker.header.frame_id = "world_ned"
            point_marker.header.stamp = rospy.Time.now()
            point_marker.ns = "frontier_clusters"
            point_marker.id = idx
            point_marker.type = Marker.POINTS
            point_marker.action = Marker.ADD
            point_marker.scale.x = 0.08 # Width of the marker points
            point_marker.scale.y = 0.08 # Height of the marker points
            point_marker.color.r, point_marker.color.g, point_marker.color.b, point_marker.color.a = np.random.rand(), np.random.rand(), np.random.rand(), 1.0

            # Create a text marker for each cluster to display labels
            text_marker = Marker()
            text_marker.header.frame_id = "world_ned"
            text_marker.header.stamp = rospy.Time.now()
            text_marker.ns = "frontier_cluster_labels"
            text_marker.id = idx
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.scale.z = 0.3
            text_marker.color.r, text_marker.color.g, text_marker.color.b, text_marker.color.a = 0.0, 0.0, 1.0, 1.0

            # Transform each point and collect for averaging and displaying
            transformed_points = [self.svc.map_to_position(point) for point in cluster]
            avg_x = np.mean([p[0] for p in transformed_points])
            avg_y = np.mean([p[1] for p in transformed_points])
            text_marker.pose.position.x = avg_x
            text_marker.pose.position.y = avg_y
            text_marker.text = f"Cluster {idx}"
            # Append each transformed point to the point marker
            for point in transformed_points:
                p = Point(x=point[0], y=point[1], z=0)
                point_marker.points.append(p)

            # Append both the points and text markers to the MarkerArray
            marker_array.markers.append(point_marker)
            marker_array.markers.append(text_marker)

        # Publish the MarkerArray to the ROS topic
        self.cluster_pub.publish(marker_array)

    """ Clear the clusters."""
    def clear_clusters(self):
        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        self.cluster_pub.publish(marker_array)
        rospy.loginfo("Cleared Clusters.")
        self.clusters.clear()



    ################################# Explore ####################################
  
    def explore(self, clusters):
        # Check if there are any clusters provided; if not, attempt to create new clusters
        if not clusters:
            rospy.loginfo("No clusters available for exploration. Attempting to re-cluster.")
            self.recluster_and_explore()  # Call the method to recluster and explore again
            return  
        # Update the clusters with the provided clusters for exploration
        self.clusters = clusters
        # Calculate the minimum distances from the robot current position to each cluster.
        min_distances = self.calculate_min_distances(clusters)
        # Calculate the maximum length of each cluster.
        max_length = self.calculate_maximum_length_of_clusters(clusters)

        """Calculate scores for each cluster based on the distances and lengths, and select the best target cluster. """
        target_cluster, target_cluster_points = self.calculate_scores_and_select_target(min_distances, max_length, clusters)

        # Check if there are any points in the selected target cluster
        if target_cluster_points:
            """Determine the best viewpoint within the target cluster based on maximum information gain. """
            best_view_point = self.maximum_information_gain(target_cluster_points)
            rospy.loginfo(f"Selected best viewpoint, {best_view_point}") 

            # Navigate to the chosen viewpoint within the selected cluster.
            self.navigate_to_viewpoint(best_view_point, target_cluster)
        else:
            # If no points are available in the target cluster, log the situation and re-evaluate clusters
            rospy.loginfo("No target cluster points available for navigation. Re-evaluating clusters.")
            self.recluster_and_explore()  # Recluster and attempt to explore again


    """ Recluster and explore."""
    def recluster_and_explore(self):
        frontiers = self.get_frontiers(self.map)
        # Check if any new frontiers were found; if none, exploration may be complete and return.
        if not frontiers:
            rospy.logerr("No new frontiers found, exploration might be complete.")
            return  

        clusters = self.cluster_frontiers(frontiers)
        # Check if the clustering iss successful.
        if clusters:
            # If clusters are found, publish them to allow for visualization or monitoring.
            self.publish_clusters(clusters)
            # Call explore with the new clusters to continue the exploration process.
            self.explore(clusters)
            
        else:
            # If no valid clusters could be formed, log that clustering failed.
            rospy.loginfo("Failed to identify new clusters with current data.")


    """"Publish the Best viewpoint for Visualization in Rviz """""
    def publish_viewpoint_marker(self, viewpoint, marker_id=0):
        marker = Marker()
        marker.header.frame_id = "world_ned"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "viewpoint_markers"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = viewpoint[0]
        marker.pose.position.y = viewpoint[1]
        marker.pose.position.z = 0  
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2  
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color = ColorRGBA(0.0, 1.0, 0.0, 1.0)  # Green color with full opacity

        # Create the text marker
        text_marker = Marker()
        text_marker.header.frame_id = "world_ned"
        text_marker.header.stamp = rospy.Time.now()
        text_marker.ns = "viewpoint_markers"
        text_marker.id = marker_id + 1  # Ensure a unique ID for the text marker
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = viewpoint[0]
        text_marker.pose.position.y = viewpoint[1]+0.9
        text_marker.pose.position.z = 0.5 # Slightly above the sphere
        text_marker.pose.orientation.x = 0.0
        text_marker.pose.orientation.y = 0.0
        text_marker.pose.orientation.z = 0.0
        text_marker.pose.orientation.w = 1.0
        text_marker.scale.z = 0.3 # Text size
        text_marker.color = ColorRGBA(1.0, 1.0, 0.0, 1.0)  
        text_marker.text = "Best Viewpoint"  # The text to display

        self.cluster_marker.publish(marker)
        self.cluster_marker.publish(text_marker)
        rospy.loginfo("Published viewpoint marker at position: ({}, {})".format(viewpoint[0], viewpoint[1]))


    """Calculate the length of each cluster."""
    def calculate_maximum_length_of_clusters(self,clusters):
        
        cluster_length= {label: len(cluster) for label, cluster in clusters.items()}
        return cluster_length
          
         
    """ Calculate the Euclidean distance between the current position and the target position."""

    def calculate_travel_cost(self, current_position, target_position):
        
        return np.linalg.norm(np.array(current_position) - np.array(target_position))
    

    """ Calculate the minimum distances from the robot current position to each cluster. """
    def calculate_min_distances(self,clusters):
        # Check if the current pose is available for distance calculations.
        if self.current_pose is None:
            rospy.loginfo("Current pose is not available for distance calculations.")
            return {}
       # Initialize a dictionary to store the minimum distance to each cluster.
        min_distances = {}
        #Current Position.
        current_pose=self.current_pose[:2]
        # Iterate over each cluster to calculate its minimum distance to the current position.
        for label, cluster in clusters.items():
            # Iniialize the minimum distance to infinity.
            min_distance = np.inf

            for point in cluster:
                mapped_point = np.array(self.svc.map_to_position(point))
                # Calculate the travel cost by usng the travel method.
                distance = self.calculate_travel_cost(current_pose, mapped_point)
                if distance < min_distance:
                    min_distance = distance
            min_distances[label] = min_distance
            
        return min_distances
   

    """" Calculate the score for getting the best score. """
    def calculate_scores_and_select_target(self, min_distances, max_length, clusters):
        # Check for necessary data to compute scores, raise an error if any is missing.
        if not clusters or not min_distances or not max_length:
            raise ValueError("Insufficient data to compute scores.")

        scores = {}  # Dictionary to hold the scores calculated for each cluster
        # Adjust this value to prioritize length over distance
        length_weight = 0.6
        # Adjust this value to prioritize distance over length
        distance_weight = 0.4
        # Iterate over each cluster to calculate its score based on provided metrics.
        for label, points in clusters.items():
            # Ensure there are points  for the cluster to compute its score.
            if points and label in min_distances and label in max_length:
                 # Score based on max length of the cluster
                num_points_score = max_length[label] 
                # Score based on minimum distance.
                distance_score = min_distances[label] 
                # Calculate the total score for the cluster and store it
                scores[label] = (num_points_score * length_weight) + (distance_score * distance_weight)
        #If no scores are provided.
        if not scores:
            raise ValueError("No valid scores computed from the clusters.")
        
        # Sort the clusters by their computed scores in descending order for priority.
        self.sorted_clusters = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        
        # Find the cluster with the maximum score.
        best_cluster = max(scores, key=scores.get, default=None)
      
        # Retrieve the list of points for the best-scoring cluster, default to empty if not found
        best_cluster_points = self.clusters.get(best_cluster, [])
    
        # Return the best cluster label and its points for further processing
        return best_cluster, best_cluster_points


    """"" Calculate Entropy for Maximum Information Gain. It's like a window apprroach.."""
    def calculate_entropy(self, grid, x, y, radius):
        # Initialize counts for different types of cells within the grid.
        counts = {'free': 0, 'occupied': 0, 'unknown': 0}
        
        total_cells = 0  # Counter for the total number of cells examined within the radius.
        max_y, max_x = grid.shape  # Get the dimensions of the grid.

        # Calculate minimum and maximum indices to examine in the grid based on the specified radius.
        x_min = max(0, x - radius)
        x_max = min(max_x, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(max_y, y + radius + 1)

        # Iterate over each cell within the bounds determined by x_min, x_max, y_min, y_max.
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                cell_value = grid[j][i]  # Retrieve the value of the current cell.
                if cell_value == -1:  # If the cell is marked unknown.
                    counts['unknown'] += 1
                elif cell_value == 0:  # If the cell is marked free space.
                    counts['free'] += 1
                else:  # Otherwise, the cell is occupied space.
                    counts['occupied'] += 1
                total_cells += 1  # Increment the total cell count.

        # After iterating through the cells, calculate probabilities for each cell type.
        if total_cells > 0:
            probabilities = [count / total_cells for count in counts.values()]

            # Calculate and return the entropy.
            return stats.entropy(probabilities)
        else:
            # If no cells were within the radius, return an entropy of 0.
            return 0
        


    """"Compute Maximum Information Gain to select the best viewpoint from the best cluster """
    def maximum_information_gain(self, cluster_points):
        # Compute the point with the highest information gain from a set of cluster points.
        max_entropy = -np.inf  # Initialize maximum entropy as negative infinity.
        best_point = None  # Initialize the best point as None.
        
        # Iterate through each point in the cluster.
        for point in cluster_points:
            # Extract x and y coordinates from the point.
            x = int(point[0])
            y = int(point[1])
            
            # Calculate entropy around the point within a specified radius.
            local_entropy = self.calculate_entropy(self.map, x, y, int(self.distance_threshold / self.resolution))
            
            # Update the best point if the current point's entropy is higher.
            if local_entropy > max_entropy:
                max_entropy = local_entropy
                best_point = point
        
        if best_point is None:
            rospy.logwarn(f"No more valid frontiers left!.. Exploration Completed..")
        return best_point
    
    
    """ Publishing the goal point to the planner."""
    def publish_goal(self, goal_point):
        goal = PoseStamped()
        goal.header.frame_id = "world_ned"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = goal_point[0]
        goal.pose.position.y = goal_point[1]
        self.goal_pub.publish(goal)
    


    """" Navigate to viewpoint. """
    def navigate_to_viewpoint(self, best_view_point, cluster_id, attempt=0, max_attempts=10):
        # Convert the grid point to an actual position.
        mapped_point = self.svc.map_to_position(best_view_point)
        
        # Check if the mapped point is a valid location to navigate to.
        if not self.svc.is_valid_frontier(mapped_point):
            
            rospy.logwarn(f"{attempt+1}: Point {mapped_point} is invalid, selecting next best point.")
            
            """Attempt to select another point from the same cluster if the current one is invalid. """
            if attempt < max_attempts:  # allow exactly ten attempts

                next_grid_point = self.select_next_target_point(cluster_id, best_view_point)
                if next_grid_point is not None:
                    return self.navigate_to_viewpoint(next_grid_point, cluster_id, attempt + 1)
                else:
                    rospy.logdebug("No more points left in the  current cluster.")
                    return 
                
            else:
                rospy.logwarn("Maximum attempts reached within the cluster; Selecting next cluster for a new viewpoint.")
                
                # If all attempts fail or no more points are left, select a new cluster
                next_point, next_cluster = self.select_next_cluster_best_viewpoint(cluster_id)
                if next_point is not None: 
                    return self.navigate_to_viewpoint(next_point, next_cluster)
                else:
                    rospy.logerr("No more clusters to visit.")
                    return 
        else:
            """ If a valid point is found, publish it and delete the cluster."""
            rospy.loginfo("Publishing New Best ViewPoint.")
            self.publish_goal(mapped_point)
            # Delete the cluster.
            del self.clusters[cluster_id]
            print('Remaining clusters', {label: len(cluster) for label, cluster in self.clusters.items()})
            # Publish the best viewpoint.
            self.publish_viewpoint_marker(mapped_point)
            return 
    


    """" Select the next best viewpoint if the point is invalid and find the best viewpoint in the best cluster using Maximum Information Gain."""
    def select_next_target_point(self,cluster_id,best_viewpoint):
        # Get All points from the current 
        all_points=self.clusters.get(cluster_id,[])
        for i, point in enumerate(all_points):
            if np.array_equal(point,best_viewpoint):
                # Remove the current invalid best viepoint from the cluster list.
                del all_points[i]
                break
        # Update the list of points in the cluster.
        self.clusters[cluster_id] = all_points
        if all_points is not None:  # Check if there are still points left to consider.
            # Using Maximum Information Gain from all the points  from the current best cluster to get the best viewpoint.
            next_viewpoint = self.maximum_information_gain(all_points)
            return next_viewpoint
        else:
            return None  


    """" Select the next best cluster using the weight score (Maximum length of the cluster and Minimum Distance) and from there find the best viewpoint in the next best cluster cluster using Maximum Information Gain."""
    def select_next_cluster_best_viewpoint(self, current_cluster_id):
        # Remove the current cluster from sorted_clusters
        self.sorted_clusters = [cluster for cluster in self.sorted_clusters if cluster[0] != current_cluster_id]
        # Check if there are any clusters left to process
        if not self.sorted_clusters:
            rospy.logwarn("All clusters have been visited.")
            return None, None
        # Get the next best cluster from sorted_clusters.
        next_cluster = self.sorted_clusters[0][0]
        # Get all the points from the next best cluster.
        next_points = self.clusters.get(next_cluster, [])
        if next_points:
             # Using Maximum Information Gain from all the points in the next best cluster to get the best viewpoint.
            next_point = self.maximum_information_gain(next_points)
            return next_point, next_cluster
        else:
            rospy.logwarn("No points available in the next cluster.")
            return None, None


if __name__ == '__main__':
    rospy.init_node('frontier_exploration_node', anonymous=True)
    FrontierExploration()
    rospy.spin()
