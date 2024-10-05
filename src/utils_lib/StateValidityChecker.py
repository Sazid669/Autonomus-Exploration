import numpy as np
import scipy.ndimage
import math
from math import sqrt
def wrap_angle(angle):
    #corrects an angle to be within the range of [-pi, pi]
    return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )
class StateValidityChecker:
    def __init__(self, distance=0.1, is_unknown_valid=True, max_recursion=50): 
        self.map = None                  
        self.resolution = None
        self.origin = None                            
        self.there_is_map = False
        self.distance = distance                    
        self.is_unknown_valid = is_unknown_valid  
        self.max_recursion = max_recursion 
        inflation_radius=0.1
        self.inflation_radius = inflation_radius 
        self.inflated_map = None
         
    
 
    def set(self, data, resolution, origin): 
        self.map = data
        self.resolution = resolution
        self.origin = np.array(origin)
        self.there_is_map = True
        self.height = data.shape[0]
        self.width = data.shape[1]  
        self.inflated_map = self.inflate_obstacles(data)

    def inflate_obstacles(self, map_data):
        """Inflates obstacles in the map based on the inflation radius."""
        # Calculate the number of cells to inflate around each obstacle
        inflation_cells = int(self.inflation_radius / self.resolution)

        # Create a binary map where obstacles are 1 and free spaces are 0
        binary_map = np.where(map_data >= 100, 1, 0)

        # Inflate obstacles using a binary dilation
        inflated_binary_map = scipy.ndimage.binary_dilation(binary_map, iterations=inflation_cells).astype(np.uint8)

        # Create the inflated map where inflated obstacles are set to 100
        inflated_map = np.where(inflated_binary_map == 1, 100, map_data)

        return inflated_map  
    def is_valid(self, position):
        # Check if occupancy map is set
        if not self.there_is_map:
            raise ValueError("Occupancy map not set.")
        
        # Convert position to grid coordinates
        grid_coord = self.__position_to_map__(position)
        
        # If position is not in the map, return False
        if grid_coord is None:
            return False
        
        # Calculate the number of grid cells to check around the position based on the specified distance
        new_distance = int(self.distance / self.resolution)
        
        # Get the bounds for the grid cells to check
        min_x = max(grid_coord[0] - new_distance, 0)
        max_x = min(grid_coord[0] + new_distance, self.map.shape[0] - 1)
        min_y = max(grid_coord[1] - new_distance, 0)
        max_y = min(grid_coord[1] + new_distance, self.map.shape[1] - 1)
        
        # Iterate over nearby grid cells
        for grid_i in range(min_x, max_x + 1):
            for grid_j in range(min_y, max_y + 1):
                # Check if grid cell is an obstacle in the inflated map
                if self.inflated_map[grid_i, grid_j] >= 100:
                    return False
        
        return True
    def find_valid_goal_cell(self, position, recursion_count=0):
        # Check if recursion limit reached
        if recursion_count >= self.max_recursion:
            return None
        
        # Convert position to grid coordinates
        grid_coord = self.__position_to_map__(position)
        # If position is not in the map, return None
        if np.any(grid_coord == None):
            return None
        
        # Calculate the number of grid cells to check around the position based on the specified distance
        new_distance = int(self.distance / self.resolution)
        
        # Get grid coordinates of the edge
        grid_edge_x = grid_coord[0]
        grid_edge_y = grid_coord[1]
        
        # Iterate over nearby grid cells
        for grid_i in range(grid_edge_x - new_distance, grid_edge_x + new_distance + 1):
            for grid_j in range(grid_edge_y - new_distance, grid_edge_y + new_distance + 1):
                # Check if grid cell is within map boundaries
                if 0 <= grid_i < self.height and 0 <= grid_j < self.width:
                    # Check if the current cell is valid
                    if self.is_valid(self.map_to_position([grid_i, grid_j])):
                        return self.map_to_position([grid_i, grid_j])  # Return the valid cell
        # If no valid cell found in the vicinity, recursively call find_valid_cell with a larger distance
        return self.find_valid_goal_cell(position, recursion_count + 1)
    
    def check_path(self, path, step_size=0.04): 
        new_path = []
        for index_i in range(len(path)-1):
            waypoint1 = path[index_i][0:2] 
            waypoint2 = path[index_i+1][0:2]
            # Calculate distance between two waypoints
            dist = np.linalg.norm(np.array(waypoint2) - np.array(waypoint1))
            # Calculate number of steps based on distance and step_size
            num_steps = max(int(dist / step_size), 1)
            for index_j in range(num_steps+1):
                s = index_j / num_steps
                # Calculate new point using interpolation
                new_point = (1-s)*np.array(waypoint1) + s*np.array(waypoint2)
                new_path.append(list(new_point))
        
        # Check if each point in the new path is valid
        for point in new_path:
            if not self.is_valid(point):
                return False
        return True
    

    def set_new_goal(self, path, index=-1):
        step_size = 0.04

        # Ensure the index is within valid range
        if index == -1:
            index = len(path) - 1

        # Extracting only the x, y components, ignoring theta
        point1, point2 = path[index][:2], path[index - 1][:2]

        # Calculate the Euclidean distance between the points
        distance = np.linalg.norm(np.array(point2) - np.array(point1))
        num_steps = int(distance / step_size)

        for step in range(num_steps):
            interpolation = step / num_steps
            x = point1[0] * (1 - interpolation) + point2[0] * interpolation
            y = point1[1] * (1 - interpolation) + point2[1] * interpolation

            # Check validity based only on x and y
            if self.is_valid([x, y]):
                return np.array([x, y])

        # Recursive call to continue searching in the path if the endpoint is not valid
        if index > 0:
            return self.set_new_goal(path, index - 1)
        else:
            return None

  

    def is_valid_frontier(self, position):
        # return True
        
        # Check if occupancy map is set
        if not self.there_is_map:
            raise ValueError("Occupancy map not set.")
        # Convert position to grid coordinates
        grid_coord = self.__position_to_map__(position)
        # If position is not in the map, return False
        if np.any(grid_coord == None):
            return False
        # Calculate the number of grid cells to check around the position based on the specified distance
        new_distance = int(self.distance/self.resolution)
        # Initialize validity flag
        is_position_valid = True
                    
        # Get grid coordinates of the edge
        grid_edge_x = grid_coord[0]
        grid_edge_y = grid_coord[1]
        # Iterate over nearby grid cells
        for grid_i in range (grid_edge_x - new_distance, grid_edge_x + new_distance  ):
            for grid_j in range (grid_edge_y - new_distance, grid_edge_y + new_distance ):         
                # Check if grid cell is within map boundaries
                if grid_i < 0 or grid_j < 0 or grid_i >= (self.map.shape[0]) or grid_j >= (self.map.shape[1]) :
                    return False
                else:
                    # Check if the occupancy value of the grid cell is valid
                    if self.map[grid_i, grid_j] >= -1:
                        is_position_valid = True
                    # Check if the occupancy value of the grid cell indicates an obstacle
                    if self.map[grid_i, grid_j] >= 50:
                        return False
                            
        return is_position_valid
        
    
    # Transform position with respect the map origin to cell coordinates
    def __position_to_map__(self, p):
        # TODO: convert world position to map coordinates. If position outside map return `[]` or `None`
        relative_pos = (np.array([p[0], p[1]]) - self.origin) / self.resolution
        # Round the cell position to get integer indices
        map_pos = np.round(relative_pos).astype(int)
        if np.any(map_pos < 0) or np.any(map_pos >= np.shape(self.map)):
            # Position is outside the map
            return None
        else:
            # Position is within the map
            return map_pos 
 
    def map_to_position(self,m):

        x = self.origin[0] + m[0] * self.resolution + self.resolution/2
        y = self.origin[1]+m[1] * self.resolution + self.resolution/2
        mtp=[x,y]
        return mtp
    
    





