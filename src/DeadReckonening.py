#!/usr/bin/env python
import math
import rospy
import tf
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, Imu
from visualization_msgs.msg import Marker, MarkerArray
import tf.transformations
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from math import *
import threading
# Defines a ROS node for dead reckoning based on wheel encoders and IMU data
class DeadReckoningNode:
    def __init__(self):
        rospy.init_node('dead_reckoning_node', anonymous=True)  # Initialize ROS node
        
        # Initialize wheel velocities and flags to check if velocities are received
        self.left_wheel_velocity = 0.0
        self.right_wheel_velocity = 0.0
        self.left_wheel_velocity_received = False
        self.right_wheel_velocity_received = False
        # self.lock = threading.Lock()
       
      

        # TF broadcaster to publish transformations between coordinate frames
        self.odom_broadcaster = tf.TransformBroadcaster()
        
        # Robot parameters: wheel radius and distance between wheels
        self.wheel_radius = 0.035
        self.wheel_base_distance = 0.235
        #Robot Dimension
        self.xB_dim=3
        #Feature Dimension
        self.xF_dim=2
        # Initial robot state and its covariance matrix
        self.xk= np.zeros((self.xB_dim,1))
     
       
        
        self.Pk = np.eye(self.xB_dim) * 0.1
        
        # Odometry noise covariance matrix
        self.Qk = np.diag(np.array([0.2 ** 2,0.2**2, 0.02 ** 2])) 
        
        # Publisher for odometry messages
        self.odom_pub = rospy.Publisher("/turtlebot/kobuki/odom", Odometry, queue_size=10)
        
        
        self.last_time = rospy.Time.now()  # Tracks the last time a message was received
       
        # Subscribe to joint states and IMU data topics
        self.imu_sub=rospy.Subscriber('/turtlebot/kobuki/sensors/imu_data',Imu, self.imu_callback,queue_size=10)
        self.js_sub=rospy.Subscriber("/turtlebot/joint_states", JointState, self.joint_states_callback, queue_size=10)
        
        
    def wrap_angle(self,angle):
        
        "Normalizes an angle to be within the range of [-pi, pi]"
        #corrects an angle to be within the range of [-pi, pi]
        return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )
    def imu_callback(self, msg):
        """
        Callback function to handle IMU sensor data. It processes the orientation data provided in quaternion format,
        converts it to Euler angles, and updates the robot state using the yaw (orientation around the vertical axis).

        :param msg: The message received from the IMU topic, containing orientation data in quaternion format.
        """
       
        # Extract the quaternion tuple from the IMU message, which includes x, y, z, and w components.
        quaternion = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)

        # Convert the quaternion to Euler angles. Since only yaw is needed for 2D motion estimation, 
        # the roll and pitch values are discarded (denoted by underscores).
        _, _, yaw_measurement = euler_from_quaternion(quaternion)
        yaw_measurement= -yaw_measurement
        self.x, self.P=self.update(yaw_measurement)
        

    def joint_states_callback(self, msg):
        """
        Callback function that processes messages from the joint states topic.
        
        :param msg: The message received from the joint states topic, containing the names of joints and their velocities.
        """
        print('e')
       
        # Define the names for the left and right wheel joints for easier identification.
        # self.left_wheel_name = 'turtlebot/kobuki/wheel_left_joint'
        # self.right_wheel_name = 'turtlebot/kobuki/wheel_right_joint'
        # print(self.right_wheel_name)
        # Check the first joint name in the message and assign the corresponding velocity.
      
        self.left_wheel_velocity = msg.velocity[0]
        self.right_wheel_velocity = msg.velocity[1]
    #     print(msg.velocity[0])
        # self.left_wheel_velocity_received = True  # Mark the left wheel velocity as received.
        
    # elif msg.name[0] == self.right_wheel_name:
    #     self.right_wheel_velocity = msg.velocity[1]
        # self.right_wheel_velocity_received = True  # Mark the right wheel velocity as received.
    
    # Proceed only if both wheel velocities have been received.
    # if self.left_wheel_velocity_received and self.right_wheel_velocity_received:
    #     print('i')
        # Compute the linear velocity for each wheel by multiplying the angular velocity by the wheel radius.
    
        self.left_linear_velocity = self.left_wheel_velocity * self.wheel_radius
        self.right_linear_velocity = self.right_wheel_velocity * self.wheel_radius
        
        # Calculate the overall linear and angular velocities.
        self.linear_velocity = (self.left_linear_velocity + self.right_linear_velocity) / 2
        print(self.linear_velocity)
        self.angular_velocity = (self.left_linear_velocity - self.right_linear_velocity) / self.wheel_base_distance
        self.time_stamp=msg.header.stamp
        # Compute the current time from the message stamp and calculate the time elapsed since the last update.
        self.current_time = rospy.Time.from_sec(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
        self.time = (self.current_time - self.last_time).to_sec()
        self.last_time = self.current_time
        
        # Update the robot's pose using the motion model.
        self.xk[0,0] = self.xk[0,0]+np.cos(self.xk[2,0]) * self.linear_velocity * self.time
        
        self.xk[1,0] =self.xk[1,0]+ np.sin(self.xk[2,0]) * self.linear_velocity * self.time
        
        # Normalize the robot's orientation angle.
        self.xk[2,0] = self.wrap_angle(self.xk[2,0] + self.angular_velocity * self.time)
        
        # operation modifying self.xk
        self.xk, self.Pk = self.prediction(self.xk, self.Pk, self.linear_velocity, self.angular_velocity, self.time)
        print(self.xk)
            
            
        
        # Predict the next state using the motion model.
        
        # Publish the updated odometry and the visual markers for the robot.
        self.publish_odometry()
        
        
        # # Reset the velocity reception flags for the next iteration.
        # self.left_wheel_velocity_received = False
        # self.right_wheel_velocity_received = False
    def prediction(self, xk, Pk, v, w, t):
        """
        Predicts the next state of the robot using the motion model.

        :param xk: Current state vector of the robot.
        :param Pk: Current state covariance matrix.
        :param v: Linear velocity of the robot.
        :param w: Angular velocity of the robot.
        :param t: Time interval since the last update.
        :return: Updated state vector and covariance matrix after prediction.
        """
        
        
        # Extract the base state from the state vector.
        xk_robot = self.xk[:self.xB_dim]

        # Calculate the Jacobian of the motion model with respect to the robot's state.
        Ak = np.array([
            [1.0, 0.0, -np.sin(self.xk[2,0]) * v * t],
            [0.0, 1.0,  np.cos(self.xk[2,0]) * v * t],
            [0.0, 0.0, 1.0]
        ])

        # Calculate the Jacobian of the motion model with respect to the process noise.
        # Wk = np.array([
        #     [np.cos(self.xk[2,0]), -np.sin(self.xk[2,0]), 0.0],
        #     [np.sin(self.xk[2,0]),  np.cos(self.xk[2,0]), 0.0],
        #     [0.0, 0.0, 1.0]
        # ])
        Wk = np.array([[np.cos(self.xk[2,0])*t*0.5*self.wheel_radius, np.cos(self.xk[2,0])*t*0.5*self.wheel_radius,0.0],    
                            [np.sin(self.xk[2,0])*t*0.5*self.wheel_radius, np.sin(self.xk[2,0])*t*0.5*self.wheel_radius,0.0],   
                                
                            [(t*self.wheel_radius)/self.wheel_base_distance, -(t*self.wheel_radius)/self.wheel_base_distance,1.0]])     
        

        # Retrieve the process noise covariance matrix.
        Qk = self.Qk
        
        # Check if there are additional state components beyond the robot's base state.
        if len(self.xk) > self.xB_dim:
            # If additional states exist, extract and concatenate them to the robot's base state.
            xk_extend = self.xk[self.xB_dim:]
            self.xk = np.concatenate((xk_robot, xk_extend))
        
        else:
            # If no additional states, use the base state directly.
            self.xk = xk_robot
            
        
        # Prepare the extended Jacobians and covariance matrices to include additional states.
        # Fk1 = np.eye(len(self.xk))
        # Fk2 = np.zeros((len(self.xk), len(Qk)))

        # # Assign the Jacobian blocks to the extended matrices.
        # Fk1[:self.xB_dim, :self.xB_dim] = Ak
        # Fk2[:self.xB_dim, :] = Wk
        
        
    
        # print('before Pk', self.Pk.shape)
        # print("Fk1 shape:", Fk1.shape)
        # print("Fk2 shape:", Fk2.shape)
        # print("Pk shape:", self.Pk.shape)
        # print("Qk shape:", Qk.shape)

        # Update the state covariance matrix.
        
        self.Pk = Ak @ self.Pk @ Ak.T + Wk @ Qk @ Wk.T
        
        
     
     
        
        # print('Prediction mean',self.xk.shape)
        # print('prediction covariance',self.Pk.shape)
        
        
         
        return self.xk, self.Pk
    
    def update(self, yaw_measurement):
        """
        Performs a Kalman filter update step using the yaw measurement from the IMU. This method integrates the new yaw
        measurement into the state estimate, updating both the state vector and its covariance.

        :param yaw_measurement: The yaw angle measured by the IMU.
        """
        
   
       
        # Actual measurement received from the IMU callback.
        self.zk = np.array([yaw_measurement]).reshape(-1, 1)
    
        # Jacobian of the observation model with respect to the noise vector.
        self.Vk = np.eye(1)  # It's an identity matrix because the observation noise directly affects the measured yaw.

        # Covariance matrix of the observation noise.
        self.Rk = np.array([[0.01]])

        # Expected observation from the current state estimate.
        self.h = np.array([self.xk[2,0]]).reshape(-1, 1)# The expected yaw from the state.
    

        # Innovation: difference between the actual measurement and the exp
        # ected observation.
        innovation = self.wrap_angle(self.zk - self.h)
        

        # Compute the Jacobian of the observation model with respect to the state vector.
        # Jacobian of the observation model (Hk) with respect to the state
        self.Hk = np.zeros((1, len(self.xk)))
        self.Hk[0, 2] = 1  # Only the yaw component, which affects the measurement

            
        # Compute the Kalman gain.
        S = self.Hk @ self.Pk @ self.Hk.T + self.Vk @ self.Rk @ self.Vk.T
        Kk = self.Pk @ self.Hk.T @ np.linalg.inv(S)
        

        # Update the state estimate using the Kalman gain and the innovation.
        self.xk= self.xk + Kk @ innovation

        # Update the covariance of the estimate.
        I = np.eye(len(self.xk))  # Identity matrix of the same dimension as the state vector.
        self.Pk= (I - Kk @ self.Hk) @ self.Pk 

       

    

        return self.xk, self.Pk
       
    def publish_odometry(self):
        """
        Publishes odometry data to ROS.

        :param xk: Current state vector [x, y, theta].
        :param Pk: Covariance matrix of the state.
        :param linear_velocity: Linear velocity of the robot.
        :param angular_velocity: Angular velocity of the robot.
        :param current_time: Current ROS time, used as the timestamp for the odometry message.
        """
        # Convert the yaw angle to a quaternion for representing 3D orientations.
        self.q = quaternion_from_euler(0, 0, self.xk[2,0])

        # Initialize an odometry message.
        odom = Odometry()
        odom.header.stamp = self.time_stamp

        odom.header.frame_id = "world_ned"
        odom.child_frame_id = "/turtlebot/kobuki/base_footprint"

       
        # Set the position in the odometry message.
        odom.pose.pose.position.x = self.xk[0,0]
        odom.pose.pose.position.y = self.xk[1,0]
       
        # Set the orientation in the odometry message.
        odom.pose.pose.orientation.x = self.q[0]
        odom.pose.pose.orientation.y = self.q[1]
        odom.pose.pose.orientation.z = self.q[2]
        odom.pose.pose.orientation.w = self.q[3]

        # Set the velocities in the odometry message.
        odom.twist.twist.linear.x = self.linear_velocity
        odom.twist.twist.angular.z = self.angular_velocity

        # Setup the covariance matrix in the odometry message.
        odom.pose.covariance = [self.Pk[0,0], self.Pk[0,1], 0.0, 0.0, 0.0, self.Pk[0,2],    
                                        self.Pk[1,0], self.Pk[1,1], 0.0, 0.0, 0.0, self.Pk[1,2],   
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,     
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                        self.Pk[2,0], self.Pk[2,1], 0.0, 0.0, 0.0, self.Pk[2,2]]             

        # print('Odometry Values:',odom)
        # Publish the odometry message.
        self.odom_pub.publish(odom)
        

        # Publish the transform over tf (transformation frames in ROS).
        self.odom_broadcaster.sendTransform((self.xk[0,0], self.xk[1,0], 0.0), self.q, self.time_stamp, odom.child_frame_id, odom.header.frame_id)

if __name__ == '__main__':
    try:
        # Call the DeadReckoning function 
        DeadReckoningNode()
        # Keep the program running until rospy is shut down
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
