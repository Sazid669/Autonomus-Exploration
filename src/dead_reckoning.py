#!/usr/bin/python3

import rospy
import tf
import numpy as np
from sensor_msgs.msg import JointState
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from nav_msgs.msg import Odometry


class DifferentialDrive(object):
    """ This class implements a differential drive robot model  and publishes the odometry of the robot as a transformation and an odometry message """ 

    def __init__(self):
        # variable declaration for the pose and velocity of the robot 
        self.x = 0
        self.y = 0
        self.theta = 0

        # declare a transformation broadcaster and an odometry to publish the pose of the robot as a transformation and an odometry message respectively
        self.robot_pose = tf.TransformBroadcaster()
        self.kobuki_pose = Odometry()

        # define a publisher to publish the odometry message
        self.pub = rospy.Publisher("/turtlebot/kobuki/odom", Odometry, queue_size=10)

        # declare a subscriber to joint_states topic and a publisher to the odometry topic
        self.sub = rospy.Subscriber("/kobuki/ground_truth", Odometry, self.callback)

        # set the rate of the node: 10Hz
        self.rate = rospy.Rate(10)
        rospy.spin()
        pass

    def callback(self, msg):
        """this function is called when a message is received from the topic
            and it updates the global variables with the latest encoder readings

        Args:
            msg (Odometry): the message received from the /kobuki/sensors/virtual_odom_sensor topic 
        """


        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        _, _, self.theta = tf.transformations.euler_from_quaternion([msg.pose.pose.orientation.x, 
                                                                     msg.pose.pose.orientation.y,
                                                                     msg.pose.pose.orientation.z,
                                                                     msg.pose.pose.orientation.w])
        self.P = msg.pose.covariance
        self.linear_velocity = msg.twist.twist.linear.x
        self.angular_velocity = msg.twist.twist.angular.z 

        self.robot_pose.sendTransform((self.x, self.y, 0), quaternion_from_euler(0, 0, self.theta), rospy.Time.now(), "/kobuki/base_footprint", "world_ned")
        # publish the robot's pose as an odometry message
        self.publish_odom_msg()


    def publish_odom_msg(self):
        """this function publishes the robot's pose as an odometry message
        """

        # create an odometry message
        self.kobuki_pose.header.stamp = rospy.Time.now()
        self.kobuki_pose.header.frame_id = "world_ned"
        self.kobuki_pose.child_frame_id = "/kobuki/base_footprint"
        self.kobuki_pose.pose.pose.position.x = self.x
        self.kobuki_pose.pose.pose.position.y = self.y
        self.kobuki_pose.pose.pose.position.z = 0

        #convert the yaw angle to a quaternion
        orientation = quaternion_from_euler(0, 0, self.theta)
        self.kobuki_pose.pose.pose.orientation.x = orientation[0]
        self.kobuki_pose.pose.pose.orientation.y = orientation[1]
        self.kobuki_pose.pose.pose.orientation.z = orientation[2]
        self.kobuki_pose.pose.pose.orientation.w = orientation[3]

        self.kobuki_pose.twist.twist.linear.x = self.linear_velocity
        self.kobuki_pose.twist.twist.linear.y = 0
        self.kobuki_pose.twist.twist.linear.z = 0
        self.kobuki_pose.twist.twist.angular.x = 0
        self.kobuki_pose.twist.twist.angular.y = 0
        self.kobuki_pose.twist.twist.angular.z = self.angular_velocity




        #initialize the covariance matrix
        self.kobuki_pose.pose.covariance = self.P
        # self.kobuki_pose.pose.covariance = [self.P[0,0], self.P[0,1], 0, 0, 0, self.P[0,2], 
        #                                     self.P[1,0], self.P[1,1], 0, 0, 0, self.P[1,2], 
        #                                     0, 0, 0, 0, 0, 0, 
        #                                     0, 0, 0, 0, 0, 0, 
        #                                     0, 0, 0, 0, 0, 0, 
        #                                     self.P[2,0], self.P[2,1], 0, 0, 0, self.P[2,2]]

        #publish the robot's pose as odometry messages
        self.pub.publish(self.kobuki_pose) 
        pass


if __name__ == "__main__":
    try:
        # initialize the node
        rospy.init_node("differential_drive")

        # create an instance of the class
        kobuki = DifferentialDrive()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass