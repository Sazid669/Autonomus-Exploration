# Hands on Planning
In this project, we are exploring close environment to discover the area and create a map. For explanation, we are using frontier algorithm to create a best view points and using RRTstar Informed with dubins path to create the path of the robot to visit those areas. Using the pure persuit controller, the path is followed. 

## Group Members:
1. Mir Mohibullah Sazid
2. Syma Afsha

## How to run the project
To run the project, please install the following libaries. 

1. For the planner and state validity check the following libraries are required
``` $ pip install git+git://github.com/AndrewWalker/pydubins.git ```
``` pip install scipy ``` or ```sudo apt-get install python3-scipy```

2. For the exploration the following libraries are required
```pip install scikit-learn```
```pip install scipy```

3. To run the project, compile the project in catkin workspace and then run the following commands:
```roslaunch hands_on_planning stonefish.launch```

After this, the project will be start running in the computer. please add all the visulization markes in rviz see the cluster, frontier, path and tree of RRT's.

## Files Description
* `StateValidityChecker.py`: It contains all the validation checking, enlarging the obstacles, finding valid goal points.
* `RRTInformed.py`: It contains RRTstar Informed path palnner algorithm.
* `RRTstar.py`: It contains RRTStar path palnner algorithm
*  `dubins.py`: It contains all the necessary steps to add dubins path to the planner. You can change the path planner algorithm here in `compute_path()` function.
* `controller.py`: It contains all the pure pursuits controller and the low level controller that can be used to follow path without dubins path. 
* ``plannerNode.py``: Main node for following the planner and the low level controller with all the visualization markers for planner. 
* ``frontier.py``: Main node for frontier algorithm with all the visualization markers in rviz. 
* `dead_reckoning.py`: From the ground truth of robot base to world_ned
* `DeadReckoning.py`: From the joint state of robot to calculate the odometry with imu update which is used in real robot.

All the rostopic is seted for stonefish simulator. 

Please go to this for video `https://youtu.be/pkE6xhLc7Oo?si=DxJTnHIL2tj6G9-n`

### Thank you!






