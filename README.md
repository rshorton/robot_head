# Robot Head

This node controls the robot head which includes this functionality:
1. Depthai Camera - configures processing pipeline and receives detection output from the camera
2. Camera tracker - controls the camera (head) tracking servos to keep a person in view.
3. Face control - controls the emotional output features (head tilt, smile, talking indicator)

This node is based on the original Deptha AI ROS2 sample.

# Running

Set the venv for the Depthai python dependencies before running.

ros2 run robot_head robot_head