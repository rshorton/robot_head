# Robot Head

This node controls the robot head which includes this functionality:
1. Vision input -  configures Depthai Camera processing pipeline and receives detection output from the camera
2. Camera tracker - controls the camera (head) tracking servos to keep a person in view.
3. Face control - controls the emotional output features (head tilt, smile, talking indicator)
4. Viewer - displays camera view

The tracking and face control functionality are implemented by the tracker node since they both use the I2C controlled PWM output board.

# Running

Set the venv for the Depthai python dependencies before running.

ros2 run robot_head vision
ros2 run robot_head tracker
ros2 run robot_head viewer