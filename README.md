# robot_head

This ROS 2 package includes nodes which control the ElsaBot robot head with this functionality:
- Vision input -  configures OAK-D Camera processing pipeline and receives detection output from the camera
- Camera tracker - controls the camera/head tracking servos to keep a person in view.
- Face control - controls the emotional output features (head tilt, smile, talking indicator)
- Viewer - displays the camera output with overlaid annotations

The tracking and face control functionality are implemented by the tracker node since they both use the I2C-controlled PWM output board.

## Running

Use the launch file of the **elsabot_bt** package to launch all the nodes of this package, or run the launch script of this package to only run the nodes of this package.

Set the venv for the Depthai python dependencies before running. Example:

pushd ~/depthai/depthai-python/; . venv/bin/activate; popd

Revise the above command to use your install of DepthAi.

