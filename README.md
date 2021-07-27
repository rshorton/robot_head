# robot_head

This ROS 2 package includes nodes which control the ElsaBot robot head with this functionality:
- Vision node -  configures OAK-D Camera processing pipeline and receives detection output from the camera
- Tracker node - controls the camera/head tracking servos to keep a person in view.  Also controls the emotional output features (head tilt, smile, talking indicator).
- Viewer node - displays the camera output with overlaid annotations

## Running

Set the venv for the Depthai python dependencies before running. Example:

````
pushd ~/depthai-python/; . venv/bin/activate; popd

    (Revise the above command to use your install of DepthAi.)

ros2 launch robot_head robot_head.launch.py
````

Or, use the launch file of the **elsabot_bt** package to launch the nodes of this package.
