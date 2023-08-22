# robot_head

This ROS 2 package includes nodes which control the ElsaBot robot head with this functionality:
- Vision node -  configures OAK-D Camera processing pipeline and receives detection output from the camera
- Tracker node - controls the camera/head tracking servos to keep a person in view.  Also controls the emotional output features (head tilt, smile, talking indicator).
- Viewer node - displays the camera output with overlaid annotations
- Face node - controls the smile and antenna LEDs

## Install

### web_video_server

The web video server is used to provide a compressed video stream that you can play via a web browser.  Example url:

     http://<robot ip address>:8095/stream?topic=/color/image

Clone and build 'ros2' branch:
  
    git clone --branch ros2 git@github.com:rshorton/web_video_server
    colcon build

## Running

Set the venv for the Depthai python dependencies before running. Example:

    pushd ~/depthai-python/; . venv/bin/activate; popd
    (Revise the above command to use your install of DepthAi.)

    ros2 launch robot_head robot_head.launch.py

Or, use the launch file of the **elsabot_bt** package to launch the nodes of this package.
