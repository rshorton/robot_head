# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
import sys
import cv2
import json
import math

import msgpack
import msgpack_numpy as m
import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from rclpy.node import Node

# Used for publishing the camera joint positions
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile

class CameraViewer(Node):
    meta = []

    def __init__(self):
        super().__init__('camera_viewer')

        self.imageSub = self.create_subscription(
            Image,
            '/color/image',
            self.image_callback,
            10)
        self.bridge = CvBridge()

        self.setWinPos = True

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv_image = cv2.resize(cv_image, (int(640*1.4), int(360*1.4)), interpolation = cv2.INTER_AREA)
        cv2.imshow("image", cv_image)

        # Set initial window pos
        if self.setWinPos:
            self.setWinPos = False
            cv2.moveWindow("image", 78, 25)

        key = cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    viewer = CameraViewer()

    rclpy.spin(viewer)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    viewer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
