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

import threading
import sys
import time
import pathlib

import numpy as np
import cv2

import depthai as dai

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from pose import getKeypoints, getValidPairs, getPersonwiseKeypoints
from tracker import CameraTracker
from pose_interp import analyze_pose

# Custom object detection messages
from object_detection_msgs.msg import ObjectDescArray
from object_detection_msgs.msg import ObjectDesc

from human_pose_interfaces.msg import DetectedPose, EnablePoseDetection

human_pose = True
human_pose_process = True

running = True
pose = None

keypoints_list = None
detected_keypoints = None
personwiseKeypoints = None
new_pose = False

colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 0],
          [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 0, 255], [255, 0, 0],
          [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]
POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]

labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def get_model_path(model_name):
    return str(pathlib.Path(__file__).parent.absolute()) + '/models/' + model_name

def create_pipeline():
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Define a source - color camera
    colorCam = pipeline.createColorCamera()
    spatialDetectionNetwork = pipeline.createMobileNetSpatialDetectionNetwork()
    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()

    xoutRgb = pipeline.createXLinkOut()
    xoutNN = pipeline.createXLinkOut()
    xoutBoundingBoxDepthMapping = pipeline.createXLinkOut()
    xoutDepth = pipeline.createXLinkOut()

    xoutRgb.setStreamName("rgb")
    xoutNN.setStreamName("detections")
    xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
    xoutDepth.setStreamName("depth")


    colorCam.setPreviewSize(452, 256)
    colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    colorCam.setInterleaved(False)
    colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    colorCam.setFps(12.0);

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # Setting node configs
    stereo.setOutputDepth(True)
    stereo.setConfidenceThreshold(255)

    spatialDetectionNetwork.setBlobPath(get_model_path('mobilenet-ssd_openvino_2021.2_6shave.blob'))
    spatialDetectionNetwork.setConfidenceThreshold(0.5)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    manip2 = pipeline.createImageManip()
    manip2.initialConfig.setResize(300, 300)
    manip2.setKeepAspectRatio(False)
    colorCam.preview.link(manip2.inputImage)
    manip2.out.link(spatialDetectionNetwork.input)


    if human_pose:
        # NeuralNetwork - human pose
        print("Creating Human Pose Estimation Neural Network...")
        pose_nn = pipeline.createNeuralNetwork()
        pose_nn.setBlobPath(get_model_path('human-pose-estimation-0001_openvino_2021.2_6shave.blob'))

        # Increase threads for detection
        # This slowed overall thruput
#        pose_nn.setNumInferenceThreads(2)
        pose_nn.setNumInferenceThreads(1)
        # Specify that network takes latest arriving frame in non-blocking manner
        pose_nn.input.setQueueSize(1)
        pose_nn.input.setBlocking(False)
        pose_nn_xout = pipeline.createXLinkOut()
        pose_nn_xout.setStreamName("pose_nn")
        pose_nn.out.link(pose_nn_xout.input)

        manip = pipeline.createImageManip()
        manip.initialConfig.setResize(456, 256)
        manip.setKeepAspectRatio(False)
        colorCam.preview.link(manip.inputImage)
        manip.out.link(pose_nn.input)

    # Create outputs
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

#    if syncNN:
#    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
#    else:
    colorCam.preview.link(xoutRgb.input)

    spatialDetectionNetwork.out.link(xoutNN.input)
    spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

    stereo.depth.link(spatialDetectionNetwork.inputDepth)
    spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

    print("Pipeline created.")
    return pipeline

def clear_pose_detection():
    global keypoints_list, detected_keypoints, personwiseKeypoints, new_pose
    print('Clearing pose detections')
    keypoints_list = None
    detected_keypoints = None
    personwiseKeypoints = None
    new_pose = True

class RobotHead(Node):
    def __init__(self):
        super().__init__('robot_head')

        self.image = None
        self.imagePub = self.create_publisher(Image, '/color/image', 10)
        self.bridge = CvBridge()

        # Publisher for list of detected objects
        self.objectPublisher = self.create_publisher(
            ObjectDescArray,
            '/head/detected_objects',
            10)

        # Publisher for pose
        self.posePublisher = self.create_publisher(
            DetectedPose,
            '/head/detected_pose',
             1)

        self.subPoseDetectEnable = self.create_subscription(
            EnablePoseDetection,
            '/head/enable_pose_detect',
            self.pose_detect_enable_callback,
            1)

        self.tracker = CameraTracker(self)

        self.pose_cnt = 0
        self.h = 256
        self.w = 456

        self.setWinPos = True;
        show_depth = False

        pose_last = None

        with dai.Device(create_pipeline()) as device:
            print("Starting pipeline...")
            device.startPipeline()

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

            if human_pose:
                pose_nn = device.getOutputQueue("pose_nn", 1, False)
                t = threading.Thread(target=self.pose_thread, args=(pose_nn, ))
                t.start()

            frame = None
            detections = []

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (64, 255, 64)

            while True:

                rclpy.spin_once(self);

                inPreview = previewQueue.get()
                inNN = detectionNNQueue.get()
                depth = depthQueue.get()

                counter+=1
                current_time = time.monotonic()
                if (current_time - startTime) > 1 :
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                frame = inPreview.getCvFrame()

                self.h, self.w = frame.shape[:2]  # 256, 456

                detections = inNN.detections

                if show_depth:
                    depthFrame = depth.getFrame()

                    depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                    depthFrameColor = cv2.equalizeHist(depthFrameColor)
                    depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
                    if len(detections) != 0:
                        boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
                        roiDatas = boundingBoxMapping.getConfigData()

                        for roiData in roiDatas:
                            roi = roiData.roi
                            roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                            topLeft = roi.topLeft()
                            bottomRight = roi.bottomRight()
                            xmin = int(topLeft.x)
                            ymin = int(topLeft.y)
                            xmax = int(bottomRight.x)
                            ymax = int(bottomRight.y)

                            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

                # If the frame is available, draw bounding boxes on it and show the frame
                height = frame.shape[0]
                width  = frame.shape[1]

                # Publish and update tracker
                self.process_detections(detections, width, height)

                global new_pose
                if new_pose == True:
                    new_pose = False
                    pose = analyze_pose(detected_keypoints, keypoints_list, personwiseKeypoints)
                    #print("pose: %s" % str(pose))
                    msg = DetectedPose()
                    msg.detected = pose["detected"]
                    msg.left = pose['left']
                    msg.right = pose['right']
                    self.posePublisher.publish(msg)
                    pose_last = pose;

                # Display Human pose
                try:
                    if keypoints_list is not None and detected_keypoints is not None and personwiseKeypoints is not None:
                        for i in range(18):
                            for j in range(len(detected_keypoints[i])):
                                cv2.circle(frame, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
                        for i in range(17):
                            for n in range(len(personwiseKeypoints)):
                                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                                if -1 in index:
                                    continue
                                B = np.int32(keypoints_list[index.astype(int), 0])
                                A = np.int32(keypoints_list[index.astype(int), 1])
                                cv2.line(frame, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
                except:
                    print("keypoint out of bound")

                flipped = cv2.flip(frame, 1)

                # Display detections
                for detection in detections:
                    try:
                        label = labelMap[detection.label]
                    except:
                        label = detection.label
                    if label != 'person':
                        continue

                    # Denormalize bounding box
                    x1 = int(detection.xmin * width)
                    x2 = int(detection.xmax * width)
                    y1 = int(detection.ymin * height)
                    y2 = int(detection.ymax * height)
                    font_scale = 0.4
                    cv2.putText(flipped, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color)
                    cv2.putText(flipped, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color)
                    cv2.putText(flipped, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color)
                    cv2.putText(flipped, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color)
                    cv2.putText(flipped, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color)

                    # Fix - handle multiple persons
                    if pose_last is not None and label == 'person':
                        cv2.putText(flipped, f"PoseL: {pose_last['left']}", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color)
                        cv2.putText(flipped, f"PoseR: {pose_last['right']}", (x1 + 10, y1 + 110), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color)

                    cv2.rectangle(flipped, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                cv2.putText(flipped, "NN fps: {:.2f}".format(fps), (2, flipped.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

                if show_depth:
                    cv2.imshow("depth", depthFrameColor)

                resized = cv2.resize(flipped, (int(456.0*1.9), int(256.0*1.9)), interpolation = cv2.INTER_AREA)
                cv2.imshow("rgb", resized)
                if self.setWinPos:
                    self.setWinPos = False
                    cv2.moveWindow("rgb", 78, 30)

                self.imagePub.publish(self.bridge.cv2_to_imgmsg(flipped, "bgr8"))

                if cv2.waitKey(1) == ord('q'):
                    break

    def pose_detect_enable_callback(self, msg):
        global human_pose_process
        self.get_logger().info('Received EnablePoseDetect msg: mode: %d' % msg.enable)
        human_pose_process = msg.enable
        if msg.enable is False:
            clear_pose_detection()

    def process_detections(self, detections, w, h):
        # Build a message containing the objects.  Uses
        # a custom message format
        objList = []
        objListTrack = []

        for detection in detections:
            # Denormalize bounding box
            x1 = int(detection.xmin * w)
            x2 = int(detection.xmax * w)
            y1 = int(detection.ymin * h)
            y2 = int(detection.ymax * h)

            try:
                label = labelMap[detection.label]
            except:
                label = detection.label

            #print(detection)
            desc = ObjectDesc()
            desc.id = detection.label
            desc.name = label
            # Map to ROS convention
            desc.x = detection.spatialCoordinates.z
            desc.y = -1*detection.spatialCoordinates.x
            desc.z = 0.0
            desc.c = 0
            objList += (desc,)

            # Add to a list used by pan-tilt tracking if the object we
            # want to track
            if label == 'person':
                obj = {}
                obj['name'] = label
                obj['x'] = desc.x
                obj['y'] = desc.y
                obj['z'] = desc.z
                obj['x_min'] = detection.xmin
                obj['x_max'] = detection.xmax
                obj['y_min'] = detection.ymin
                obj['y_max'] = detection.ymax
                objListTrack.append(obj.copy())

        # Publish the object message to our topic
        msgObjects = ObjectDescArray()
        msgObjects.objects = objList
        self.objectPublisher.publish(msgObjects)

        # Update pan/tilt
        self.tracker.process_detections(objListTrack)

    def pose_thread(self, in_queue):
        global keypoints_list, detected_keypoints, personwiseKeypoints, new_pose

        while running:
            if human_pose_process != True:
                time.sleep(0.5)
                continue

            try:
                raw_in = in_queue.get()
            except RuntimeError:
                return

            self.pose_cnt += 1
            #if (self.pose_cnt % 2) == 0:
            if True:
                #fps.tick('nn')
                heatmaps = np.array(raw_in.getLayerFp16('Mconv7_stage2_L2')).reshape((1, 19, 32, 57))
                pafs = np.array(raw_in.getLayerFp16('Mconv7_stage2_L1')).reshape((1, 38, 32, 57))
                heatmaps = heatmaps.astype('float32')
                pafs = pafs.astype('float32')
                outputs = np.concatenate((heatmaps, pafs), axis=1)

                new_keypoints = []
                new_keypoints_list = np.zeros((0, 3))
                keypoint_id = 0

                for row in range(18):
                    probMap = outputs[0, row, :, :]
                    probMap = cv2.resize(probMap, (self.w, self.h))  # (456, 256)
                    keypoints = getKeypoints(probMap, 0.3)
                    new_keypoints_list = np.vstack([new_keypoints_list, *keypoints])
                    keypoints_with_id = []

                    for i in range(len(keypoints)):
                        keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                        keypoint_id += 1

                    new_keypoints.append(keypoints_with_id)

                valid_pairs, invalid_pairs = getValidPairs(outputs, self.w, self.h, new_keypoints)
                newPersonwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, new_keypoints_list)

                if human_pose_process == True:
                    detected_keypoints, keypoints_list, personwiseKeypoints = (new_keypoints, new_keypoints_list, newPersonwiseKeypoints)
                    new_pose = True

def main(args=None):
    rclpy.init(args=args)
    depthai_publisher = RobotHead()

    rclpy.spin(depthai_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    depthai_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


