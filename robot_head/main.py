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

show_depth = False
show_mono = False

running = True
pose = None

syncNN = True

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

    ######################################################
    # Detection
    # mono left/right  400P
    #    |        |
    #    V        V
    # --------------
    #| Stereo Depth |
    #|              |------|-----> to host (depth)
    # --------------       |
    #       | Rectified    | Depth
    #       V Right        V
    # --------------     ------------
    # |Image Manip |     |MobileNet  |--> to host (detections)
    # |300x300 rgb | --> |Spacial Det|--> to host (right - pass-thru image manip)
    # |grey        |  |  |           |--> to host (boundingBoxDepthMapping)
    # --------------  |  |-----------|
    #                 |
    #                 |--> to host (right)

    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoLeft.setFps(18.0);
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    monoLeft.setFps(18.0);

    print("Mono res: %s" % str(monoRight.getResolutionSize()))

    # Stereo Depth
    stereo = pipeline.createStereoDepth()
    stereo.setConfidenceThreshold(255)
    # Its inputs
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    # Image manip
    manip = pipeline.createImageManip()
    manip.initialConfig.setResize(300, 300)
    # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    manip.setKeepAspectRatio(False)
    # Its input
    stereo.rectifiedRight.link(manip.inputImage)

    # Define a neural network that will make predictions based on the source frames
    spatialDetectionNetwork = pipeline.createMobileNetSpatialDetectionNetwork()
    spatialDetectionNetwork.setConfidenceThreshold(0.5)
    spatialDetectionNetwork.setBlobPath(get_model_path('mobilenet-ssd_openvino_2021.2_6shave.blob'))
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)
    # Its inputs
    manip.out.link(spatialDetectionNetwork.input)
    stereo.depth.link(spatialDetectionNetwork.inputDepth)

    # Create outputs to the host

    if show_mono:
        xoutManip = pipeline.createXLinkOut()
        xoutManip.setStreamName("right")
        if syncNN:
            spatialDetectionNetwork.passthrough.link(xoutManip.input)
        else:
            manip.out.link(xoutManip.input)

    depthRoiMap = pipeline.createXLinkOut()
    depthRoiMap.setStreamName("boundingBoxDepthMapping")
    spatialDetectionNetwork.boundingBoxMapping.link(depthRoiMap.input)

    if show_depth:
        xoutDepth = pipeline.createXLinkOut()
        xoutDepth.setStreamName("depth")
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

    nnOut = pipeline.createXLinkOut()
    nnOut.setStreamName("detections")
    spatialDetectionNetwork.out.link(nnOut.input)

    ######################################################
    # Human Pose Detection

    # Color camera
    colorCam = pipeline.createColorCamera()
    colorCam.setPreviewSize(456, 256)
    colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    colorCam.setInterleaved(False)
    colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    # Run at 18 fps but only display/publish/process pose as a fraction of that rate.
    # This provides a faster update rate for the tracking function which is based on the
    # detection nn output.
    colorCam.setFps(18.0);

    if human_pose:
        # NeuralNetwork - human pose
        print("Creating Human Pose Estimation Neural Network...")

        pose_nn = pipeline.createNeuralNetwork()
        pose_nn.setBlobPath(get_model_path('human-pose-estimation-0001_openvino_2021.2_6shave.blob'))
        # Increase threads for detection
        # This slowed overall thru put
        #pose_nn.setNumInferenceThreads(2)
        pose_nn.setNumInferenceThreads(1)
        # Specify that network takes latest arriving frame in non-blocking manner
        pose_nn.input.setQueueSize(1)
        pose_nn.input.setBlocking(False)
        # Its input
        colorCam.preview.link(pose_nn.input)

        # Create outputs to the host
        pose_nn_xout = pipeline.createXLinkOut()
        pose_nn_xout.setStreamName("pose_nn")
        pose_nn.out.link(pose_nn_xout.input)

    xoutRgb = pipeline.createXLinkOut()
    xoutRgb.setStreamName("rgb")
    colorCam.preview.link(xoutRgb.input)

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
        # Needed for pose processing thread
        self.h = 256
        self.w = 456

        self.setWinPos = True;
        show_mono = False

        pose_last = None

        with dai.Device(create_pipeline()) as device:
            print("Starting pipeline...")
            device.startPipeline()

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueueRGB = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
            if show_mono:
                previewQueueMono = device.getOutputQueue(name="right", maxSize=1, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=1, blocking=False)
            xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=2, blocking=False)
            if show_depth:
                depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)

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

            disp_cnt = 0

            flipMono = False
            flipRGB = False

            # Rough scaling for converting bounding box relative to 300x300 right mono
            # as seen by detection, and the 456x256 color camera output. (Obtained
            # by compare video.)
            bbXScale = 456.0/(291.0-28.0)
            bbYScale = 256.0/(274.0-39.0)
            bbXOffset = 28.0
            bbYOffset = 39.0

            detectXScale = 1.0 #300.0/640.0

            while True:

                rclpy.spin_once(self);

                inPreviewRGB = previewQueueRGB.get()
                if show_mono:
                    inPreviewMono = previewQueueMono.get()
                inNN = detectionNNQueue.get()

                counter+=1
                current_time = time.monotonic()
                if (current_time - startTime) > 1 :
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                disp_cnt += 1
                show_frame = (disp_cnt % 2) == 0

                frameRGB = inPreviewRGB.getCvFrame()
                self.h, self.w = frameRGB.shape[:2]  # 256, 456

                detections = inNN.detections

                if show_depth:
                    depth = depthQueue.get()
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
                    cv2.imshow("depth", depthFrameColor)

                if show_mono:
                    frameMono= inPreviewMono.getCvFrame()

                    if flipMono:
                        frameMono = cv2.flip(frameMono, 1)

                    # If the frame is available, draw bounding boxes on it and show the frame
                    height = frameMono.shape[0]
                    width  = frameMono.shape[1]

                    for detection in detections:
                        # Denormalize bounding box
                        if flipMono:
                            x2 = int((1.0 - detection.xmin) * width)
                            x1 = int((1.0 - detection.xmax) * width)
                        else:
                            x1 = int(detection.xmin * width)
                            x2 = int(detection.xmax * width)
                        y1 = int(detection.ymin * height)
                        y2 = int(detection.ymax * height)

                        try:
                            label = labelMap[detection.label]
                        except:
                            label = detection.label

                        cv2.putText(frameMono, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                        cv2.putText(frameMono, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                        cv2.putText(frameMono, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                        cv2.putText(frameMono, f"Y: {int(detection.spatialCoordinates.y*detectXScale)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                        cv2.putText(frameMono, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

                        cv2.rectangle(frameMono, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                    cv2.putText(frameMono, "NN fps: {:.2f}".format(fps), (2, frameMono.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
                    cv2.imshow("Mono", frameMono)

                # Publish and update tracker
                self.process_detections(detections, 300, 300, detectXScale)

                # Human pose detection processing
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

                if show_frame:
                    heightRGB = frameRGB.shape[0]
                    widthRGB  = frameRGB.shape[1]

                    # Display Human pose
                    try:
                        if keypoints_list is not None and detected_keypoints is not None and personwiseKeypoints is not None:
                            for i in range(18):
                                for j in range(len(detected_keypoints[i])):
                                    cv2.circle(frameRGB, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
                            for i in range(17):
                                for n in range(len(personwiseKeypoints)):
                                    index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                                    if -1 in index:
                                        continue
                                    B = np.int32(keypoints_list[index.astype(int), 0])
                                    A = np.int32(keypoints_list[index.astype(int), 1])
                                    cv2.line(frameRGB, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
                    except:
                        print("keypoint out of bound")

                    frameRGB = cv2.flip(frameRGB, 1)

                    # Display detections
                    for detection in detections:
                        try:
                            label = labelMap[detection.label]
                        except:
                            label = detection.label
                        if label != 'person':
                            continue

                        # Denormalize bounding box
                        # Approx scaling of detection BBs
                        x1 = max(0, int((detection.xmin*300 - bbXOffset)*bbXScale))
                        x2 = min(widthRGB, int((detection.xmax*300 - bbXOffset)*bbXScale))
                        y1 = max(0, int((detection.ymin*300 - bbYOffset)*bbYScale))
                        y2 = min(heightRGB, int((detection.ymax*300 - bbYOffset)*bbYScale))

                        if flipRGB:
                            swap = x2
                            x2 = widthRGB - x1
                            x1 = widthRGB - swap
                        font_scale = 0.4
                        cv2.putText(frameRGB, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color)
                        #cv2.putText(frameRGB, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color)
                        # ROS coords
                        cv2.putText(frameRGB, f"X,Y,Z: {int(detection.spatialCoordinates.z)}, {-1*int(detection.spatialCoordinates.x*detectXScale)}, {int(detection.spatialCoordinates.y)} mm",
                                    (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color)

                        # Fix - handle multiple persons
                        if pose_last is not None and label == 'person':
                            cv2.putText(frameRGB, f"PoseL: {pose_last['left']}", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color)
                            cv2.putText(frameRGB, f"PoseR: {pose_last['right']}", (x1 + 10, y1 + 110), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color)

                        cv2.rectangle(frameRGB, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                    cv2.putText(frameRGB, "NN fps: {:.2f}".format(fps), (2, frameRGB.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

                    #frameRGB = cv2.resize(frameRGB, (int(456.0), int(256.0)), interpolation = cv2.INTER_AREA)
                    frameRGB = cv2.resize(frameRGB, (int(456.0*1.9), int(256.0*1.9)), interpolation = cv2.INTER_AREA)
                    cv2.imshow("rgb", frameRGB)
                    if self.setWinPos:
                        self.setWinPos = False
                        cv2.moveWindow("rgb", 78, 30)

                    self.imagePub.publish(self.bridge.cv2_to_imgmsg(frameRGB, "bgr8"))

                if cv2.waitKey(1) == ord('q'):
                    break

    def pose_detect_enable_callback(self, msg):
        global human_pose_process
        self.get_logger().info('Received EnablePoseDetect msg: mode: %d' % msg.enable)
        human_pose_process = msg.enable
        if msg.enable is False:
            clear_pose_detection()

    def process_detections(self, detections, w, h, detectXScale):
        # Build a message containing the objects.  Uses
        # a custom message format
        objList = []
        objListTrack = []

        for detection in detections:
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
            # Scale since detections are running on a distorted 640 wide image
            desc.y = -1*detection.spatialCoordinates.x*detectXScale
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
            if (self.pose_cnt % 2) == 0:
            #if True:
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


