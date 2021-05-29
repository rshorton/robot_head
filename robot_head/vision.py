
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

from pose_interp import analyze_pose

# Custom object detection messages
from object_detection_msgs.msg import ObjectDescArray
from object_detection_msgs.msg import ObjectDesc

from human_pose_interfaces.msg import DetectedPose, EnablePoseDetection

import mediapipe_utils as mpu
from BlazeposeDepthai import BlazeposeDepthai, to_planar

human_pose = True
human_pose_process = True

show_depth = True

syncNN = False

labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def get_model_path(model_name):
    return str(pathlib.Path(__file__).parent.absolute()) + '/models/' + model_name

class RobotVision(Node):
    def __init__(self):
        super().__init__('robot_vision')

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

        self.setWinPos = True;

        pose_last = None

        blaze_pose = BlazeposeDepthai()

        with dai.Device(self.create_pipeline()) as device:
            print("Starting pipeline...")
            device.startPipeline()

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueueRGB = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=1, blocking=False)
            xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=2, blocking=False)
            if show_depth:
                depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
            if human_pose:
                q_pd_out = device.getOutputQueue(name="pd_out", maxSize=1, blocking=False)
                q_lm_out = device.getOutputQueue(name="lm_out", maxSize=2, blocking=False)
                q_lm_in = device.getInputQueue(name="lm_in")

            frame = None
            detections = []

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (64, 255, 64)
            font_scale = 0.4
            disp_cnt = 0

            flipRGB = True

            last_region = None
            last_poses = None

            while True:

                rclpy.spin_once(self, timeout_sec=0.001);

                try:
                    inPreviewRGB = previewQueueRGB.tryGet()
                    inNN = detectionNNQueue.tryGet()
                except:
                    print("Failed to read from queue.")
                    continue

                if inNN != None:
                    detections = inNN.detections
                    #print("got detections")
                    # Publish detections
                    self.publish_detections(detections)

                if inPreviewRGB == None:
                    continue

                try:
                    frameRGB = inPreviewRGB.getCvFrame()
                except:
                    print("Failed to read preview frame")
                    continue

                #print('got rgb frame')

                # The image preview can be non-square although a square area is fed to the PD NN and LM NN.
                # Determine the xoffset on the left and right side of the image around
                # the square area.  This assumes a wider than taller image.

                heightRGB, widthRGB = frameRGB.shape[0:2]
                frame_size_lm = heightRGB
                xoffset = int((widthRGB - frame_size_lm)/2)

                counter += 1
                current_time = time.monotonic()
                if (current_time - startTime) > 1 :
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                disp_cnt += 1
                #show_frame = (disp_cnt % 2) == 0
                show_frame = False
                pub_frame = (disp_cnt % 2) == 0

                if human_pose and human_pose_process:
                    frameRGBOrig = frameRGB.copy()

                    # Get pose detection
                    inference = q_pd_out.tryGet()
                    if inference != None:
                        regions = blaze_pose.pd_postprocess(inference, frame_size_lm)
                        blaze_pose.pd_render(frameRGB, frame_size_lm, xoffset)

                        # Landmarks
                        blaze_pose.nb_active_regions = 0
                        for i,r in enumerate(regions):
                            frame_cropped = frameRGBOrig[0:(frameRGBOrig.shape[1]+1),xoffset:(xoffset+frame_size_lm)]
                            frame_nn = mpu.warp_rect_img(r.rect_points, frame_cropped, blaze_pose.lm_input_length, blaze_pose.lm_input_length)
                            nn_data = dai.NNData()
                            nn_data.setLayer("input_1", to_planar(frame_nn, (blaze_pose.lm_input_length, blaze_pose.lm_input_length)))
                            q_lm_in.send(nn_data)

                            # Get landmarks
                            inference = q_lm_out.get()
                            blaze_pose.lm_postprocess(r, inference, xoffset)
                            last_region = r

                            last_poses = analyze_pose(r)
                            if last_poses != None:
                                self.publish_poses(last_poses)

                        blaze_pose.check_filter_reset()
                        if blaze_pose.nb_active_regions == 0:
                            last_region = None
                            last_pose = None

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

                if show_frame or pub_frame:

                    if last_region != None:
                        blaze_pose.lm_render(frameRGB, last_region)

                    if flipRGB:
                        frameRGB = cv2.flip(frameRGB, 1)

                    if last_poses is not None:
                        cv2.putText(frameRGB, f"PoseL: {last_poses['left']}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color)
                        cv2.putText(frameRGB, f"PoseR: {last_poses['right']}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color)

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
                        x1 = int(detection.xmin*widthRGB)
                        x2 = int(detection.xmax*widthRGB)
                        y1 = int(detection.ymin*heightRGB)
                        y2 = int(detection.ymax*heightRGB)

                        if flipRGB:
                            swap = x2
                            x2 = widthRGB - x1
                            x1 = widthRGB - swap

                        cv2.putText(frameRGB, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color)
                        cv2.putText(frameRGB, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color)
                        # ROS coords
                        cv2.putText(frameRGB, f"X,Y,Z: {int(detection.spatialCoordinates.z)}, {-1*int(detection.spatialCoordinates.x)}, {int(detection.spatialCoordinates.y)} mm",
                                    (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color)

                        # Fix - handle multiple persons
                        if pose_last is not None:
                            cv2.putText(frameRGB, f"PoseL: {pose_last['left']}", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color)
                            cv2.putText(frameRGB, f"PoseR: {pose_last['right']}", (x1 + 10, y1 + 110), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color)

                        cv2.rectangle(frameRGB, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                    cv2.putText(frameRGB, "NN fps: {:.2f}".format(fps), (2, frameRGB.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

                    #frameRGB = cv2.resize(frameRGB, (int(456.0), int(256.0)), interpolation = cv2.INTER_AREA)
                    #frameRGB = cv2.resize(frameRGB, (int(456.0*1.9), int(256.0*1.9)), interpolation = cv2.INTER_AREA)

                    if show_frame:
                        cv2.imshow("rgb", frameRGB)
                        if self.setWinPos:
                            self.setWinPos = False
                            cv2.moveWindow("rgb", 78, 30)

                    if pub_frame:
                        self.imagePub.publish(self.bridge.cv2_to_imgmsg(frameRGB, "bgr8"))

                if cv2.waitKey(1) == ord('q'):
                    break

    def create_pipeline(self):
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_2)

        monoLeft = pipeline.createMonoCamera()
        monoRight = pipeline.createMonoCamera()
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoLeft.setFps(10.0);
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        monoLeft.setFps(10.0);

        print("Mono res: %s" % str(monoRight.getResolutionSize()))

        # Stereo Depth
        stereo = pipeline.createStereoDepth()
        stereo.setConfidenceThreshold(255)
        # Its inputs
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        # Color camera
        colorCam = pipeline.createColorCamera()
        colorCam.setPreviewSize(640, 360)
        colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        colorCam.setInterleaved(False)
        colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        colorCam.setFps(10.0)
        colorCam.setPreviewKeepAspectRatio(False)

        # Mobilenet SSD detection
        # Use Image manip to size accordingly
        manip_mn = pipeline.createImageManip()
        manip_mn.initialConfig.setResize(300, 300)
        manip_mn.setKeepAspectRatio(False)
        colorCam.preview.link(manip_mn.inputImage)

        spatialDetectionNetwork = pipeline.createMobileNetSpatialDetectionNetwork()
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.setBlobPath(get_model_path('mobilenet-ssd_openvino_2021.2_6shave.blob'))
        spatialDetectionNetwork.input.setQueueSize(1)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.3)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(10000)
        # Its inputs
        manip_mn.out.link(spatialDetectionNetwork.input)
        stereo.depth.link(spatialDetectionNetwork.inputDepth)

        # Create outputs to the host
        xoutRgb = pipeline.createXLinkOut()
        xoutRgb.setStreamName("rgb")
        if syncNN:
            spatialDetectionNetwork.passthrough.link(xoutRgb.input)
        else:
            colorCam.preview.link(xoutRgb.input)

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
        # Human Pose Detection, Blazepose

        if human_pose:
            # Scale cam preview output for Blazepose person detector
            manip_bp = pipeline.createImageManip()
            manip_bp.initialConfig.setResize(128, 128)
            manip_bp.setKeepAspectRatio(True)
            # Its input
            colorCam.preview.link(manip_bp.inputImage)

            # First stage pose detector
            print("Creating Pose Detection Neural Network...")
            pd_nn = pipeline.createNeuralNetwork()
            pd_nn.setBlobPath(get_model_path("pose_detection.blob"))
            # Increase threads for detection
            # pd_nn.setNumInferenceThreads(2)
            # Specify that network takes latest arriving frame in non-blocking manner
            # Pose detection input
            pd_nn.input.setQueueSize(1)
            pd_nn.input.setBlocking(False)
            # Its input
            manip_bp.out.link(pd_nn.input)

            # Pose detection output
            pd_out = pipeline.createXLinkOut()
            pd_out.setStreamName("pd_out")
            pd_nn.out.link(pd_out.input)

             # Define landmark model
            print("Creating Landmark Neural Network...")
            lm_nn = pipeline.createNeuralNetwork()
            lm_nn.setBlobPath(get_model_path("pose_landmark_full_body.blob"))
            lm_nn.setNumInferenceThreads(1)
            # Landmark input
            lm_in = pipeline.createXLinkIn()
            lm_in.setStreamName("lm_in")
            lm_in.out.link(lm_nn.input)
            # Landmark output
            lm_out = pipeline.createXLinkOut()
            lm_out.setStreamName("lm_out")
            lm_nn.out.link(lm_out.input)

        print("Pipeline created.")
        return pipeline

    def publish_detections(self, detections):
        # Build a message containing the objects.  Uses
        # a custom message format
        objList = []

        for detection in detections:
            #label = 'person'
            try:
                label = labelMap[detection.label]
            except:
                label = detection.label

            # Hack to avoid false person detection of blank wall
            if (label == 'person' and (detection.xmax - detection.xmin) > 0.7):
                continue

            #print(detection)
            desc = ObjectDesc()
            desc.id = detection.label
            desc.name = label
            desc.confidence = detection.confidence
            # Map to ROS convention
            desc.x = detection.spatialCoordinates.z
            desc.y = -1*detection.spatialCoordinates.x
            desc.z = 0.0
            desc.c = 0
            desc.x_min = detection.xmin
            desc.x_max = detection.xmax
            desc.y_min = detection.ymin
            desc.y_max = detection.ymax
            objList += (desc,)

        # Publish the object message to our topic
        msgObjects = ObjectDescArray()
        msgObjects.objects = objList
        self.objectPublisher.publish(msgObjects)

    def publish_poses(self, poses):
        msg = DetectedPose()
        msg.detected = poses["detected"]
        msg.left = poses['left']
        msg.right = poses['right']
        msg.num_points = poses['num_points']
        self.posePublisher.publish(msg)

    def pose_detect_enable_callback(self, msg):
        global human_pose_process
        self.get_logger().info('Received EnablePoseDetect msg: mode: %d' % msg.enable)
        human_pose_process = msg.enable
        if msg.enable is False:
            last_region = None

def main(args=None):
    rclpy.init(args=args)
    depthai_publisher = RobotVision()

    rclpy.spin(depthai_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    depthai_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


