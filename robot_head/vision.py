
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

show_depth = False
cam_out_use_preview = True
use_tracker = True
print_detections = False
syncNN = False

labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def get_model_path(model_name):
    return str(pathlib.Path(__file__).parent.absolute()) + '/models/' + model_name

def OverlayTextOnBox(img, x, y, xpad, ypad, text, bg_color, alpha, font, font_scale, text_color, text_thick):
    w = 0
    hmax = 0
    for t in text:
        (tw, th) = cv2.getTextSize(t, font, fontScale=font_scale, thickness=text_thick)[0]
        w = max(w, 2*xpad + tw)
        hmax = max(hmax, th)
    h = (ypad + hmax)*len(text) + ypad

    sub = img[y:y+h, x:x+w]
    bg = np.zeros_like(sub)
    bg[:] = bg_color
    blend = cv2.addWeighted(sub, 1.0 - alpha, bg, alpha, 0)

    yy = 0
    for t in text:
        cv2.putText(blend, t, (xpad, yy + ypad + hmax), font, font_scale, text_color, text_thick, cv2.LINE_AA)
        yy += ypad + hmax
    img[y:y+h, x:x+w] = blend
    return x, y, x + w, y + h

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

        self.subTracked = self.create_subscription(
            ObjectDesc,
            '/head/tracked',
            self.tracked_callback,
            1)

        self.setWinPos = True

        self.tracked_obj = None

        pose_last = None

        blaze_pose = BlazeposeDepthai(multi_detection = True)

        with dai.Device(self.create_pipeline()) as device:
            print("Starting pipeline...")
            device.startPipeline()

            previewQueueCAM = device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)

            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=1, blocking=False)
            xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=2, blocking=False)
            if show_depth:
                depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
            if human_pose:
                q_pd_out = device.getOutputQueue(name="pd_out", maxSize=1, blocking=False)
                q_lm_out = device.getOutputQueue(name="lm_out", maxSize=2, blocking=False)
                q_lm_in = device.getInputQueue(name="lm_in")

            frame = None
            tracklets = []

            startTime = time.monotonic()
            counter = 0
            fps = 0
            white = (255, 255, 255)
            color = (0, 255, 0)
            rect_color = (128, 128, 128)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            disp_cnt = 0

            flipCAM = True

            last_region = None
            last_poses = None

            while True:

                rclpy.spin_once(self, timeout_sec=0.001);

                try:
                    inPreviewCAM = previewQueueCAM.tryGet()
                    inNN = detectionNNQueue.tryGet()
                except:
                    print("Failed to read from queue.")
                    continue

                if inNN != None:
                    tracklets = inNN.tracklets

                    if print_detections:
                        for tracklet in tracklets:
                            print("------------")
                            print(tracklet.id)
                            print(tracklet.label)
                            roi = tracklet.roi.denormalize(640, 360)
                            print("roi: %d,%d -- %d,%d" % (int(roi.topLeft().x), int(roi.topLeft().y), int(roi.bottomRight().x), int(roi.bottomRight().y)))
                            print("spacial: %f, %f, %f" % (tracklet.spatialCoordinates.x, tracklet.spatialCoordinates.y, tracklet.spatialCoordinates.z))
                            print("lable %s, conf %f, xmin %f, ymin %f, xmax %f, ymax %f" % (tracklet.srcImgDetection.label, \
                                tracklet.srcImgDetection.confidence, tracklet.srcImgDetection.xmin, tracklet.srcImgDetection.ymin, \
                                tracklet.srcImgDetection.xmax, tracklet.srcImgDetection.ymax))
                            print(tracklet.status)

                    # Publish detections
                    self.publish_detections(tracklets)

                if inPreviewCAM == None:
                    continue

                try:
                    frameCAM = inPreviewCAM.getCvFrame()
                except:
                    print("Failed to read preview frame")
                    continue

                #print('got cam frame')

                # The Pose landmark NN input needs to be square.
                # Determine the xoffset on the left and right side of the image around
                # the square area.  This assumes a wider than taller image.
                #print("frame size %s" % str(frameCAM.shape))
                heightCAM, widthCAM = frameCAM.shape[0:2]
                frame_size_lm = heightCAM
                xoffset = int((widthCAM - frame_size_lm)/2)

                counter += 1
                current_time = time.monotonic()
                if (current_time - startTime) > 1 :
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                disp_cnt += 1
                show_frame = False
                pub_frame = (disp_cnt % 2) == 0

                if human_pose and human_pose_process:
                    frameCAMOrig = frameCAM.copy()

                    # Get pose detection
                    inference = q_pd_out.tryGet()
                    if inference != None:
                        regions = blaze_pose.pd_postprocess(inference, frame_size_lm)
                        #regions = blaze_pose.pd_postprocess(inference, frame_size_lm)
                        blaze_pose.pd_render(frameCAM, frame_size_lm, xoffset)

                        # Try to find a region (person) that corresponds to the
                        # the detected person reported by the Tracker node
                        sel_region = None
                        if self.tracked_obj != None and self.tracked_obj.confidence > 0.0:
                            tracked_x = (self.tracked_obj.x_min + self.tracked_obj.x_max)/2
                            closest_diff = 0.0

                            for i,r in enumerate(regions):
                                # Calc normalized center of detected person
                                x = ((r.rect_points[0][1] + r.rect_points[3][1])/2 + xoffset)/widthCAM
                                diff = abs(x - tracked_x)
                                #print("region= %d, x center= %f, tracked_x= %f" % (i, x, tracked_x))
                                if sel_region == None or closest_diff > diff:
                                    sel_region = i
                                    closest_diff = diff

                        # Landmarks
                        blaze_pose.nb_active_regions = 0
                        for i,r in enumerate(regions):
                            if i != sel_region:
                                continue

                            video_frame = frameCAMOrig[0:(frame_size_lm+1),xoffset:(xoffset+frame_size_lm)]
                            frame_nn = mpu.warp_rect_img(r.rect_points, video_frame, blaze_pose.lm_input_length, blaze_pose.lm_input_length)

                            nn_data = dai.NNData()
                            nn_data.setLayer("input_1", to_planar(frame_nn, (blaze_pose.lm_input_length, blaze_pose.lm_input_length)))
                            q_lm_in.send(nn_data)

                            # Get landmarks
                            inference = q_lm_out.get()
                            blaze_pose.lm_postprocess(r, inference, xoffset)
                            #blaze_pose.lm_postprocess(r, inference)
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

                    #if len(detections) != 0:

                    boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
                    if boundingBoxMapping != None:
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
                        blaze_pose.lm_render(frameCAM, last_region)

                    if flipCAM:
                        frameCAM = cv2.flip(frameCAM, 1)

                    if last_poses is not None:
                        line1 = f"PoseL: {last_poses['left']}"
                        line2 = f"PoseR: {last_poses['right']}"
                        OverlayTextOnBox(frameCAM, 0, 0, 10, 10, [line1, line2], (0, 0, 0), 0.4, font, 0.6, white, 1)

                    # Display detections
                    for tracklet in tracklets:
                        try:
                            label = labelMap[tracklet.label]
                        except:
                            label = tracklet.label
                        if label != 'person' or tracklet.status != dai.Tracklet.TrackingStatus.TRACKED:
                            continue

                        # Denormalize bounding box
                        x1 = int(tracklet.srcImgDetection.xmin*widthCAM)
                        x2 = int(tracklet.srcImgDetection.xmax*widthCAM)
                        y1 = int(tracklet.srcImgDetection.ymin*heightCAM)
                        y2 = int(tracklet.srcImgDetection.ymax*heightCAM)

                        if flipCAM:
                            swap = x2
                            x2 = max(0, widthCAM - x1)
                            x1 = max(0, widthCAM - swap)

                        y1 = max(0, y1)
                        y2 = max(0, y2)

                        if self.tracked_obj != None and self.tracked_obj.id == tracklet.id:
                            conf = "{:d}%".format(int(tracklet.srcImgDetection.confidence*100))
                            pos_x = f"x: {int(tracklet.spatialCoordinates.z)}"
                            pos_y = f"y: {int(tracklet.spatialCoordinates.x)*-1}"
                            pos_z = f"z: {int(tracklet.spatialCoordinates.y)}"
                            trk_id = f'id: {tracklet.id}'
                            OverlayTextOnBox(frameCAM, x1 + 2, y1 + 2, 5, 5, [conf, pos_x, pos_y, pos_z, trk_id], (0, 0, 0), 0.4, font, font_scale, white, 1)

                        cv2.rectangle(frameCAM, (x1, y1), (x2, y2), white, font)

                    OverlayTextOnBox(frameCAM, 0, frameCAM.shape[0] - 25, 5, 5, ["fps: {:.2f}".format(fps)], (0, 0, 0), 0.4, font, font_scale, white, 1)

                    #cv2.putText(frameCAM, "fps: {:.2f}".format(fps), (2, frameCAM.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

                    if show_frame:
                        cv2.imshow("camera", frameCAM)
                        if self.setWinPos:
                            self.setWinPos = False
                            cv2.moveWindow("camera", 78, 30)

                    if pub_frame:
                        self.imagePub.publish(self.bridge.cv2_to_imgmsg(frameCAM, "bgr8"))

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

        # Stereo Depth
        stereo = pipeline.createStereoDepth()
        stereo.setConfidenceThreshold(255)
        #stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
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

        # Scale video from 1920x1080 to 640x360
        colorCam.setIspScale((1,3))

        manip_mn = pipeline.createImageManip()
        manip_mn.initialConfig.setResize(300, 300)
        #manip_mn.setKeepAspectRatio(True)
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
        #colorCam.preview.link(spatialDetectionNetwork.input)
        stereo.depth.link(spatialDetectionNetwork.inputDepth)

        if use_tracker:
            # Create object tracker
            objectTracker = pipeline.createObjectTracker()
            # track only person
            objectTracker.setDetectionLabelsToTrack([15])
            # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS
            objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
            objectTracker.setTrackerIdAssigmentPolicy(dai.TrackerIdAssigmentPolicy.UNIQUE_ID)
            # Its input
            spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)
            spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
            spatialDetectionNetwork.out.link(objectTracker.inputDetections)

        # Create outputs to the host
        xoutCam = pipeline.createXLinkOut()
        xoutCam.setStreamName("cam_out")
        if cam_out_use_preview:
            if syncNN:
                spatialDetectionNetwork.passthrough.link(xoutCam.input)
            else:
                colorCam.preview.link(xoutCam.input)
        else:
            colorCam.video.link(xoutCam.input)

        depthRoiMap = pipeline.createXLinkOut()
        depthRoiMap.setStreamName("boundingBoxDepthMapping")
        spatialDetectionNetwork.boundingBoxMapping.link(depthRoiMap.input)

        if show_depth:
            xoutDepth = pipeline.createXLinkOut()
            xoutDepth.setStreamName("depth")
            spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

        nnOut = pipeline.createXLinkOut()
        nnOut.setStreamName("detections")

        if use_tracker:
            objectTracker.out.link(nnOut.input)
        else:
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

    def publish_detections(self, tracklets):
        # Build a message containing the objects.  Uses
        # a custom message format
        objList = []

        for tracklet in tracklets:
            #label = 'person'
            try:
                label = labelMap[tracklet.label]
            except:
                label = tracklet.label

            # Hack to avoid false person detection of blank wall
            if (label == 'person' and (tracklet.srcImgDetection.xmax - tracklet.srcImgDetection.xmin) > 0.7):
                continue

            #print(detection)
            desc = ObjectDesc()
            desc.id = tracklet.id
            desc.track_status = str(tracklet.status).split(".")[1]
            desc.name = label
            desc.confidence = tracklet.srcImgDetection.confidence
            # Map to ROS convention
            desc.x = tracklet.spatialCoordinates.z
            desc.y = -1*tracklet.spatialCoordinates.x
            desc.z = tracklet.spatialCoordinates.y
            desc.c = 0
            desc.x_min = tracklet.srcImgDetection.xmin
            desc.x_max = tracklet.srcImgDetection.xmax
            desc.y_min = tracklet.srcImgDetection.ymin
            desc.y_max = tracklet.srcImgDetection.ymax
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

    def tracked_callback(self, msg):
        self.tracked_obj = msg

def main(args=None):
    rclpy.init(args=args)
    depthai_publisher = RobotVision()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    depthai_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


