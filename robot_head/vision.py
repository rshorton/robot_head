# Copyright 2021 Scott Horton
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
#
# Adapted from various Luxonis DepthAI sample applications.
#
# Includes Blazepose tracking with DepthAI
# From https://github.com/geaxgx/depthai_blazepose
# Credit:  geaxgx

import time
import pathlib
from itertools import islice

import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2

import depthai as dai

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage

from object_detection_msgs.msg import ObjectDescArray
from object_detection_msgs.msg import ObjectDesc
from robot_head_interfaces.msg import TrackStatus, SaveImage, DetectedPose, EnablePoseDetection, HeadImu

from pose_interp import analyze_pose

import mediapipe_utils as mpu
from BlazeposeDepthai import BlazeposeDepthai, to_planar
from hailo_zmq_meta_sink import hailo_zmq_meta_sink

# Fix - not working since updating to latest Depthai - 
human_pose = False
human_pose_process = True

ball_detect = False

show_depth = False
cam_out_use_preview = True
use_tracker = True
print_detections = False
show_det_info = True
show_hailo_meta_info = True
show_edge_image = False 
syncNN = False
use_tyolo_v4 = False
use_imu = False
pub_detection_frame = True

hailo_ai_pipeline = True
pub_compressed_image = True

objects_to_track = ["person", "cat"]
min_conf = 0.6

frame_rate = 10.0

labelMap_MNetSSD = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

labelMap_TYolo4 = [
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
    "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"]

def get_model_path(model_name):
    return str(pathlib.Path(__file__).parent.absolute()) + '/models/' + model_name

# Helper for displaying overlay text on an image
def OverlayTextOnBox(img, x, y, xpad, ypad, text, bg_color, alpha, font, font_scale, text_color, text_thick):
    #Fix
    try:
        w = 0
        hmax = 0
        for t in text:
            if t == None:
                continue
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
            if t == None:
                continue
            cv2.putText(blend, t, (xpad, yy + ypad + hmax), font, font_scale, text_color, text_thick, cv2.LINE_AA)
            yy += ypad + hmax
        img[y:y+h, x:x+w] = blend
        return x, y, x + w, y + h
    except:
        return x, y, x, y
    
def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None

class RobotVision(Node):
    def __init__(self):
        super().__init__('robot_vision')

        self.image = None
        # Publisher for camera color image
        if pub_compressed_image:
            self.imagePub = self.create_publisher(CompressedImage, '/color/image/compressed', 10)
        else:            
            self.imagePub = self.create_publisher(Image, '/color/image', 10)

        self.bridge = CvBridge()

        self.detectionImagePub = self.create_publisher(Image, '/detection_image', 10)

        # Publisher for list of detected objects
        self.objectPublisher = self.create_publisher(
            ObjectDescArray,
            '/head/detected_objects',
            10)

        # Publisher for list of target objects
        self.targetPublisher = self.create_publisher(
            ObjectDescArray,
            '/head/detected_targets',
            10)

        # Publisher for detected human pose
        self.posePublisher = self.create_publisher(
            DetectedPose,
            '/head/detected_pose',
             1)

        # Publisher for head rotation
        self.headImuPublisher = self.create_publisher(
            HeadImu,
            '/head/imu',
             1)

        # Subscribe to topic for enabling/disabling pose detection
        self.subPoseDetectEnable = self.create_subscription(
            EnablePoseDetection,
            '/head/enable_pose_detect',
            self.pose_detect_enable_callback,
            1)

        # Subscribe to topic specifying which object is being
        # tracked by the Tracker node
        self.subTracked = self.create_subscription(
            TrackStatus,
            '/head/track_status',
            self.tracked_callback,
            1)

        # Subscribe to topic for triggering an image save
        self.subSaveImage = self.create_subscription(
            SaveImage,
            '/head/save_image',
            self.save_image_callback,
            1)

        self.setWinPos = True

        self.tracked = None
        self.save_image = None

        if hailo_ai_pipeline:
            self.hailo_meta_sink_face_recog = hailo_zmq_meta_sink(port = 5558)
            self.hailo_meta_sink_face_recog.open()

            self.hailo_meta_sink_person_reid = hailo_zmq_meta_sink(port = 5559)
            self.hailo_meta_sink_person_reid.open()

            # Publisher for camera color image sent to Hailo AI
            # pipeline without annotations.
            self.imagePubToHailo = self.create_publisher(Image, '/color/image_to_hailo', 10)


        blaze_pose = BlazeposeDepthai(multi_detection = True)

        with dai.Device(self.create_pipeline()) as device:
            self.get_logger().debug("Starting pipeline...")
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
            if ball_detect:
                edgeImgQueue = device.getOutputQueue(name="edge_image", maxSize=4, blocking=False)
                ballDetectionNNQueue = device.getOutputQueue(name="ball_detections", maxSize=4, blocking=False)
            if use_imu:
                imuQueue = device.getOutputQueue(name="imu", maxSize=2, blocking=False)

            edgeFrame = None
            inEdgeFrame = None
            inBallDetectNN = None
            imuData = None
            tracklets = []

            startTime = time.monotonic()
            counter = 0
            fps = 0
            white = (255, 255, 255)
            yellow = (0, 255, 255)
            blue = (255, 0, 0)
            color = (0, 255, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            disp_cnt = 0

            flipCAM = True

            last_region = None
            last_poses = None

            save_cnt = 0

            self.meta_sync_buffer = {}
            self.next_person_id = 1
            self.hailo_person_list = {}
            self.tracklet_to_hailo_meta = {}

            while True:

                rclpy.spin_once(self, timeout_sec=1.0/frame_rate/2.0)

                # Read meta data from the Hailo AI pipeline
                if hailo_ai_pipeline:
                    self.read_hailo_meta()

                try:
                    inPreviewCAM = previewQueueCAM.tryGet()
                    inNN = detectionNNQueue.tryGet()

                    if ball_detect:
                        if show_edge_image:
                            inEdgeFrame = edgeImgQueue.tryGet()
                        if inEdgeFrame:
                            inBallDetectNN = ballDetectionNNQueue.tryGet()

                    if use_imu:
                        imuData = imuQueue.tryGet()

                except:
                    self.get_logger().error("Failed to read from queue.")
                    continue

                if inNN != None:
                    tracklets = inNN.tracklets

                    # Map the Hailo meta data (currently facial recog)
                    # to the tracklets
                    if hailo_ai_pipeline and len(self.hailo_person_list) > 0:
                        self.map_hailo_meta_to_tracklets(tracklets)

                    # For debug
                    if print_detections:
                        for tracklet in tracklets:
                            self.get_logger().debug("------------")
                            self.get_logger().debug("id: %d" % tracklet.id)
                            self.get_logger().debug("label: %s" % self.labelMap[tracklet.label])
                            roi = tracklet.roi.denormalize(640, 360)
                            #self.get_logger().debug("roi: %d,%d -- %d,%d" % (int(roi.topLeft().x), int(roi.topLeft().y), int(roi.bottomRight().x), int(roi.bottomRight().y)))
                            self.get_logger().info("spacial: %f, %f, %f" % (tracklet.spatialCoordinates.x, tracklet.spatialCoordinates.y, tracklet.spatialCoordinates.z))

                            #if tracklet.srcImgDetection.label != "NoneType":
                            #    label = labelMap[tracklet.srcImgDetection.label]
                            #else:
                            #    label = "none"
                            self.get_logger().debug("label %s, conf %f, xmin %f, ymin %f, xmax %f, ymax %f" % (self.labelMap[int(tracklet.srcImgDetection.label)], \
                                tracklet.srcImgDetection.confidence, tracklet.srcImgDetection.xmin, tracklet.srcImgDetection.ymin, \
                                tracklet.srcImgDetection.xmax, tracklet.srcImgDetection.ymax))
                            self.get_logger().debug(tracklet.status)

                    # Publish detections
                    self.publish_detections(self.labelMap, tracklets, "oakd_center_camera", self.objectPublisher)

                if inPreviewCAM == None:
                    continue

                if ball_detect and inEdgeFrame:
                    try:
                        edgeFrame = inEdgeFrame.getCvFrame()
                    except:
                        self.get_logger().error("Failed to read edge frame")
                    else:
                        if show_edge_image:
                            height = edgeFrame.shape[0]
                            width = edgeFrame.shape[1]

                            if inBallDetectNN != None:
                                # Display detections
                                for tracklet in inBallDetectNN.tracklets:
                                    # Denormalize bounding box
                                    x1 = int(tracklet.srcImgDetection.xmin*width)
                                    x2 = int(tracklet.srcImgDetection.xmax*width)
                                    y1 = int(tracklet.srcImgDetection.ymin*height)
                                    y2 = int(tracklet.srcImgDetection.ymax*height)

                                    conf = "{:.2f}".format(tracklet.srcImgDetection.confidence*100)
                                    self.get_logger().debug(f"id: {tracklet.id}, Conf {conf}, bb {x1},{y1} - {x2},{y2}, xyz {int(tracklet.spatialCoordinates.x)}, {int(tracklet.spatialCoordinates.y)}, {int(tracklet.spatialCoordinates.z)}")

                                    cv2.putText(edgeFrame, f"{int(tracklet.id)}", (x2 + 10, y1 + 0), cv2.FONT_HERSHEY_TRIPLEX, 0.50, 255)
                                    cv2.putText(edgeFrame, "{:.2f}".format(tracklet.srcImgDetection.confidence*100), (x2 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.25, 255)
                                    cv2.putText(edgeFrame, f"X: {int(tracklet.spatialCoordinates.x)}", (x2 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.25, 255)
                                    cv2.putText(edgeFrame, f"Y: {int(tracklet.spatialCoordinates.y)}", (x2 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.25, 255)
                                    cv2.putText(edgeFrame, f"Z: {int(tracklet.spatialCoordinates.z)}", (x2 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.25, 255)

                                    cv2.rectangle(edgeFrame, (x1, y1), (x2, y2), (255, 255, 0), 1)

                                self.publish_detections(["ball"], inBallDetectNN.tracklets, [], "oakd_right_camera", self.targetPublisher)

                            cv2.imshow("Edge", edgeFrame)
                            if pub_detection_frame:
                                self.detectionImagePub.publish(self.bridge.cv2_to_imgmsg(edgeFrame, "mono8"))

                try:
                    frameCAM = inPreviewCAM.getCvFrame()
                except:
                    self.get_logger().error("Failed to read preview frame")
                    continue

                # Pass image to Hailo pipeline without annotations in RGB format
                if hailo_ai_pipeline:
                    frameHailo = frameCAM.copy()
                    hailo_rgb = cv2.cvtColor(frameHailo, cv2.COLOR_BGR2RGB)
                    self.imagePubToHailo.publish(self.bridge.cv2_to_imgmsg(hailo_rgb, "rgb8"))

                # The Pose landmark NN input needs to be square.
                # Determine the xoffset on the left and right side of the image around
                # the square area.  This assumes a wider than taller image.
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

                # Blazepose post/host-side processing
                if human_pose and human_pose_process:
                    frameCAMOrig = frameCAM.copy()

                    # Get pose detection
                    inference = q_pd_out.tryGet()
                    if inference != None:
                        regions = blaze_pose.pd_postprocess(inference, frame_size_lm)
                        blaze_pose.pd_render(frameCAM, frame_size_lm, xoffset)

                        # Try to find a region (person) that corresponds to the
                        # the detected person reported by the Tracker node
                        sel_region = None
                        if self.tracked != None and self.tracked.tracking:
                            tracked_x = (self.tracked.object.bb_x_min + self.tracked.object.bb_x_max)/2
                            closest_diff = 0.0

                            for i,r in enumerate(regions):
                                # Calc normalized center of detected person
                                x = ((r.rect_points[0][1] + r.rect_points[3][1])/2 + xoffset)/widthCAM
                                diff = abs(x - tracked_x)
                                #self.get_logger().debug("region= %d, x center= %f, tracked_x= %f" % (i, x, tracked_x))
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
                            last_poses = None
                    else:
                        last_poses = None

                # For debug
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

                            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 1)

                    cv2.imshow("depth", depthFrameColor)

                # Image annotations
                if show_frame or pub_frame:

                    if last_region != None:
                        blaze_pose.lm_render(frameCAM, last_region)

                    if flipCAM:
                        frameCAM = cv2.flip(frameCAM, 1)

                    if human_pose and human_pose_process:
                        if last_poses != None:
                            lp = last_poses['left']
                            rp = last_poses['right']
                        else:
                            lp = rp = "none"
                        OverlayTextOnBox(frameCAM, 0, 0, 10, 10, [f"PoseL: {lp}", f"PoseR: {rp}"], (0, 0, 0), 0.4, font, 0.6, white, 1)

                    # Display detections
                    for tracklet in tracklets:
                        try:
                            label = self.labelMap[tracklet.label]
                        except:
                            label = tracklet.label
                        if label not in objects_to_track or tracklet.status != dai.Tracklet.TrackingStatus.TRACKED:
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

                        if self.tracked != None and self.tracked.object.id == tracklet.id:
                            box_color = yellow
                            box_thickness = 2
                        else:
                            box_color = white
                            box_thickness = 1

                        if show_det_info:
                            drawx = x1 + 2
                            drawy = y1 + 2        
                            if tracklet.id in self.tracklet_to_hailo_meta.keys():
                                face_id = self.hailo_person_list[self.tracklet_to_hailo_meta[tracklet.id]]['face_id']
                                drawy, _, _, drawy = OverlayTextOnBox(frameCAM, drawx, drawy, 5, 5, [face_id], (0, 0, 0), 0.7, font, font_scale*3, box_color, 3)

                            conf = "{:d}%".format(int(tracklet.srcImgDetection.confidence*100))
                            #conf = None
                            pos_x = f"x: {int(tracklet.spatialCoordinates.z)}"
                            pos_y = f"y: {int(tracklet.spatialCoordinates.x)*-1}"
                            pos_z = f"z: {int(tracklet.spatialCoordinates.y)}"
                            trk_id = f'id: {tracklet.id}'
                            OverlayTextOnBox(frameCAM, drawx, drawy, 5, 5, [conf, pos_x, pos_y, pos_z, trk_id], (0, 0, 0), 0.7, font, font_scale*1.5, white, 1)

                        cv2.rectangle(frameCAM, (x1, y1), (x2, y2), box_color, box_thickness)

                    # Draw annotations for Hailo meta data    
                    if show_hailo_meta_info and len(self.hailo_person_list) > 0:
                        for k, person in self.hailo_person_list.items():
                            if person['stale']: # or person['face_id'] == None:
                                continue

                            # If the face recog is stale, but we still know which person it
                            # belongs to, then draw face recog ID for person bb.  Otherwise
                            # draw a box around face and label it.
                            bb = None
                            if not person['face_recog_stale']:
                                bb = person['face_bb']
                            else:
                                bb = person['bb']
                            #print('det: %s' % value)

                            label = ''
                            if person['face_id'] != None:
                                label = person['face_id']
                            if person['reid_id'] != None:
                                label += ', ' + str(person['reid_id'])

                            # Denormalize bounding box
                            x1 = int(bb['xmin']*widthCAM)
                            x2 = int(bb['xmax']*widthCAM)
                            y1 = int(bb['ymin']*heightCAM)
                            y2 = int(bb['ymax']*heightCAM)

                            if flipCAM:
                                swap = x2
                                x2 = max(0, widthCAM - x1)
                                x1 = max(0, widthCAM - swap)

                            y1 = max(0, y1)
                            y2 = max(0, y2)

                            OverlayTextOnBox(frameCAM, x1 + 2, y1 + 2, 5, 5, [label], (0, 0, 0), 0.8, font, font_scale*3, white, 3)
                            cv2.rectangle(frameCAM, (x1, y1), (x2, y2), blue, 1)

                    OverlayTextOnBox(frameCAM, 0, frameCAM.shape[0] - 25, 5, 5, ["fps: {:.2f}".format(fps)], (0, 0, 0), 0.4, font, font_scale, white, 1)

                    # Normally image is shown by Viewer node when showing
                    # on robot display
                    if show_frame:
                        cv2.imshow("camera", frameCAM)
                        if self.setWinPos:
                            self.setWinPos = False
                            cv2.moveWindow("camera", 78, 30)

                    # Publish image to other nodes
                    if pub_frame:
                        if pub_compressed_image:
                            self.imagePub.publish(self.bridge.cv2_to_compressed_imgmsg(frameCAM, dst_format='jpg'))
                        else:                            
                            self.imagePub.publish(self.bridge.cv2_to_imgmsg(frameCAM, "bgr8"))
                        #im_rgb = cv2.cvtColor(frameCAM, cv2.COLOR_BGR2RGB)
                        #self.imagePub.publish(self.bridge.cv2_to_imgmsg(im_rgb, "rgb8"))

                    # Frame capture to file when commanded from another node
                    if self.save_image != None:
                        if self.save_image.filepath.find("#") != -1:
                            self.save_image.filepath = self.save_image.filepath.replace("#", str(save_cnt))
                            save_cnt += 1

                        cv2.imwrite(self.save_image.filepath, frameCAMOrig)
                        self.get_logger().debug("Saved frame to [%s]" % self.save_image.filepath)
                        self.save_image = None

                if use_imu and imuData != None:
                    imuPackets = imuData.packets
                    for imuPacket in imuPackets:
                        rVvalues = imuPacket.rotationVector
                        acceleroValues = imuPacket.acceleroMeter
                        self.publish_head_rotation(rVvalues, acceleroValues)

                if cv2.waitKey(1) == ord('q'):
                    break

    def create_ball_detector(self, pipeline, stereo):
        imageManip = pipeline.create(dai.node.ImageManip)
        imageManip.initialConfig.setResize(512, 320)
        imageManip.setKeepAspectRatio(False)
        # Its input
        stereo.rectifiedRight.link(imageManip.inputImage)

        edgeDetector = pipeline.create(dai.node.EdgeDetector)
        # Its input
        imageManip.out.link(edgeDetector.inputImage)

        xoutEdge = pipeline.create(dai.node.XLinkOut)
        xoutEdge.setStreamName("edge_image")
        edgeDetector.outputImage.link(xoutEdge.input)

        spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        spatialDetectionNetwork.setConfidenceThreshold(0.40)
        spatialDetectionNetwork.setBlobPath(get_model_path('ball_detect_yolo_v4_tiny_openvino_2021.4_5shave.blob'))
        spatialDetectionNetwork.setNumInferenceThreads(2)

        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)

        spatialDetectionNetwork.setNumClasses(1)
        spatialDetectionNetwork.setCoordinateSize(4)
        spatialDetectionNetwork.setAnchors(np.array([10,14, 23,27, 37,58, 81,82, 135,169, 344,319]))
        spatialDetectionNetwork.setAnchorMasks({ "side32": np.array([0,1,2]), "side16": np.array([3,4,5]) })
        spatialDetectionNetwork.setIouThreshold(0.3)
        # Its inputs
        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        edgeDetector.outputImage.link(spatialDetectionNetwork.input)

        # Create object tracker
        objectTracker = pipeline.createObjectTracker()
        # track only person
        # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS
        objectTracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)
        # Not in depthai 2.14.1.0
        #objectTracker.setTrackerIdAssigmentPolicy(dai.TrackerIdAssigmentPolicy.UNIQUE_ID)
        # Its input
        spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)
        spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
        spatialDetectionNetwork.out.link(objectTracker.inputDetections)

        nnBallDetOut = pipeline.create(dai.node.XLinkOut)
        nnBallDetOut.setStreamName("ball_detections")
        # Its input
        objectTracker.out.link(nnBallDetOut.input)
    
    def setup_imu(self, pipeline):
        imu = pipeline.create(dai.node.IMU)
        xlinkOut = pipeline.create(dai.node.XLinkOut)
        xlinkOut.setStreamName("imu")

        # enable ROTATION_VECTOR at 10 hz rate
        imu.enableIMUSensor([dai.IMUSensor.ROTATION_VECTOR, dai.IMUSensor.ACCELEROMETER_RAW], 10)
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)
        # Link plugins IMU -> XLINK
        imu.out.link(xlinkOut.input)

    def create_pipeline(self):
        # Start defining a pipeline
        pipeline = dai.Pipeline()
#        pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_2)
        pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_4)

        monoLeft = pipeline.createMonoCamera()
        monoRight = pipeline.createMonoCamera()
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoLeft.setFps(frame_rate);
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        monoLeft.setFps(frame_rate);

        # Stereo Depth
        stereo = pipeline.createStereoDepth()
        stereo.setConfidenceThreshold(255)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        #stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
        # Its inputs
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        # Color camera
        colorCam = pipeline.createColorCamera()
        #colorCam.setPreviewSize(640, 360)
        colorCam.setPreviewSize(1280, 720)

        colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        colorCam.setInterleaved(False)
        colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        colorCam.setFps(frame_rate)
        colorCam.setPreviewKeepAspectRatio(False)

        # Scale video from 1920x1080 to 640x360
        # (This was a workaround for an issue with depthai)
        #colorCam.setIspScale((1,3))

        manip_mn = pipeline.createImageManip()
        if use_tyolo_v4:
            manip_mn.initialConfig.setResize(416, 416)
        else:
            manip_mn.initialConfig.setResize(300, 300)
        #manip_mn.setKeepAspectRatio(True)
        manip_mn.setKeepAspectRatio(False)
        colorCam.preview.link(manip_mn.inputImage)

        # Spatial detection network
        if use_tyolo_v4:
            spatialDetectionNetwork = pipeline.createYoloSpatialDetectionNetwork()
            spatialDetectionNetwork.setBlobPath(get_model_path('tiny-yolo-v4_openvino_2021.2_6shave.blob'))
            spatialDetectionNetwork.setNumClasses(80)
            spatialDetectionNetwork.setCoordinateSize(4)
            spatialDetectionNetwork.setAnchors(np.array([10,14, 23,27, 37,58, 81,82, 135,169, 344,319]))
            spatialDetectionNetwork.setAnchorMasks({ "side26": np.array([1,2,3]), "side13": np.array([3,4,5]) })
            spatialDetectionNetwork.setIouThreshold(0.5)
            self.labelMap = labelMap_TYolo4
        else:
            spatialDetectionNetwork = pipeline.createMobileNetSpatialDetectionNetwork()
#            spatialDetectionNetwork.setBlobPath(get_model_path('mobilenet-ssd_openvino_2021.2_6shave.blob'))
            spatialDetectionNetwork.setBlobPath(get_model_path('mobilenet-ssd_openvino_2021.4_5shave.blob'))
            self.labelMap = labelMap_MNetSSD

        track_types = [self.labelMap.index(object) for object in objects_to_track]

        spatialDetectionNetwork.setNumInferenceThreads(1)
        spatialDetectionNetwork.setConfidenceThreshold(min_conf)
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
            objectTracker.setDetectionLabelsToTrack(track_types)
            # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS
            objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
            # Not in depthai 2.14.1.0
            #objectTracker.setTrackerIdAssigmentPolicy(dai.TrackerIdAssigmentPolicy.UNIQUE_ID)
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
        # Ball detector

        if ball_detect:
            self.create_ball_detector(pipeline, stereo)

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
            self.get_logger().debug("Creating Pose Detection Neural Network...")
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
            self.get_logger().debug("Creating Landmark Neural Network...")
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

        if use_imu:
            self.setup_imu(pipeline)

        self.get_logger().debug("Pipeline created.")
        return pipeline
    
    def read_hailo_meta(self):
        # Get list of face recognitions
        meta_face = self.hailo_meta_sink_face_recog.read(self.get_logger())

        # Get list of persons tracked with person re-id ID if available
        meta_person = self.hailo_meta_sink_person_reid.read(self.get_logger())

        # Group the face and person info based on the frame used to generate the data
        for frame in meta_face:
            if not frame['frame_idx'] in self.meta_sync_buffer.keys():
                self.meta_sync_buffer[frame['frame_idx']] = {'meta_person': None }
            self.meta_sync_buffer[frame['frame_idx']]['meta_face'] = frame['items']
            
        for frame in meta_person:
            if not frame['frame_idx'] in self.meta_sync_buffer.keys():
                self.meta_sync_buffer[frame['frame_idx']] = {'meta_face': None}
            self.meta_sync_buffer[frame['frame_idx']]['meta_person'] = frame['items']

        # Sort from oldest to newest since we want to process the most current info
        self.meta_sync_buffer = dict(sorted(self.meta_sync_buffer.items(),  reverse = True))
        buffer = self.meta_sync_buffer

        # Get the first complete set
        buffer_key = None
        for key in buffer.keys():
            if buffer[key]['meta_face'] != None and buffer[key]['meta_person'] != None:
                buffer_key = key
                break
            else:
                self.get_logger().debug('Buffer: %d, face= %s, person= %s' % (key, buffer[key]['meta_face'], buffer[key]['meta_person']))

        if buffer_key == None:
            self.get_logger().debug('Waiting for face and person data for same frame')
            # Only keep the meta data for the most recent frame
            self.meta_sync_buffer = dict(islice(self.meta_sync_buffer.items(), 1))
            return

        # We have a complete set to process...
        meta_face = buffer[key]['meta_face']
        meta_person = buffer[key]['meta_person']

        self.get_logger().info('meta_face cnt %d, meta_person cnt %d' % (len(meta_face), len(meta_person)))

        now = time.monotonic()

        # Associate face rec to person detection using linear assignment
        if len(meta_face) > 0 and len(meta_person) > 0:
            # Handle the case where there are more faces than person detections or vice versa.
            dim = max(len(meta_face), len(meta_person))
            cost = np.zeros([dim, dim])
                     
            for p in range(len(meta_person)):
                if 'reid_id' not in meta_person[p]:
                    meta_person[p]['reid_id'] = None

                self.get_logger().info('new person, tracker_id: %d, reid_id: %s' % (meta_person[p]['tracker_id'], str(meta_person[p]['reid_id'])))

                for f in range(len(meta_face)):
                    # Cost is how far away the face BB is from the ideal face location on the person BB
                    xmid_p = (meta_person[p]['bb']['xmin'] + meta_person[p]['bb']['xmax'])/2.0
                    xmid_f = (meta_face[f]['bb']['xmin'] + meta_face[f]['bb']['xmax'])/2.0

                    ytop_p = meta_person[p]['bb']['ymin']
                    ytop_f = meta_face[f]['bb']['ymin']

                    cost[p, f] = abs(xmid_p - xmid_f) + abs(ytop_p - ytop_f)

            #print('faces: %s' % str(meta_face))
            #print('persons: %s' % str(meta_person))
            #print('cost %s' % str(cost))

            _, col_ind = linear_sum_assignment(cost)
            #self.get_logger().info('col_ind %s' % str(col_ind))

            self.get_logger().info('Assignment of faces to persons:')
            for r in range(len(meta_person)):
                c_idx = col_ind[r]
                if c_idx >= len(meta_face):
                    continue

                person = meta_person[r]
                face = meta_face[c_idx]

                # Reject the assignment if not practical (face bb outside of person bb)
                if person['bb']['xmin'] <= face['bb']['xmin'] < face['bb']['xmax'] < person['bb']['xmax'] and \
                    person['bb']['ymin'] <= face['bb']['ymin'] < face['bb']['ymax'] < person['bb']['ymax']:

                    person['face_id'] = meta_face[c_idx]['id']
                    person['face_bb'] = meta_face[c_idx]['bb']
                
                    self.get_logger().debug('face %s:  person: tracker_id %d, xmin %f, xmax %f,   face bb: xmin %f, xmax %f ' % 
                        (person['face_id'], person['tracker_id'],
                        person['bb']['xmin'], person['bb']['xmax'],
                        person['face_bb']['xmin'], person['face_bb']['xmax']))

        # Aggregate the combined person/face ID with persisted person info...
        for person in meta_person:

            # Not typical
            if 'tracker_id' not in person:
                self.get_logger().info('tracker_id not in person')
                continue

            if 'reid_id' not in person:
                person['reid_id'] = None

            if 'face_id' not in person:
                person['face_id'] = None    

            if 'face_bb' not in person:
                person['face_bb'] = None    

            self.get_logger().debug('person, tracker_id: %s, reid_id: %s, face_id: %s' % (person['tracker_id'], person['reid_id'], person['face_id']))

            key = None
            key_reid = None
            key_tid = None

            # Try to match the current person with persisted person info.
            for k, value in self.hailo_person_list.items():
                if person['face_id'] != None and person['face_id'] == value['face_id']:
                    key = k
                    break

                if person['reid_id'] != None and person['reid_id'] == value['reid_id']:
                    key_reid = k

                if person['tracker_id'] == value['tracker_id']:
                    key_tid = k

            # Match first on face, then reid, then tracker
            if key == None:
                if key_reid != None:
                    key = key_reid
                elif key_tid != None:
                    key = key_tid

            # Add if not found            
            if key == None:
                # The list uses an ID that is independent of both the tracker ID and the re-id ID
                # since the tracker ID might change later and since the re-id ID may not be available initially
                key = str(self.next_person_id)
                self.next_person_id += 1
                self.hailo_person_list[key] = {'face_id': None, 'tracklet_id': None, 'reid_id': None, 'face_id': None, 'stale': True, 'face_recog_stale': False, 'face_last_update': 0.0}

            self.hailo_person_list[key]['bb'] = person['bb']
            self.hailo_person_list[key]['conf'] = person['conf']
            self.hailo_person_list[key]['tracker_id'] = person['tracker_id']

            if person['reid_id'] != None:
                # Remove use of this id for another person
                for _, p2 in self.hailo_person_list.items():
                    if p2['reid_id'] == person['reid_id']:
                        p2['reid_id'] = None
                self.hailo_person_list[key]['reid_id'] = person['reid_id']

            if person['face_id'] != None:
                # Remove use of this id for another person
                for _, p2 in self.hailo_person_list.items():
                    if p2['face_id'] == person['face_id']:
                        p2['face_id'] = None
                self.hailo_person_list[key]['face_id'] = person['face_id']
                self.hailo_person_list[key]['face_bb'] = person['bb']
                self.hailo_person_list[key]['face_last_update'] = now

            self.hailo_person_list[key]['last_update'] = now

        self.get_logger().info('face cnt %d, hailo person cnt %d' % (len(meta_face), len(self.hailo_person_list)))

        # Prune expired objects that don't have a face mapping since keeping
        # them around isn't useful
        to_remove = []
        for k, value in self.hailo_person_list.items():
            if value['face_id'] == None and (time.monotonic() - value['last_update'] > 2.0):
                to_remove.append(k)
                self.get_logger().info('Removing old person object, key %s' % k)
            else:
                value['stale'] = now - value['last_update'] > 1.0
                value['face_recog_stale'] = now - value['face_last_update'] > 0.5

        for key in to_remove:
            del self.hailo_person_list[key]                        

        self.get_logger().info('List after updating:')
        for k, value in self.hailo_person_list.items():
            self.get_logger().info('person, key: %s, tracker id: %s, reid id: %s, face id: %s, last_update: %f, %s'
                                    % (k, value['tracker_id'], value['reid_id'], 
                                    str(value['face_id']), value['last_update'],
                                    ('Stale' if value['stale'] else '')))
            
        # Drop current meta data and any pending
        self.meta_sync_buffer = {}
            
    # https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA['xmin'], boxB['xmin'])
        yA = max(boxA['ymin'], boxB['ymin'])
        xB = min(boxA['xmax'], boxB['xmax'])
        yB = min(boxA['ymax'], boxB['ymax'])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both rectangles
        boxAArea = (boxA['xmax'] - boxA['xmin'] + 1) * (boxA['ymax'] - boxA['ymin'] + 1)
        boxBArea = (boxB['xmax'] - boxB['xmax'] + 1) * (boxB['ymax'] - boxB['ymax'] + 1)

        return interArea / float(boxAArea + boxBArea - interArea)

    # Update mapping of Hailo person/face detection result to tracklet
    def map_hailo_meta_to_tracklets(self, tracklets):
    
        self.tracklet_to_hailo_meta = {};

        # Associate person to tracklet using linear assignment
        if len(self.hailo_person_list) > 0 and len(tracklets) > 0:
            # Handle the case where there are more persons than tracklets detections or vice versa.
            dim = max(len(self.hailo_person_list), len(tracklets))
            cost = np.zeros([dim, dim])

            p = 0
            for _, person in self.hailo_person_list.items(): 
                for t in range(len(tracklets)):
                    # Cost is IOU of the BBs
                    bb = {}
                    bb['xmin'] = tracklets[t].srcImgDetection.xmin
                    bb['ymin'] = tracklets[t].srcImgDetection.ymin
                    bb['xmax'] = tracklets[t].srcImgDetection.xmax
                    bb['ymax'] = tracklets[t].srcImgDetection.ymax
                    cost[p, t] = 1.0 - self.bb_intersection_over_union(person['bb'], bb)
                p += 1

            _, col_ind = linear_sum_assignment(cost)

            self.get_logger().info('Assignment of tracklets to persons:')
            r = -1
            for k, person in self.hailo_person_list.items():             
                r += 1
                c_idx = col_ind[r]

                if c_idx >= len(tracklets):
                    continue

                if cost[r, c_idx] < 0.8:
                    continue

                self.tracklet_to_hailo_meta[tracklets[c_idx].id] = k
                self.get_logger().info('Mapped tracklet id %d to face_id %s (%s)' % (tracklets[c_idx].id, self.hailo_person_list[k]['face_id'], k))
        
    # Publish detections to other ROS nodes
    # Uses a custom message.
    def publish_detections(self, labelMap, tracklets, camera, publisher):
        # Build a message containing the objects.  Uses
        # a custom message format
        objList = []

        self.get_logger().debug("Publishing detections for camera: %s" % camera)

        for tracklet in tracklets:
            try:
                label = labelMap[tracklet.label]
            except:
                label = tracklet.label

            # Hack to avoid false person detection of blank wall
            if (label == 'person' and (tracklet.srcImgDetection.xmax - tracklet.srcImgDetection.xmin) > 0.7):
                continue

            face_id = None
            if tracklet.id in self.tracklet_to_hailo_meta.keys():
                face_id = self.hailo_person_list[self.tracklet_to_hailo_meta[tracklet.id]]['face_id']

            desc = ObjectDesc()
            desc.id = tracklet.id
            desc.track_status = str(tracklet.status).split(".")[1]
            desc.name = str(label)
            desc.confidence = tracklet.srcImgDetection.confidence
            desc.position.header.frame_id = camera
            desc.position.header.stamp = self.get_clock().now().to_msg()
            # Map to ROS convention
            desc.position.point.x = tracklet.spatialCoordinates.z/1000.0
            desc.position.point.y = -1*tracklet.spatialCoordinates.x/1000.0
            desc.position.point.z = tracklet.spatialCoordinates.y/1000.0
            desc.bb_x_min = tracklet.srcImgDetection.xmin
            desc.bb_x_max = tracklet.srcImgDetection.xmax
            desc.bb_y_min = tracklet.srcImgDetection.ymin
            desc.bb_y_max = tracklet.srcImgDetection.ymax
            desc.unique_id = '' if face_id is None else face_id
            objList += (desc,)
            self.get_logger().info('Pub tracker: id: %s, unique_id: %s' % (desc.id, desc.unique_id))


        # Publish the object message to our topic
        msgObjects = ObjectDescArray()
        msgObjects.objects = objList
        publisher.publish(msgObjects)

    # Publish the interpreted pose from the Blazepose detection
    def publish_poses(self, poses):
        msg = DetectedPose()
        msg.detected = poses["detected"]
        msg.left = poses['left']
        msg.right = poses['right']
        msg.num_points = poses['num_points']
        self.posePublisher.publish(msg)

    # Subscribed-to topic for enabling/disabling pose detection
    def pose_detect_enable_callback(self, msg):
        global human_pose_process
        self.get_logger().info('Received EnablePoseDetect msg: mode: %d' % msg.enable)
        human_pose_process = msg.enable
        if msg.enable is False:
            last_region = None

    # Subscribed-to topic specifying which object is of interest
    # and should be annotated in the image.
    def tracked_callback(self, msg):
        self.tracked = msg
        self.get_logger().debug("tracked_callback, id= %d" % msg.object.id)

    # Subscribed-to topic for triggering an image capture
    def save_image_callback(self, msg):
        self.save_image = msg
        self.get_logger().debug("save_image_callback, fname= %s" % msg.filepath)

    # Publish the head (camera) orientation as measured by the IMU
    def publish_head_rotation(self, q, a):
        msg = HeadImu()
        msg.rotation.x = q.i
        msg.rotation.y = q.j
        msg.rotation.z = q.k
        msg.rotation.w = q.real
        msg.accelx = a.x
        msg.accely = a.y
        msg.accelz = a.z
        self.get_logger().debug('accely: %f' % a.y)        
        self.headImuPublisher.publish(msg)

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


