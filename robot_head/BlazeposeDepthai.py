import numpy as np
from collections import namedtuple
import mediapipe_utils as mpu
import cv2
from pathlib import Path
import argparse
import os
import depthai as dai
from math import atan2
import time

# LINES_*_BODY are used when drawing the skeleton onto the source image.
# Each variable is a list of continuous lines.
# Each line is a list of keypoints as defined at https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
LINES_FULL_BODY = [[28,30,32,28,26,24,12,11,23,25,27,29,31,27],
                    [23,24],
                    [22,16,18,20,16,14,12],
                    [21,15,17,19,15,13,11],
                    [8,6,5,4,0,1,2,3,7],
                    [10,9],
                    ]
LINES_UPPER_BODY = [[12,11,23,24,12],
                    [22,16,18,20,16,14,12],
                    [21,15,17,19,15,13,11],
                    [8,6,5,4,0,1,2,3,7],
                    [10,9],
                    ]
# LINE_MESH_*_BODY are used when drawing the skeleton in 3D.
rgb = {"right":(0,1,0), "left":(1,0,0), "middle":(1,1,0)}
LINE_MESH_FULL_BODY = [ [9,10],[4,6],[1,3],
                        [12,14],[14,16],[16,20],[20,18],[18,16],
                        [12,11],[11,23],[23,24],[24,12],
                        [11,13],[13,15],[15,19],[19,17],[17,15],
                        [24,26],[26,28],[32,30],
                        [23,25],[25,27],[29,31]]
LINE_TEST = [ [12,11],[11,23],[23,24],[24,12]]

COLORS_FULL_BODY = ["middle","right","left",
                    "right","right","right","right","right",
                    "middle","middle","middle","middle",
                    "left","left","left","left","left",
                    "right","right","right","left","left","left"]
COLORS_FULL_BODY = [rgb[x] for x in COLORS_FULL_BODY]
LINE_MESH_UPPER_BODY = [[9,10],[4,6],[1,3],
                        [12,14],[14,16],[16,20],[20,18],[18,16],
                        [12,11],[11,23],[23,24],[24,12],
                        [11,13],[13,15],[15,19],[19,17],[17,15]
                        ]

# def to_planar(arr: np.ndarray, shape: tuple) -> list:
def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape)
    return resized.transpose(2,0,1)

class BlazeposeDepthai:
    def __init__(self,
                pd_score_thresh=0.5, pd_nms_thresh=0.3,
                lm_score_threshold=0.7,
                full_body=True,
                smoothing= True,
                filter_window_size=5,
                filter_velocity_scale=10,
                multi_detection=False):

        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.lm_score_threshold = lm_score_threshold
        self.full_body = full_body
        self.smoothing = smoothing
        self.multi_detection = multi_detection
        if self.multi_detection:
            print("With multi-detection, smoothing filter is disabled.")
            self.smoothing = False

        self.nb_kps = 33 if self.full_body else 25

        self.lm_input_length = 256
        self.pad_h = 0
        self.pad_w = 0

        if self.smoothing:
            self.filter = mpu.LandmarksSmoothingFilter(filter_window_size, filter_velocity_scale, (self.nb_kps, 3))

        # Create SSD anchors
        # https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt
        anchor_options = mpu.SSDAnchorOptions(num_layers=4,
                                min_scale=0.1484375,
                                max_scale=0.75,
                                input_size_height=128,
                                input_size_width=128,
                                anchor_offset_x=0.5,
                                anchor_offset_y=0.5,
                                strides=[8, 16, 16, 16],
                                aspect_ratios= [1.0],
                                reduce_boxes_in_lowest_layer=False,
                                interpolated_scale_aspect_ratio=1.0,
                                fixed_anchor_size=True)
        self.anchors = mpu.generate_anchors(anchor_options)
        self.nb_anchors = self.anchors.shape[0]
        print(f"{self.nb_anchors} anchors have been created")

        # Rendering flags
        self.show_pd_box = False
        self.show_pd_kps = False
        self.show_rot_rect = False
        self.show_landmarks = True
        self.show_scores = False

    def pd_postprocess(self, inference, frame_size_lm):
        scores = np.array(inference.getLayerFp16("classificators"), dtype=np.float16) # 896
        bboxes = np.array(inference.getLayerFp16("regressors"), dtype=np.float16).reshape((self.nb_anchors,12)) # 896x12

        # Decode bboxes
        self.regions = mpu.decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors, best_only=not self.multi_detection)
        # Non maximum suppression (not needed if best_only is True)
        if self.multi_detection:
            self.regions = mpu.non_max_suppression(self.regions, self.pd_nms_thresh)

        mpu.detections_to_rect(self.regions, kp_pair=[0,1] if self.full_body else [2,3])
        mpu.rect_transformation(self.regions, frame_size_lm, frame_size_lm)

        return self.regions

    def pd_render(self, frame, frame_size_lm, xoffset):
        for r in self.regions:
            if self.show_pd_box:
                box = (np.array(r.pd_box) * frame_size_lm).astype(int)
                cv2.rectangle(frame, (box[0]+xoffset, box[1]), (box[0]+box[2]+xoffset, box[1]+box[3]), (0,255,0), 2)
            if self.show_pd_kps:
                # Key point 0 - mid hip center
                # Key point 1 - point that encodes size & rotation (for full body)
                # Key point 2 - mid shoulder center
                # Key point 3 - point that encodes size & rotation (for upper body)
                if self.full_body:
                    # Only kp 0 and 1 used
                    list_kps = [0, 1]
                else:
                    # Only kp 2 and 3 used for upper body
                    list_kps = [2, 3]
                for kp in list_kps:
                    x = int(r.pd_kps[kp][0] * frame_size_lm)
                    y = int(r.pd_kps[kp][1] * frame_size_lm)
                    cv2.circle(frame, (x, y), 3, (0,0,255), -1)
                    cv2.putText(frame, str(kp), (x, y+12), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
            if self.show_scores:
                cv2.putText(frame, f"Pose score: {r.pd_score:.2f}",
                        (int(r.pd_box[0] * frame_size_lm+10), int((r.pd_box[1]+r.pd_box[3])*frame_size_lm+60)),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)


    def lm_postprocess(self, region, inference, xoffset):
        region.lm_score = inference.getLayerFp16("output_poseflag")[0]
        if region.lm_score > self.lm_score_threshold:
            self.nb_active_regions += 1

            lm_raw = np.array(inference.getLayerFp16("ld_3d")).reshape(-1,5)
            # Each keypoint have 5 information:
            # - X,Y coordinates are local to the region of
            # interest and range from [0.0, 255.0].
            # - Z coordinate is measured in "image pixels" like
            # the X and Y coordinates and represents the
            # distance relative to the plane of the subject's
            # hips, which is the origin of the Z axis. Negative
            # values are between the hips and the camera;
            # positive values are behind the hips. Z coordinate
            # scale is similar with X, Y scales but has different
            # nature as obtained not via human annotation, by
            # fitting synthetic data (GHUM model) to the 2D
            # annotation.
            # - Visibility, after user-applied sigmoid denotes the
            # probability that a keypoint is located within the
            # frame and not occluded by another bigger body
            # part or another object.
            # - Presence, after user-applied sigmoid denotes the
            # probability that a keypoint is located within the
            # frame.

            # Normalize x,y,z. Scaling in z = scaling in x = 1/self.lm_input_length
            lm_raw[:,:3] /= self.lm_input_length
            # Apply sigmoid on visibility and presence (if used later)
            # lm_raw[:,3:5] = 1 / (1 + np.exp(-lm_raw[:,3:5]))

            # region.landmarks contains the landmarks normalized 3D coordinates in the relative oriented body bounding box
            region.landmarks = lm_raw[:,:3]
            # Calculate the landmark coordinate in square padded image (region.landmarks_padded)
            src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
            dst = np.array([ (x, y) for x,y in region.rect_points[1:]], dtype=np.float32) # region.rect_points[0] is left bottom point and points going clockwise!
            mat = cv2.getAffineTransform(src, dst)
            lm_xy = np.expand_dims(region.landmarks[:self.nb_kps,:2], axis=0)
            lm_xy = np.squeeze(cv2.transform(lm_xy, mat))
            # A segment of length 1 in the coordinates system of body bounding box takes region.rect_w_a pixels in the
            # original image. Then we arbitrarily divide by 4 for a more realistic appearance.
            lm_z = region.landmarks[:self.nb_kps,2:3] * region.rect_w_a / 4
            lm_xyz = np.hstack((lm_xy, lm_z))
            if self.smoothing:
                lm_xyz = self.filter.apply(lm_xyz)
            region.landmarks_padded = lm_xyz.astype(np.int)

            # Adjust for case when frame is non-square and landmarks are detected on cropped square region
            # of full image
            region.landmarks_padded[:,0] += xoffset

            # If we added padding to make the image square, we need to remove this padding from landmark coordinates
            # region.landmarks_abs contains absolute landmark coordinates in the original image (padding removed))
            region.landmarks_abs = region.landmarks_padded.copy()
            if self.pad_h > 0:
                region.landmarks_abs[:,1] -= self.pad_h
            if self.pad_w > 0:
                region.landmarks_abs[:,0] -= self.pad_w

            region.landmarks_abs[:,0] += xoffset

    def lm_render(self, frame, region):
        if region.lm_score > self.lm_score_threshold:
            if self.show_rot_rect:
                cv2.polylines(frame, [np.array(region.rect_points)], True, (0,255,255), 2, cv2.LINE_AA)
            if self.show_landmarks:

                list_connections = LINES_FULL_BODY if self.full_body else LINES_UPPER_BODY
                lines = [np.array([region.landmarks_padded[point,:2] for point in line]) for line in list_connections]
                cv2.polylines(frame, lines, False, (255, 180, 90), 4, cv2.LINE_AA)

                for i,x_y in enumerate(region.landmarks_padded[:,:2]):
                    if i > 10:
                        color = (0,255,0) if i%2==0 else (0,0,255)
                    elif i == 0:
                        color = (0,255,255)
                    elif i in [4,5,6,8,10]:
                        color = (0,255,0)
                    else:
                        color = (0,0,255)
                    cv2.circle(frame, (x_y[0], x_y[1]), 4, color, -11)

            if self.show_scores:
                cv2.putText(frame, f"Landmark score: {region.lm_score:.2f}",
                        (int(region.pd_box[0] * self.frame_size+10), int((region.pd_box[1]+region.pd_box[3])*self.frame_size+90)),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)


    def check_filter_reset(self):
        if self.smoothing and self.nb_active_regions == 0:
            self.filter.reset()

