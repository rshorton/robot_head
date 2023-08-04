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

# Controls the camera/head tracking servos to keep a person in view.
# Also controls the emotional output features (head tilt, smile, talking indicator).

import threading
import time
import math
import serial

import lewansoul_lx16a

import rclpy
from rclpy.node import Node
import rclpy.time
from rclpy.duration import Duration

from std_msgs.msg import Bool
from std_msgs.msg import Int32
from std_msgs.msg import Float32

from robot_head_interfaces.msg import HeadTilt, HeadPose, Track, TrackStatus, ScanStatus, HeadImu
from speech_action_interfaces.msg import Wakeword

# Custom object detection messages
from object_detection_msgs.msg import ObjectDescArray

# Used for publishing the camera joint positions
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile

import tf2_ros
from geometry_msgs.msg import PoseStamped, Twist, PointStamped
from tf2_geometry_msgs import do_transform_point

min_track_confidence = 0.70

SERIAL_PORT = '/dev/head_servo_driver'
servo_ctrl = None

camera_frame = "oakd_center_camera"

# True to publish the position of the detected object to topic /goal_update to
# with Nav2 dynamic follower behavior tree.
use_tracked_pose_as_goal_update = False

PI = math.pi

# From: https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def init_servo_driver():
    global servo_ctrl
    if servo_ctrl is None:
        servo_ctrl = lewansoul_lx16a.ServoController(
            serial.Serial(SERIAL_PORT, 115200, timeout=1))

class CameraServo:
    def __init__(self, joint, logger):
        init_servo_driver()
        self.logger = logger
        self.joint = joint

        if joint == "pan":
            self.chan = 1
            # Min/max/mid positions determined by experiment
            self.servo_minpos = 122
            self.servo_midpos = 500
            self.servo_maxpos = 870

            self.servo_cal_min_servo = 122
            self.servo_cal_min_deg = -90
            self.servo_cal_max_servo = 870
            self.servo_cal_max_deg = 90
            self.servo_ms_per_degrees = 2000.0/180.0

        elif joint == "tilt":
            self.chan = 2
            self.servo_minpos = 0
            self.servo_midpos = 225
            # Fix - implement limits to avoid hitting shoulders
            # This limit will clear shoulders
            #self.servo_maxpos = 400
            self.servo_maxpos = 530
            self.servo_maxpos_mech = 600


            self.servo_cal_min_servo = 0
            self.servo_cal_min_deg = 60
            self.servo_cal_max_servo = 600
            self.servo_cal_max_deg = -90
            self.servo_ms_per_degrees = 3000.0/180.0

        elif joint == "rotate":
            self.chan = 3
            self.servo_minpos = 300
            self.servo_midpos = 500
            self.servo_maxpos = 700

            self.servo_cal_min_servo = 300
            self.servo_cal_min_deg = 51
            self.servo_cal_max_servo = 700
            self.servo_cal_max_deg = -51
            self.servo_ms_per_degrees = 3000.0/180.0

        else:
            print("Invalid camera joint: %s" % joint)
            return

        self.servo_degrees_per_step = (self.servo_cal_max_deg - self.servo_cal_min_deg)/(self.servo_cal_max_servo - self.servo_cal_min_servo)
        self.servo_ms_per_step = self.servo_degrees_per_step*self.servo_ms_per_degrees

        self.obj_ave = 0.0
        self.obj_last_dir = 0
        self.obj_last_pos = 0.0
        self.auto_center_time = 0.0
        self.target_pos = None
        self.move_steps = 1.0
        self.last_adj_by_voice = 0.0
        self.track_recenter_timeout = 3.0
        self.servo_pos = self.get_pos()
        ms = self.set_pos(self.servo_midpos)
        time.sleep(ms/1000.0)

    def is_at_servo_limit(self):
        return  self.servo_pos <= self.servo_minpos or self.servo_pos >= self.servo_maxpos

    def is_at_midpos(self):
        return  self.servo_pos == self.midpos

    def limit_pos(self, pos):
        pos = int(pos)
        if pos < self.servo_minpos:
            pos = self.servo_minpos
        elif pos > self.servo_maxpos:
            pos = self.servo_maxpos
        return pos

    def calc_move_duration(self, pos):
        #self.logger.info("calc_move_duration: %s, %f, %f ms/step" % (self.joint, pos - self.servo_pos, self.servo_ms_per_step))
        ms = int(abs((pos - self.servo_pos)*self.servo_ms_per_step))
        if ms < 50:
            ms = 0
        return ms

    def get_pos(self):
        return servo_ctrl.get_position(self.chan)

    def set_pos(self, pos_in):
        pos = self.limit_pos(pos_in)
        ms = 0
        if int(pos) != self.servo_pos:
            ms = self.calc_move_duration(pos)
            servo_ctrl.move(self.chan, pos, ms)
            #self.logger.info("set servo pos: %s, %d -> %d, %d ms" % (self.joint, self.servo_pos, pos, ms))
            self.servo_pos = int(pos_in)
        return ms

    def set_servo_relative_pos(self, pos_pct):
        # + means up:  percent of range from center to max
        # - means down: percent of range from center to min
        if pos_pct > 0.0:
            range = self.servo_maxpos - self.servo_midpos 
        else:
            range = self.servo_midpos - self.servo_minpos 

        pos = int(range*pos_pct/100.0) + self.servo_midpos 
        if pos != self.servo_pos:
            self.set_pos(pos)
            print("set_servo_relative_pos: %s, %f%%, %d %f deg" % (self.joint, pos_pct, self.servo_pos, self.get_servo_degrees()))

    def get_servo_degrees(self):
        deg = (self.servo_pos - self.servo_midpos)*self.servo_degrees_per_step
        #print("servo pos: %s, %d -> degrees: %f" % (self.joint, self.servo_pos, deg))
        return deg

    def get_servo_pos_from_degrees(self, deg):
        print("deg %f" % (deg))
        pos = deg/self.servo_degrees_per_step + self.servo_midpos
        print("pos %f" % (pos))
        return self.limit_pos(pos)

    def auto_center(self):
        self.auto_center_time = 0.0
        self.target_pos = self.servo_midpos

    def stop_tracking(self):
        self.obj_last_dir = 0
        self.auto_center()

    # Update the pan-tilt base
    def update(self, obj):
        if obj != None:
            #print('Joint: %s, Object to track: %f, %f [%f, %f]' % (self.joint, obj.x, obj.y, obj.x_min, obj.x_max))

            if self.joint == "pan":
                pos = (obj.x_min + obj.x_max)/2
                self.obj_ave = self.obj_ave*0.3 + pos*0.7

                diff = 0.5 - self.obj_ave
                adj = diff * 50.0
                if abs(adj) < 5.0:
                    adj = 0.0
            else:
                pos = obj.y_min
                self.obj_ave = self.obj_ave*0.3 + pos*0.7

                diff = -1*(0.25 - self.obj_ave)
                adj = diff * 40.0
                if abs(adj) < 5.0:
                    adj = 0.0

            self.obj_last_pos = pos

            # Debug/tuning
            #if self.joint == 'tilt':
            #    print('Joint: %s, ave= %f, pos_in= %f adj= %f, servo pos= %f' % \
            #          (self.joint, self.obj_ave, pos, adj, self.servo_pos + adj))

            self.set_pos(self.servo_pos + adj)
            self.obj_last_dir = adj

            self.target_pos = None
            self.auto_center_time = time.monotonic()
            self.last_adj_by_voice = 0.0

        elif self.target_pos != None:
            self.set_pos(self.target_pos)
            self.target_pos = None

        else:
            # No object visible now.
            # If was tracking on last update, then continue moving in that
            # direction to try and catch up.  If the limit is reached, then
            # return to center after a timeout. If wasn't tracking before,
            # then return to center after a timeout.
            if False:
            #if self.obj_last_dir != 0:
                self.set_pos(self.servo_pos + self.obj_last_dir*0.5)
                self.auto_center_time = time.monotonic()
                if self.is_at_servo_limit():
                    self.obj_last_dir = 0
            elif self.auto_center_time != 0.0 and time.monotonic() - self.auto_center_time > self.track_recenter_timeout:
                self.auto_center_time = 0.0
                self.obj_last_dir = 0
                self.set_pos(self.servo_midpos)

class CameraTracker(Node):
    def __init__(self):
        super().__init__('camera_tracker')

        init_servo_driver()

        # Topic for receiving track mode command
        self.sub_track = self.create_subscription(
            Track,
            '/head/track',
            self.track_callback,
            2)

        # Topic for receiving manual pan setting
        self.sub_manual_pan = self.create_subscription(
            Float32,
            '/head/manual_pan',
            self.manual_pan_callback,
            2)

        # Topic for receiving manual tilt setting
        self.sub_manual_tilt = self.create_subscription(
            Float32,
            '/head/manual_tilt',
            self.manual_tilt_callback,
            2)

        # Topic for receiving manual rotate setting
        self.sub_manual_rotate = self.create_subscription(
            Float32,
            '/head/manual_rotate',
            self.manual_rotate_callback,
            2)

        # Topic for receiving head twist command
        self.sub_head_rotation = self.create_subscription(
            HeadTilt,
            '/head/tilt',
            self.head_rotation_callback,
            2)

        # Topic for receiving head pose command
        self.sub_head_pose = self.create_subscription(
            HeadPose,
            '/head/pose',
            self.head_pose_callback,
            2)

        # Topic for receiving head rotation quaternion from IMU
        self.sub_head_rotation = self.create_subscription(
            HeadImu,
            '/head/imu',
            self.head_imu_callback,
            2)

        # Topic for receiving the list of detected objects from Vision node
        self.sub_obj_detection = self.create_subscription(
            ObjectDescArray,
            '/head/detected_objects',
            self.obj_detection_callback,
            2)

        # Topic for receiving Voice Activity detection messages from speech input node
        self.sub_speech_vad = self.create_subscription(
            Bool,
            '/speech_detect/vad',
            self.speech_vad_callback,
            2)

        # Topic for receiving sound angle-of-arrival messages from speech input node
        self.sub_speech_aoa = self.create_subscription(
            Int32,
            '/speech_detect/aoa',
            self.speech_aoa_callback,
            2)


        # Topic for receiving wakeword event msg messages from speech input node
        self.sub_speech_wakeword = self.create_subscription(
            Wakeword,
            '/speech_detect/wakeword',
            self.speech_wakeword_callback,
            2)

        # Topic for publishing which object is chosen for tracking
        self.pub_tracked = self.create_publisher(TrackStatus, '/head/tracked', 1)

        # Topic for controlling robot base for turning the robot in the direction
        # of a detected person
        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 1)

        # Publisher for states of the pan-tilt joints
        qos_profile = QoSProfile(depth=10)
        self.pub_joint = self.create_publisher(JointState, 'joint_states', qos_profile)

        self.pub_scan_status = self.create_publisher(ScanStatus, '/head/scan_status', qos_profile)

        if use_tracked_pose_as_goal_update:
            self.pub_goal_update = self.create_publisher(PoseStamped, '/goal_update', qos_profile)

        self.tfBuffer = tf2_ros.Buffer()
        self.tflistener = tf2_ros.TransformListener(self.tfBuffer, self)

        self.track_cmd_mode = "Track"
        self.track_rate = 0
        self.track_new_mode = None
        self.track_new_level = 0
        self.sound_track_mode = None
        self.track_turn_base = False
        self.track_object_type = 'person'

        self.pose_cmd = None
        self.pan_manual_cmd = None
        self.tilt_manual_cmd = None
        self.rotate_manual_cmd = None

        self.last_tracked_object = None
        self.last_tracked_ave_pos = None
        self.last_detected_time = None
        self.detected_time = time.monotonic()
        self.last_voice_track = 0

        self.track_base_track_vel = 0.0
        self.track_base_track_pan_ave = None

        self.head_rot_cmd_angle = None
        self.head_rot_steps = 0
        self.head_rot_dwell_ticks = 0

        self.head_imu = None
        self.cam_tilt_rad_imu_ave = None

        # Create an object for controlling each pan-tilt base joint
        self.servo_pan = CameraServo("pan", self.get_logger())
        self.servo_tilt = CameraServo("tilt", self.get_logger())
        self.servo_rotate = CameraServo("rotate", self.get_logger())
        
        self.tilt_scan_pos = self.servo_tilt.servo_midpos - 5
        self.scan_left = True
        self.scan_step = 4
        self.scan_at_left_count = 0
        self.scan_at_right_count = 0

        self.scan_limit_scans = False
        self.scan_stop_left_cnt = 0
        self.scan_stop_right_cnt = 0
        self.scan_active = False

        self.sound_aoa = None
        self.speech_detected = None
        self.sound_wakeword = None
        self.sound_wakeword_aoa = None

        self.detections = None
        self.detections_time = None

        self.cur_pose = None
        self.cur_pose_valid = False;

        self.thread = threading.Thread(target=self.tracker_thread)
        self.thread.start()

    def stop_base_pose_tracking(self):
        if self.track_base_track_vel != 0.0:
            self.track_base_track_vel = 0.0
            msg = Twist()
            msg.angular.z = self.track_base_track_vel
            self.pub_cmd_vel.publish(msg)

    def update_base_pose_tracking(self):
        pan = self.servo_pan.get_servo_degrees()
        if self.track_base_track_pan_ave == None:
            self.track_base_track_pan_ave = pan
        else:
            self.track_base_track_pan_ave = self.track_base_track_pan_ave*0.5 + pan*0.5

        pan = abs(pan)
        vel = 0.0
        # Turn if detected object more than 20deg off center.
        if pan > 20.0:
            vel = min(0.80, (pan - 20.0)*0.02 + 0.4)
            self.track_base_track_vel += (vel - self.track_base_track_vel)*0.3
            #self.get_logger().info("Base rot vel %f, pan= %f" % (vel, pan))

        if vel > 0.0:
            self.track_base_track_vel = math.copysign(vel, self.track_base_track_pan_ave)
        else:
            if self.track_base_track_vel == 0.0:
                return
            self.track_base_track_vel = 0.0

        msg = Twist()
        msg.angular.z = self.track_base_track_vel
        self.pub_cmd_vel.publish(msg)

    def update_tracking(self, detections):
        tracked_object = None
        publish = False

        if detections != None:
            for det in detections:
                if det.name != self.track_object_type or \
                    det.confidence < min_track_confidence or \
                    det.track_status != "TRACKED":
                    continue

                # Is this detection being tracked already?
                if self.last_tracked_object != None and \
                    self.last_tracked_object.id == det.id:
                    #print("Same id %d, %s" % (det.id, det.track_status))

                    if det.track_status == "TRACKED":
                        # Currently tracked object is still detected
                        tracked_object = self.last_tracked_object = det
                        self.last_detected_time = time.monotonic()
                        self.last_tracked_ave_pos[0] = self.last_tracked_ave_pos[0]*0.7 + det.x*0.3
                        self.last_tracked_ave_pos[1] = self.last_tracked_ave_pos[1]*0.7 + det.y*0.3
                        self.last_tracked_ave_pos[2] = self.last_tracked_ave_pos[2]*0.7 + det.z*0.3
                        publish = True
                    else:
                        tracked_object = None
                    break

                # Select the closest person
                # x is according to ROS conventions (pointing away from camera)
                if tracked_object == None or tracked_object.x > det.x:
                    tracked_object = det
                    self.last_tracked_ave_pos = [det.x, det.y, det.z]

        # Delay a bit before switching away to different person
        if self.last_tracked_object == None or \
            (time.monotonic() - self.last_detected_time > 1.0):

            # Reset the start time of tracking/not tracking if
            # transitioning from not tracking to tracking or
            # vice versa (but not on each timeout while not
            # tracking)
            if self.last_tracked_object != None:
                self.detected_time = time.monotonic()

            self.last_tracked_object = tracked_object
            self.last_detected_time = time.monotonic()
            publish = True
            #print("Now tracking: %s" % ("none" if tracked_object == None else tracked_object.id))

        if publish:
            if self.last_tracked_ave_pos != None:
                self.publish_tracked(tracked_object,
                    self.last_tracked_ave_pos[0],
                    self.last_tracked_ave_pos[1],
                    self.last_tracked_ave_pos[2])
        return tracked_object

    def tracker_thread(self):
        rot_cnt = 0
        track_cnt = 0
        while True:
            time.sleep(0.05)

            rot_cnt += 1
            if rot_cnt >= 2:
                rot_cnt = 0
                self.update_head_rotation()

            if self.track_new_mode != None:
                if self.track_new_mode != self.track_cmd_mode:
                    self.scan_active = False
                    self.servo_pan.stop_tracking()
                    self.servo_tilt.stop_tracking()
                    self.stop_base_pose_tracking()

                    if self.track_new_mode == "LookDown":
                        self.servo_tilt.set_pos(self.servo_tilt.servo_maxpos)
                        self.servo_pan.set_pos(self.servo_pan.servo_midpos)

                    elif self.track_new_mode == "Scan":
                        self.servo_tilt.set_pos(self.servo_tilt.servo_midpos)
                        self.init_scan(True, False, 0, 0)

                    elif self.track_new_mode == "Track":
                        self.servo_tilt.set_pos(self.servo_tilt.servo_midpos)

                    elif self.track_new_mode == "Off":
                        self.servo_tilt.set_pos(self.servo_tilt.servo_midpos)
                        self.servo_pan.set_pos(self.servo_pan.servo_midpos)

                    self.track_cmd_mode = self.track_new_mode
                    self.publish_scan_status()

                self.track_new_mode = None

            # Process the current detections to determine who to track
            tracked_object = self.update_tracking(self.detections)

            if self.track_cmd_mode == "Scan":
                self.update_scan()
                self.publish_scan_status()

            elif self.track_cmd_mode == "TrackScan":
                if tracked_object == None:
                    self.update_scan()
                    # Go back to track mode if scan completed without seeing a person
                    if self.scan_active == False:
                        self.track_cmd_mode = "Track"
                else:
                    self.track_cmd_mode = "Track"

            elif self.track_cmd_mode == "Manual":
                if self.pose_cmd != None:
                    self.servo_pan.set_pos(self.servo_pan.get_servo_pos_from_degrees(self.pose_cmd.yaw))
                    self.servo_tilt.set_pos(self.servo_tilt.get_servo_pos_from_degrees(self.pose_cmd.pitch))
                    self.servo_rotate.set_pos(self.servo_rotate.get_servo_pos_from_degrees(self.pose_cmd.roll))
                    self.get_logger().info('servo pos: pan, %d' % \
                        (self.servo_pan.servo_pos))
                    self.pose_cmd = None
                    
                if self.pan_manual_cmd != None:
                    self.servo_pan.set_servo_relative_pos(self.pan_manual_cmd)
                    self.pan_manual_cmd = None
                if self.tilt_manual_cmd != None:
                    self.servo_tilt.set_servo_relative_pos(self.tilt_manual_cmd)
                    self.tilt_manual_cmd = None
                if self.rotate_manual_cmd != None:
                    self.servo_rotate.set_servo_relative_pos(self.rotate_manual_cmd)
                    self.rotate_manual_cmd = None

            if self.track_cmd_mode == "Track":
                track_cnt += 1
                if track_cnt >= 1:
                    track_cnt = 0

                    #self.get_logger().info('tracked_object %d, voice_detect %d, sound_aoa= %s' % (tracked_object != None, self.sound_track_mode, str(self.sound_aoa)));

                    # If no object detect is detected but sound was detected, then
                    # scan in the direction of the sound.
                    if tracked_object == None and \
                        ((self.sound_track_mode == "any" and self.sound_aoa != None and time.monotonic() - self.last_voice_track > 5.0) or \
                         (self.sound_track_mode == "wakeword" and self.sound_wakeword != None)):

                        aoa = self.sound_wakeword_aoa if self.sound_wakeword != None else self.sound_aoa

                        # Mic AOA angle
                        #    225
                        # -45     135
                        #    45
                        # front of robot
                        #
                        self.get_logger().info('Starting sound-triggered scan, sound aoa %d' % (aoa))
                        scan_left = aoa > 45 and aoa < 225
                        left_cnt = 1 if scan_left else 0
                        right_cnt = 0 if scan_left else 1
                        self.init_scan(scan_left, True, left_cnt, right_cnt)
                        self.track_cmd_mode = "TrackScan"

                    else:
                        self.servo_pan.update(tracked_object)
                        self.servo_tilt.update(tracked_object)

                    self.sound_aoa = None
                    self.sound_wakeword = None
                    self.broadcast_camera_joints()

                if self.track_turn_base:
                    self.update_base_pose_tracking()
                else:
                    self.stop_base_pose_tracking()

            if self.detections != None and time.monotonic() - self.detections_time > 1.0:
                self.detections = None

            if tracked_object == None:
                self.stop_base_pose_tracking()
                self.last_detected_time == None

            self.broadcast_camera_joints()

    def publish_tracked_pose_as_goal_update(self, frame, x, y, z):
        p = PointStamped()     
        p.point.x = x/1000.0
        p.point.y = y/1000.0
        p.point.z = z/1000.0
        self.get_logger().debug('publish_tracked_pose_as_goal_update, pre-mapped: (%f, %f, %f)' % (p.point.x, p.point.y, p.point.z))

        if frame != camera_frame:
            try:
                transform = self.tfBuffer.lookup_transform(frame, camera_frame,
                                                           rclpy.time.Time(),
                                                           Duration(seconds=0.1))
            except Exception as e:
                self.get_logger().info(str(e))
                return

            xp = do_transform_point(p, transform)
            p.point.x = xp.point.x
            p.point.y = xp.point.y
       
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame
        msg.pose.position.x = p.point.x
        msg.pose.position.y = p.point.y
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0
        self.pub_goal_update.publish(msg)
        self.get_logger().debug('publish_tracked_pose_as_goal_update, pt: (%f, %f)' % (p.point.x, p.point.y))

    # Broadcast the pan-tilt joints so ROS TF can be used to tranform positions
    # in the camera frame to other frames such as the map frame when navigating.
    # My URDF file names the camera joints as: 'cam_tilt_joint', 'cam_pan_joint
    def broadcast_camera_joints(self):
        cam_pan_rad = self.servo_pan.get_servo_degrees()/180.0*PI
        cam_tilt_rad = -1.0*self.servo_tilt.get_servo_degrees()/180.0*PI

        #print("Joint states, pan,tilt: %f  %f" % (cam_pan_rad, cam_tilt_rad))

        # If IMU-based tilt measurement is available, then use it instead
        # of the servo position since it should be more accurate
        if self.head_imu != None:
            a = -1*math.atan2(self.head_imu.accelz, self.head_imu.accely)

            if self.cam_tilt_rad_imu_ave == None:
                self.cam_tilt_rad_imu_ave = a
            else:
                self.cam_tilt_rad_imu_ave = self.cam_tilt_rad_imu_ave*0.7 + a*0.3

            a_deg = self.cam_tilt_rad_imu_ave*180.0/math.pi

            self.get_logger().debug('IMU tilt angle: %f (%f, %f, %f)' % (a_deg, a, cam_tilt_rad, self.cam_tilt_rad_imu_ave))
            cam_tilt_rad = self.cam_tilt_rad_imu_ave

            #r,p,y = euler_from_quaternion(
            #        self.head_rotation_imu_quat.x,
            #        self.head_rotation_imu_quat.y,
            #        self.head_rotation_imu_quat.z,
            #        self.head_rotation_imu_quat.w)
            #p_d = p*180.0/math.pi
            #print(f"Using tilt from head rotation, Rad {p}, deg {p_d}")
            #cam_tilt_rad = p

        now = self.get_clock().now()
        joint_state = JointState()
        joint_state.header.stamp = now.to_msg()
        joint_state.name = ['cam_tilt_joint', 'cam_pan_joint']
        joint_state.position = [cam_tilt_rad, cam_pan_rad]
        self.pub_joint.publish(joint_state)
        self.get_logger().debug('Head joint pos, pan: %f, tilt: %f' % (cam_pan_rad, cam_pan_rad))


    # Move the head in a side-to-side scanning motion
    def init_scan(self, initial_scan_left, limit_num_scans, cnt_scans_left, cnt_scans_right):
        self.scan_left = initial_scan_left
        self.scan_limit_scans = limit_num_scans
        self.scan_stop_left_cnt = self.scan_at_left_count + cnt_scans_left
        self.scan_stop_right_cnt = self.scan_at_right_count + cnt_scans_right
        self.scan_active = True

    def update_scan(self):
        #self.get_logger().info('update_scan: scan_left %d, scan_at_left_count %d, scan_at_right_count %d, scan_stop_left_cnt %d, scan_stop_right_cnt %d, scan_limit_scans %d, scan_step %d' % \
        #    (self.scan_left, self.scan_at_left_count, self.scan_at_right_count, self.scan_stop_left_cnt, self.scan_stop_right_cnt, \
        #     self.scan_limit_scans, self.scan_step ))
        if self.scan_left:
            if self.servo_pan.servo_pos < self.servo_pan.servo_maxpos:
                self.servo_pan.set_pos(self.servo_pan.servo_pos + self.scan_step)
            else:
                self.scan_at_left_count += 1
                self.scan_left = False
        else:
            if self.servo_pan.servo_pos > self.servo_pan.servo_minpos:
                self.servo_pan.set_pos(self.servo_pan.servo_pos - self.scan_step)
            else:
                self.scan_at_right_count += 1
                self.scan_left = True

        if self.scan_limit_scans and \
            self.scan_stop_left_cnt <= self.scan_at_left_count and \
            self.scan_stop_right_cnt <= self.scan_at_right_count:
            # Done scanning, go back to center
            self.servo_pan.auto_center()
            self.last_voice_track = time.monotonic()
            self.scan_active = False

    def publish_tracked(self, detection, x_ave, y_ave, z_ave):
        msg = TrackStatus()
        msg.frame = "oakd"
        if detection != None:
            msg.object = detection
            msg.x_ave = x_ave
            msg.y_ave = y_ave
            msg.z_ave = z_ave
            msg.tracking = True
        else:
            msg.tracking = False
        # Duration tracked/not tracked
        msg.duration = time.monotonic() - self.detected_time
        self.pub_tracked.publish(msg)

        if detection != None and use_tracked_pose_as_goal_update:
            self.publish_tracked_pose_as_goal_update(camera_frame, x_ave, y_ave, z_ave)


    def publish_scan_status(self):
        scan_status = ScanStatus()
        scan_status.scanning = self.track_cmd_mode == "Scan"
        scan_status.angle = int(self.servo_pan.get_servo_degrees())
        scan_status.direction = 1 if self.scan_left else -1
        scan_status.at_left_count = self.scan_at_left_count
        scan_status.at_right_count = self.scan_at_right_count
        self.pub_scan_status.publish(scan_status)

    def head_rotation_callback(self, msg):
        self.get_logger().info('Received head tilt msg: angle: %s, transition_duration: %d, dwell_duration: %d' % \
            (msg.angle, msg.transition_duration, msg.dwell_duration))
        self.head_rot_cmd_angle = msg.angle
        self.head_rot_cmd_trans_dur = msg.transition_duration
        self.head_rot_cmd_dwell_dur = msg.dwell_duration

    def head_pose_callback(self, msg):
        self.get_logger().info('Received head pose msg: roll: %f, pitch: %f, yaw: %f' % \
            (msg.roll, msg.pitch, msg.yaw))
        self.track_new_mode = "Manual"
        self.pose_cmd = msg

    def update_head_rotation(self):
        if self.head_rot_cmd_angle != None:
            self.head_rot_steps = self.head_rot_cmd_angle/self.servo_rotate.servo_degrees_per_step
            self.head_rot_cmd_angle = None

            if self.head_rot_steps < 0:
                self.head_rot_dir = -1
            else:
                self.head_rot_dir = 1

            self.head_rot_steps *= self.head_rot_dir
            self.head_rot_dwell_ticks = int((self.head_rot_cmd_dwell_dur + 99)/100)

        if self.head_rot_steps > 0:
            self.servo_rotate.set_pos(self.servo_rotate.servo_pos + self.head_rot_dir*10)
            self.head_rot_steps -= 10

        elif self.head_rot_dwell_ticks > 0:
            self.head_rot_dwell_ticks -= 1

            if self.head_rot_dwell_ticks == 0:
                self.head_rot_steps = self.servo_rotate.servo_midpos - self.servo_rotate.servo_pos

                if self.head_rot_steps < 0:
                    self.head_rot_dir = -1
                else:
                    self.head_rot_dir = 1

                self.head_rot_steps *= self.head_rot_dir

    def track_callback(self, msg):
        self.get_logger().info('Received track msg: mode: %s, rate: %s, sound_track_mode: %s, turn_base: %d' % (msg.mode, msg.rate, msg.sound_track_mode, msg.turn_base))
        self.track_new_mode = msg.mode
        self.sound_track_mode = msg.sound_track_mode
        self.track_turn_base = msg.turn_base
        self.scan_step = int(msg.rate)
        self.track_object_type = msg.object_type

    # Command to set pan angle at the specified percent (+ right, - left)
    def manual_pan_callback(self, msg):
        self.get_logger().info('Received manual pan msg: pan amount: %f' % (msg.data))
        self.track_new_mode = "Manual"
        self.pan_manual_cmd = msg.data

    # Command to set tilt angle at the specified percent (+ up, - down)
    def manual_tilt_callback(self, msg):
        self.get_logger().info('Received manual tilt msg: tilt amount: %f' % (msg.data))
        self.track_new_mode = "Manual"
        self.tilt_manual_cmd = msg.data

    # Command to set rotate angle at the specified percent (+ cw, - ccw)
    def manual_rotate_callback(self, msg):
        self.get_logger().info('Received manual rotate msg: rotate amount: %f' % (msg.data))
        self.track_new_mode = "Manual"
        self.rotate_manual_cmd = msg.data

    def speech_aoa_callback(self, msg):
        self.get_logger().info('Received speech AOA msg: angle: %d' % msg.data)
        self.sound_aoa = msg.data

    def speech_vad_callback(self, msg):
        self.get_logger().info('Received speech VAD msg: detected: %d' % msg.data)
        self.speech_detected = msg.data

    def speech_wakeword_callback(self, msg):
        self.get_logger().info('Received speech wakeword msg: word: %s, angle: %d' % (msg.word, msg.angle))
        self.sound_wakeword = msg.word
        self.sound_wakeword_aoa = msg.angle

    def obj_detection_callback(self, msg):
        self.get_logger().debug('Received object detection msg')
        self.detections = msg.objects
        self.detections_time = time.monotonic()

    def pose_callback(self, msg):
        #self.get_logger().info('Received pose')
        self.cur_pose = msg.pose
        self.cur_pose_valid = True

    def head_imu_callback(self, msg):
        self.head_imu = msg

def main(args=None):
    rclpy.init(args=args)
    cam_tracker = CameraTracker()

    rclpy.spin(cam_tracker)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    cam_tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


