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

import os
import threading
import time
import math
import serial

import lewansoul_lx16a

import rclpy
from rclpy.node import Node
import rclpy.time

from std_msgs.msg import Bool
from std_msgs.msg import Int32
from std_msgs.msg import Float32

from robot_head_interfaces.msg import HeadTilt, HeadPose, TrackCmd, TrackStatus, ScanStatus, HeadImu
from speech_action_interfaces.msg import Wakeword

# Custom object detection messages
from object_detection_msgs.msg import ObjectDescArray

# Used for publishing the camera joint positions
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile

from sensor_msgs.msg import Joy

import tf2_ros
from geometry_msgs.msg import PoseStamped, Twist

SERIAL_PORT = '/dev/head_servo_driver'
servo_ctrl = None

# True to publish the position of the detected object to topic /goal_update to
# with Nav2 dynamic follower behavior tree.
use_tracked_pose_as_goal_update = False

JOY_MSG_AXIS_PAN = 2
JOY_MSG_AXIS_TILT = 3
joy_range_pan_deg = 120.0
joy_range_tilt_deg = 80.0

PAN_TRACK_GAIN = 50.0
TILT_TRACK_GAIN = 30.0

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
            serial.Serial(os.environ.get('HEAD_SERVO_SERIAL_DEVICE', SERIAL_PORT), 115200, timeout=1))

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

        self.servo_speed_factor = 1.0

        self.obj_ave = 0.0
        self.obj_last_dir = 0
        self.obj_last_pos = 0.0
        self.auto_center_time = 0.0
        self.target_pos = None
        self.move_steps = 1.0
        self.last_adj_by_voice = 0.0
        self.track_recenter_timeout = 3.0
        self.move_complete_time = time.monotonic()
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
        ms = int(abs((pos - self.servo_pos)*self.servo_ms_per_step*self.servo_speed_factor))
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
            self.servo_pos = int(pos)
            # Movement should be finished after this time
            self.move_complete_time = time.monotonic() + ms/1000.0
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
            #print("set_servo_relative_pos: %s, %f%%, %d %f deg" % (self.joint, pos_pct, self.servo_pos, self.get_servo_degrees()))

    def get_servo_degrees(self):
        deg = (self.servo_pos - self.servo_midpos)*self.servo_degrees_per_step
        #print("servo pos: %s, %d -> degrees: %f" % (self.joint, self.servo_pos, deg))
        return deg

    def get_servo_pos_from_degrees(self, deg):
        pos = deg/self.servo_degrees_per_step + self.servo_midpos
        return self.limit_pos(pos)

    def is_moving(self):
        return self.move_complete_time > time.monotonic()

    def auto_center(self):
        self.auto_center_time = 0.0
        self.target_pos = self.servo_midpos

    def stop_tracking(self):
        self.obj_last_dir = 0
        self.auto_center()

    # Update the pan-tilt base
    def update(self, obj):
        if obj != None:
            #print('Joint: %s, Object to track: %f, %f [%f, %f]' % (self.joint, obj.x, obj.y, obj.bb_x_min, obj.bb_x_max))

            if self.joint == "pan":
                pos = (obj.bb_x_min + obj.bb_x_max)/2
                self.obj_ave = self.obj_ave*0.3 + pos*0.7

                diff = 0.5 - self.obj_ave
                adj = diff * PAN_TRACK_GAIN
                if abs(adj) < 5.0:
                    adj = 0.0
            else:
                pos = obj.bb_y_min
                self.obj_ave = self.obj_ave*0.3 + pos*0.7

                diff = -1*(0.25 - self.obj_ave)
                adj = diff * TILT_TRACK_GAIN
                if abs(adj) < 3.0:
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
            TrackCmd,
            '/head/track_cmd',
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

        self.sub_joy = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            2)

        # Topic for publishing the tracker status
        self.pub_track_status = self.create_publisher(TrackStatus, '/head/track_status', QoSProfile(depth=1))

        # Topic for controlling robot base for turning the robot in the direction
        # of a detected person
        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 1)

        # Publisher for states of the pan-tilt joints
        qos_profile = QoSProfile(depth=10)
        self.pub_joint = self.create_publisher(JointState, 'joint_states', qos_profile)

        if use_tracked_pose_as_goal_update:
            self.pub_goal_update = self.create_publisher(PoseStamped, '/goal_update', qos_profile)

        self.pub_tracked_pose = self.create_publisher(PoseStamped, '/head/tracked_pose', qos_profile)

        self.tfBuffer = tf2_ros.Buffer()
        self.tflistener = tf2_ros.TransformListener(self.tfBuffer, self)

        self.track_cmd_mode = 'Track'           # Current track mode: None, Track, TrackScan, Scan, Manual, Joy
        self.track_new_mode = None              # New mode received from the topic
        self.track_sound_mode = 'None'          # Type of sound to trigger a scan: None, Any, Wakeword
        self.track_turn_base = False            # True if base should be turned to face tracked object
        self.track_object_type = 'person'       # Type of object to track
        self.track_object_unique_id = ''        # Unique ID of object to track (such as for a person via face recog)
        self.track_min_confidence = 0.7         # Minimum confidence to start tracking
        self.track_scan_delay_sec = 1.0         # Number seconds to wait before scanning after losing a track

        self.track_last_update = 0.0            # Time of last track update
        self.track_last_sound_scan = 0.0        # Time of last sound scan
        self.track_scanning_to_sound = False    # True if scanning in the direction of detected sound

        self.pose_cmd = None
        self.pan_manual_cmd = None
        self.tilt_manual_cmd = None
        self.rotate_manual_cmd = None

        self.last_tracked_object = None
        self.last_tracked_time = 0.0
        self.last_tracked_start_time = time.monotonic()

        self.last_logged_track_object = None

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
        self.detections_time = 0.0

        self.cur_pose = None
        self.cur_pose_valid = False;

        self.joy_msg = None
        self.joy_pan_ave = 0.0
        self.joy_tilt_ave = 0.0

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

        if detections != None:
            # For the list of objects detected and being tracked by the low-level tracker,
            # choose the best object to be reported as the tracked object.

            sorted_detections = sorted(detections, key=lambda x: x.position.point.x, reverse=False)

            for det in sorted_detections:
                #print("Det %d, %s, %f" % (det.id, det.track_status, det.position.point.x))

                if det.name != self.track_object_type:
                    continue

                # Is this detection being tracked?
                if self.last_tracked_object != None and \
                    self.last_tracked_object.id == det.id:
                    #print("Same id %d, %s" % (det.id, det.track_status))

                    if det.track_status == "TRACKED":
                        # Currently tracked object is still detected, update it
                        self.last_tracked_time = time.monotonic()

                        old_pos = self.last_tracked_object.position
                        
                        self.last_tracked_object = det
                        self.last_tracked_object.position.point.x = old_pos.point.x*0.7 + det.position.point.x*0.3
                        self.last_tracked_object.position.point.y = old_pos.point.y*0.7 + det.position.point.y*0.3
                        self.last_tracked_object.position.point.z = old_pos.point.z*0.7 + det.position.point.z*0.3

                        tracked_object = self.last_tracked_object

                    else:
                        tracked_object = None
                        continue

                # Not the last tracked object (if any), skip if this object not tracked
                elif det.track_status != "TRACKED":
                    continue

                self.get_logger().debug('track_object_unique_id: %s, unique_id: %s' % (self.track_object_unique_id, det.unique_id))

                # Always track the object having the specified unique_id
                if self.track_object_unique_id != '':
                    if det.unique_id != None and \
                        det.unique_id.lower() == self.track_object_unique_id and \
                        (self.last_tracked_object == None or det.id != self.last_tracked_object.id):

                        tracked_object = det
                        self.last_tracked_object  = det
                        self.last_tracked_start_time = time.monotonic()
                        self.get_logger().info('Now tracking object with unique_id: %s' % det.unique_id)
                        break

                # Select the closest object; either a new object if no last tracked
                # or a closer object if more than one object of the desired type is being tracked.
                # x is according to ROS conventions (pointing away from camera)
                elif (tracked_object == None or tracked_object.position.point.x > det.position.point.x) and \
                    det.confidence >= self.track_min_confidence:
                    tracked_object = det
                    self.last_tracked_start_time = time.monotonic()

        # At this point we have the determined the best tracked object to use (if any).
        # Replace the previous tracked object if it has timed-out, or just use the new one if no previous.

        # Delay a bit before switching away to a different object to give a little time for the currently tracked
        # object to become the best again.

        if self.last_tracked_object == None or \
            (time.monotonic() - self.last_tracked_time > 1.0):

            # Reset the start time of tracking/not tracking if
            # transitioning from not tracking to tracking or
            # vice versa (but not on each timeout while not
            # tracking)
            if self.last_tracked_object != None:
                self.last_tracked_start_time = time.monotonic()

            self.last_tracked_object = tracked_object
            self.last_tracked_time = time.monotonic()

        return tracked_object

    def tracker_thread(self):
        rot_cnt = 0
        while True:
            time.sleep(0.05)

            update_base_pose = False

            rot_cnt += 1
            if rot_cnt >= 2:
                rot_cnt = 0
                self.update_head_rotation()

            # Tracking modes:
            #   Scan - Continuously pan side-to-side
            #   TrackScan - Track, but scan if nothing tracked
            #   Track - Track if a person becomes in view, optionally turn to sound
            #   Manual - Only manual commands control head position
            #   LookDown - Center and look down
            #   Off - Disable tracking and go to home position

            if self.track_new_mode != None:
                if self.track_new_mode != self.track_cmd_mode:
                    self.cancel_scan()
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
                        self.sound_aoa = None
                        self.sound_wakeword = None
                        self.track_scanning_to_sound = False
                        self.servo_tilt.set_pos(self.servo_tilt.servo_midpos)
                        self.servo_pan.set_pos(self.servo_pan.servo_midpos)

                    elif self.track_new_mode == "TrackScan":
                        self.track_last_update = 0.0
                        self.servo_tilt.set_pos(self.servo_tilt.servo_midpos)

                    elif self.track_new_mode == "Off":
                        self.servo_tilt.set_pos(self.servo_tilt.servo_midpos)
                        self.servo_pan.set_pos(self.servo_pan.servo_midpos)

                    elif self.track_new_mode == "Manual":
                        self.manual_pan_ave = 0.0
                        self.manual_tilt_ave = 0.0
                        self.manual_rotate_ave = 0.0

                    elif self.track_new_mode == "Joy":
                        self.servo_tilt.set_pos(self.servo_tilt.servo_midpos)
                        self.servo_pan.set_pos(self.servo_pan.servo_midpos)
                        self.joy_pan_ave = 0.0
                        self.joy_tilt_ave = 0.0

                    self.track_cmd_mode = self.track_new_mode
                    
                self.track_new_mode = None

            # Process the current detections to determine who to track
            tracked_object = self.update_tracking(self.detections)

            if self.track_cmd_mode == "Manual":
                if self.pose_cmd != None:
                    self.manual_pan_ave = self.manual_pan_ave*0.9 + self.pose_cmd.yaw*0.1
                    self.manual_tilt_ave = self.manual_tilt_ave*0.7 + self.pose_cmd.pitch*0.3
                    self.manual_rotate_ave = self.manual_rotate_ave*0.7 + self.pose_cmd.roll*0.3

                    self.servo_pan.set_pos(self.servo_pan.get_servo_pos_from_degrees(self.manual_pan_ave))
                    self.servo_tilt.set_pos(self.servo_tilt.get_servo_pos_from_degrees(self.manual_tilt_ave))
                    self.servo_rotate.set_pos(self.servo_rotate.get_servo_pos_from_degrees(self.manual_rotate_ave))
                    #self.pose_cmd = None
                    
                if self.pan_manual_cmd != None:
                    self.servo_pan.set_servo_relative_pos(self.pan_manual_cmd)
                    self.pan_manual_cmd = None

                if self.tilt_manual_cmd != None:
                    self.servo_tilt.set_servo_relative_pos(self.tilt_manual_cmd)
                    self.tilt_manual_cmd = None

                if self.rotate_manual_cmd != None:
                    self.servo_rotate.set_servo_relative_pos(self.rotate_manual_cmd)
                    self.rotate_manual_cmd = None

            elif self.track_cmd_mode == "Joy":
                if self.joy_msg != None:
                    # Fix
                    self.servo_pan.servo_speed_factor = 1
                    self.servo_tilt.servo_speed_factor = 1

                    self.get_logger().info('servo pos: joy pan, %d, tilt, %d' % (self.joy_msg.axes[0], self.joy_msg.axes[1]))
                    pan = self.joy_msg.axes[2] * joy_range_pan_deg / 2.0
                    tilt = self.joy_msg.axes[3] * joy_range_tilt_deg / 2.0

                    self.joy_pan_ave = self.joy_pan_ave*0.9 + pan*0.1
                    self.joy_tilt_ave = self.joy_tilt_ave*0.7 + tilt*0.3

                    self.servo_pan.set_pos(self.servo_pan.get_servo_pos_from_degrees(self.joy_pan_ave))
                    self.servo_tilt.set_pos(self.servo_tilt.get_servo_pos_from_degrees(self.joy_tilt_ave))
                    self.get_logger().debug('servo pos: joy pan, %d, tilt, %d' % (pan, tilt))
                    self.joy_msg = None

            elif self.track_cmd_mode == "Scan":
                self.update_scan()

            elif self.track_cmd_mode == "Track":
                if tracked_object == None:
                    if self.track_scanning_to_sound:
                        self.update_scan()
                        self.track_last_sound_scan = time.monotonic()
                        if not self.is_scan_active():
                            self.track_scanning_to_sound = False

                    elif ((self.track_sound_mode == "Any" and self.sound_aoa != None and time.monotonic() - self.track_last_sound_scan > 4.0) or \
                        (self.track_sound_mode == "Wakeword" and self.sound_wakeword != None)):

                        #self.get_logger().info('tracked_object %d, voice_detect %d, sound_aoa= %s' % (tracked_object != None, self.track_sound_mode, str(self.sound_aoa)));

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
                        self.track_scanning_to_sound = True
                        self.servo_tilt.set_pos(self.servo_tilt.servo_midpos)
                    else:
                        # Update so servos auto center after tracking timeout
                        self.servo_pan.update(tracked_object)
                        self.servo_tilt.update(tracked_object)

                else:
                    self.track_scanning_to_sound = False
                    self.servo_pan.update(tracked_object)
                    self.servo_tilt.update(tracked_object)
                    update_base_pose = True

                self.sound_aoa = None
                self.sound_wakeword = None

            elif self.track_cmd_mode == "TrackScan":
                if tracked_object == None:
                    # If nothing being tracked, then scan until something is tracked
                    if time.monotonic() - self.track_last_update > self.track_scan_delay_sec:
                        if not self.is_scan_active():
                            self.init_scan(True, False, 0, 0)
                            self.servo_tilt.set_pos(self.servo_tilt.servo_midpos)
                        self.update_scan()
                    else:
                        self.cancel_scan()                        

                else:
                    self.cancel_scan()
                    self.servo_pan.update(tracked_object)
                    self.servo_tilt.update(tracked_object)
                    self.track_last_update = time.monotonic()
                    update_base_pose = True

            if self.track_turn_base:
                # Turn the base toward the tracked object if enabled
                if update_base_pose:
                    self.update_base_pose_tracking()
            else:
                self.stop_base_pose_tracking()

            if self.detections != None and time.monotonic() - self.detections_time > 1.0:
                self.detections = None
        
            self.publish_status()
            self.broadcast_camera_joints()
            self.log_track_status()

    def log_track_status(self):
        if self.last_tracked_object == None and self.last_logged_track_object != None or \
            self.last_tracked_object != None and self.last_logged_track_object == None  or \
            (self.last_tracked_object != None and self.last_logged_track_object != None and
            (self.last_tracked_object.id != self.last_logged_track_object.id or \
            self.last_tracked_object.unique_id != self.last_logged_track_object.unique_id)):
            
            self.last_logged_track_object = self.last_tracked_object

            if self.last_logged_track_object == None:
                self.get_logger().info('Not tracking')
            else:
                self.get_logger().info('Tracking: %s  %s' % (self.last_tracked_object.id, self.last_tracked_object.unique_id))


    def publish_status(self):
        msg = TrackStatus()
        msg.settings.mode = self.track_cmd_mode
        msg.settings.scan_step = str(self.scan_step)
        msg.settings.sound_mode = self.track_sound_mode
        msg.settings.turn_base = self.track_turn_base
        msg.settings.object_type = self.track_object_type
        msg.settings.object_unique_id = self.track_object_unique_id

        if self.last_tracked_object != None:
            msg.object = self.last_tracked_object
            msg.tracking = True
        else:
            msg.tracking = False

        # Duration tracked/not tracked state
        msg.duration = time.monotonic() - self.last_tracked_start_time

        msg.pan_angle = self.servo_pan.get_servo_degrees()
        msg.tilt_angle = self.servo_tilt.get_servo_degrees()
        msg.rotate_angle = self.servo_rotate.get_servo_degrees()
        msg.moving = self.servo_pan.is_moving() or self.servo_tilt.is_moving() or self.servo_rotate.is_moving()

        msg.scan_status = self.get_scan_status()
        self.pub_track_status.publish(msg)

        # Also publish the tracked object as a pose
        if self.last_tracked_object != None:
            msg = self.detection_to_point_stamped_msg(self.last_tracked_object)
            self.publish_tracked_as_pose(msg)

            # Publish the tracked object as a goal update (Nav follow)
            if use_tracked_pose_as_goal_update:
                self.publish_tracked_pose_as_goal_update(msg)


    def detection_to_point_stamped_msg(self, det):
        self.get_logger().debug('detection_to_point_stamped_msg: (%f, %f, %f)' % (det.position.point.x, det.position.point.y, det.position.point.z))
       
        msg = PoseStamped()
        msg.header = det.position.header
        msg.pose.position = det.position.point
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0
        return msg

    def publish_tracked_pose_as_goal_update(self, msg):
        self.pub_goal_update.publish(msg)
        self.get_logger().debug('publish_tracked_pose_as_goal_update, pt: (%f, %f)' % (msg.pose.position.x, msg.pose.position.y))

    # For rviz visualization
    def publish_tracked_as_pose(self, msg):
        self.pub_tracked_pose.publish(msg)

    # Broadcast the pan-tilt joints so ROS TF can be used to tranform positions
    # in the camera frame to other frames such as the map frame when navigating.
    # My URDF file names the camera joints as: 'cam_tilt_joint', 'cam_pan_joint
    def broadcast_camera_joints(self):
        cam_pan_deg = self.servo_pan.get_servo_degrees()
        cam_pan_rad = cam_pan_deg/180.0*PI

        cam_tilt_deg = -1.0*self.servo_tilt.get_servo_degrees()
        cam_tilt_rad = cam_tilt_deg/180.0*PI

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
        self.get_logger().debug('Head joint pos, pan: %f (%f), tilt: %f (%f)' % (cam_pan_rad, cam_pan_deg, cam_tilt_rad, cam_tilt_deg))

    # Move the head in a side-to-side scanning motion
    def init_scan(self, initial_scan_left, limit_num_scans, cnt_scans_left, cnt_scans_right):
        self.scan_left = initial_scan_left
        self.scan_limit_scans = limit_num_scans
        self.scan_stop_left_cnt = self.scan_at_left_count + cnt_scans_left
        self.scan_stop_right_cnt = self.scan_at_right_count + cnt_scans_right
        self.scan_active = True

    def cancel_scan(self):
        self.scan_active = False;
    
    def is_scan_active(self):
        return self.scan_active

    def update_scan(self):
        #self.get_logger().info('update_scan: scan_left %d, scan_at_left_count %d, scan_at_right_count %d, scan_stop_left_cnt %d, scan_stop_right_cnt %d, scan_limit_scans %d, scan_step %d' % \
        #    (self.scan_left, self.scan_at_left_count, self.scan_at_right_count, self.scan_stop_left_cnt, self.scan_stop_right_cnt, \
        #     self.scan_limit_scans, self.scan_step ))
        if self.scan_left:
            if self.servo_pan.servo_pos < self.servo_pan.servo_maxpos:
                self.servo_pan.set_pos(self.servo_pan.servo_pos + self.scan_step)
                self.scan_active = True
            else:
                self.scan_at_left_count += 1
                self.scan_left = False
        else:
            if self.servo_pan.servo_pos > self.servo_pan.servo_minpos:
                self.servo_pan.set_pos(self.servo_pan.servo_pos - self.scan_step)
                self.scan_active = True
            else:
                self.scan_at_right_count += 1
                self.scan_left = True

        if self.scan_limit_scans and \
            self.scan_stop_left_cnt <= self.scan_at_left_count and \
            self.scan_stop_right_cnt <= self.scan_at_right_count:
            # Done scanning, go back to center
            self.servo_pan.auto_center()
            self.scan_active = False

    def get_scan_status(self):
        scan_status = ScanStatus()
        scan_status.scanning = self.scan_active
        scan_status.angle = int(self.servo_pan.get_servo_degrees())
        scan_status.direction = 1 if self.scan_left else -1
        scan_status.at_left_count = self.scan_at_left_count
        scan_status.at_right_count = self.scan_at_right_count
        return scan_status

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
        self.get_logger().info('Received track cmd: mode: %s, scan_step: %s, sound_mode: %s, turn_base: %d, object_unique_id: %s, min_track_confidence: %f' % \
            (msg.mode, msg.scan_step, msg.sound_mode, msg.turn_base, msg.object_unique_id, msg.min_confidence))
        self.track_new_mode = msg.mode
        self.track_sound_mode = msg.sound_mode
        self.track_turn_base = msg.turn_base
        # Fix change scan_step to degrees
        self.scan_step = int(msg.scan_step)
        self.track_object_type = msg.object_type
        self.track_object_unique_id = msg.object_unique_id.lower()
        self.track_min_confidence = msg.min_confidence

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

    def joy_callback(self, msg):
        self.joy_msg = msg

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


