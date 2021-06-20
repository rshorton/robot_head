import threading
import sys
import time
import math

import os
import os.path
from os import path

import rclpy
from rclpy.node import Node
import rclpy.time
from rclpy.duration import Duration

from std_msgs.msg import String
from std_msgs.msg import Bool
from std_msgs.msg import Int32

from face_control_interfaces.msg import Smile, HeadTilt, Track, ScanStatus, Antenna

# Custom object detection messages
from object_detection_msgs.msg import ObjectDescArray
from object_detection_msgs.msg import ObjectDesc

# Used for publishing the camera joint positions
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile

from geometry_msgs.msg import PoseStamped, Twist
import tf2_ros
#from tf2_ros.transform_listener import TransformListener

# Servo control - uses Adafruit ServoKit to drive
# ADAFRUIT PCA9685 16-channel servo driver
import time
from adafruit_servokit import ServoKit

min_track_confidence = 0.70

PI = math.pi

smile_led_map =   [1, 2, 3, 8, 4, 5, 6, 7, 9]
smile_patterns = [[0, 0, 0, 1, 1, 1, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1]]

talk_patternsX =  [[0, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 1, 1, 1, 0, 0],
                  [0, 0, 0, 1, 1, 1, 0, 0, 0],
                  [0, 0, 1, 1, 1, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0, 0, 0, 0, 0]];

talk_patterns =  [[0, 1, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 1, 0, 1, 1, 0],
                  [0, 0, 0, 1, 0, 1, 0, 1, 0],
                  [0, 1, 1, 0, 0, 1, 1, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 1, 0, 0, 1, 0],
                  [0, 0, 1, 1, 0, 0, 1, 0, 0]];

antenna_left_ch = 10
antenna_right_ch = 11

servo_inited = False
servo_kit = None
pca = None

def init_servo_driver():
    global servo_inited
    global servo_kit
    global pca
    if not servo_inited:
        servo_kit = ServoKit(channels=16, frequency=50)
        servo_inited = True
        pca = servo_kit._pca

class CameraServo:
    def __init__(self, joint):
        init_servo_driver()

        self.joint = joint

        if joint == "pan":
            # Servo booard channel
            self.chan = 12

            # Min/max/mid positions determined by experiment
            self.servo_minpos = 10
            self.servo_maxpos_cal = 170
            self.servo_maxpos = 170
            self.servo_midpos = 87

            # Approx angle measurements for the above positions
            self.servo_degrees_per_step = (58.0 + 59.0)/(self.servo_maxpos_cal - self.servo_minpos)
            #self.servo_mid_degrees = 57.0

        elif joint == "tilt":
            self.chan = 13

            self.servo_minpos = 5
            self.servo_maxpos_cal = 142
            self.servo_maxpos = 80
            self.servo_midpos = 26

            self.servo_degrees_per_step = (15.0 + 90.0)/(self.servo_maxpos_cal - self.servo_minpos)
            #self.servo_mid_degrees = 14.0

        elif joint == "rotate":
            self.chan = 14

            self.servo_minpos = 0
            self.servo_maxpos = 150
            self.servo_midpos = 90

            self.servo_degrees_per_step = (40.0 + 65.0)/(self.servo_maxpos - self.servo_minpos)
            #self.servo_mid_degrees = 57.0

        else:
            print("Invalid camera joint: %s" % joint)
            return

        self.servo_pos = 0
        # Steps per position adjustment
        self.set_servo_pos(self.servo_midpos)

        self.obj_ave = 0.0
        self.obj_last_dir = 0
        self.obj_last_pos = 0.0
        self.auto_center_time = 0.0
        self.target_pos = None
        self.move_steps = 1.0
        self.last_adj_by_voice = 0.0

    def is_at_servo_limit(self):
        return  self.servo_pos <= self.servo_minpos or self.servo_pos >= self.servo_maxpos

    def is_at_midpos(self):
        return  self.servo_pos == self.midpos

    def set_servo_pos(self, pos_in):
        if pos_in < self.servo_minpos:
            pos_in = self.servo_minpos
        elif pos_in > self.servo_maxpos:
            pos_in = self.servo_maxpos

        pos = int(pos_in)

        global servo_kit
        servo_kit.servo[self.chan].angle = pos
        self.servo_pos = pos_in

        #if self.joint == 'tilt':
        #    print("Set servo pos %s %d %d" % (self.joint, pos, pos_in))

    def get_servo_degrees(self):
        deg = (self.servo_pos - self.servo_midpos)*self.servo_degrees_per_step
        #print("servo pos: %d -> degrees: %f" % (self.servo_pos, deg))
        return deg

    def get_servo_pos_from_degrees(self, deg):
        if deg < -90.0:
            deg = -90.0;
        elif deg > 90.0:
            deg = 90.0
        return deg/self.servo_degrees_per_step + self.servo_midpos

    def auto_center(self):
        self.auto_center_time = 0.0
        self.target_pos = self.servo_midpos
        #self.set_servo_pos(self.servo_midpos)

    def stop_tracking(self):
        self.obj_last_dir = 0
        self.auto_center()

    # Update the pan-tilt base
    def update(self, obj):
        if obj != None:
            #print('Joint: %s, Object to track: %f, %f [%f, %f]' % (self.joint, obj.x, obj.y, obj.x_min, obj.x_max))

            factor = 0.0

            if self.joint == "pan":
                pos = (obj.x_min + obj.x_max)/2
                self.obj_ave = self.obj_ave*0.7 + pos*0.3

                diff = 0.5 - self.obj_ave
                adj = diff * 20.0
                #if abs(adj) < 0.4:
                if abs(adj) < 1.0:
                    adj = 0.0
            else:
                pos = obj.y_min
                self.obj_ave = self.obj_ave*0.5 + pos*0.5

                diff = 0.25 - self.obj_ave
                adj = diff * 10.0
                if abs(adj) < 1.0:
                    adj = 0.0

            self.obj_last_pos = pos

            # Debug/tuning
            #if self.joint == 'tilt':
            #    print('Joint: %s, ave= %f, pos_in= %f adj= %f, servo pos= %f' % \
            #          (self.joint, self.obj_ave, pos, adj, self.servo_pos + adj))

            self.set_servo_pos(self.servo_pos + adj)
            self.obj_last_dir = adj

            self.target_pos = None
            self.auto_center_time = time.monotonic()
            self.last_adj_by_voice = 0.0

        elif self.target_pos != None:
            diff = self.target_pos - self.servo_pos
            if abs(diff) < self.move_steps:
                self.move_steps = abs(diff)
            self.set_servo_pos(self.servo_pos + math.copysign(self.move_steps, diff))

            #print("Moving %s %f" % (self.joint, self.servo_pos))

            if self.servo_pos == self.target_pos or \
                self.servo_pos <= self.servo_minpos or \
                self.servo_pos >= self.servo_maxpos:
                self.target_pos = None;

        else:
            # No object visible now.
            # If was tracking on last update, then continue moving in that
            # direction to try and catch up.  If the limit is reached, then
            # return to center after a timeout. If wasn't tracking before,
            # then return to center after a timeout.
            if False:
            #if self.obj_last_dir != 0:
                self.set_servo_pos(self.servo_pos + self.obj_last_dir*0.75)
                self.auto_center_time = time.monotonic()
                if self.is_at_servo_limit():
                    self.obj_last_dir = 0
            elif self.auto_center_time != 0.0 and time.monotonic() - self.auto_center_time > 6.0:
                self.auto_center_time = 0.0
                self.obj_last_dir = 0
                self.target_pos = self.servo_midpos
                self.move_steps = 3

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

class CameraTracker(Node):
    def __init__(self):
        super().__init__('camera_tracker')

        init_servo_driver()

        self.sub_smile = self.create_subscription(
            Smile,
            '/head/smile',
            self.smile_callback,
            2)

        self.sub_speaking = self.create_subscription(
            Bool,
            '/head/speaking',
            self.speaking_callback,
            2)

        self.sub_listening = self.create_subscription(
            Bool,
            '/speech_detect/listening',
            self.listening_callback,
            2)

        self.sub_track = self.create_subscription(
            Track,
            '/head/track',
            self.track_callback,
            2)

        self.sub_head_rotation = self.create_subscription(
            HeadTilt,
            '/head/tilt',
            self.head_rotation_callback,
            2)

        self.sub_antenna = self.create_subscription(
            Antenna,
            '/head/antenna',
            self.antenna_callback,
            2)

        self.sub_obj_detection = self.create_subscription(
            ObjectDescArray,
            '/head/detected_objects',
            self.obj_detection_callback,
            2)

        self.sub_speech_vad = self.create_subscription(
            Bool,
            '/speech_detect/vad',
            self.speech_vad_callback,
            2)

        self.sub_speech_aoa = self.create_subscription(
            Int32,
            '/speech_detect/aoa',
            self.speech_aoa_callback,
            2)

        #self.sub_pose = self.create_subscription(
        #    PoseStamped,
        #    '/robot_pose',
        #    self.pose_callback,
        #    1)

        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 1)

        # Publisher for states of the pan-tilt joints
        qos_profile = QoSProfile(depth=10)
        self.pub_joint = self.create_publisher(JointState, 'joint_states', qos_profile)

        self.pub_scan_status = self.create_publisher(ScanStatus, '/head/scan_status', qos_profile)

        #self.goal_pub = self.create_publisher(PoseStamped, '/goal_update', qos_profile)

        #self.tfBuffer = tf2_ros.Buffer()
        #self.tflistener = tf2_ros.TransformListener(self.tfBuffer, self)

        self.smile_mode_def = "smile"
        self.smile_level_def = 0

        self.smile_mode = self.smile_mode_def
        self.smile_level = self.smile_level_def
        self.smile_leds = smile_patterns[self.smile_level]

        self.smile_duration = 0
        self.smile_delta = 0
        self.smile_target_level = 0

        self.smile_cmd_mode = None
        self.smile_cmd_level = 0
        self.smile_cmd_duration_ms = 0

        self.set_smile()

        self.new_antenna = None
        self.antenna = None

        self.track_cmd_mode = "Track"
        self.track_rate = 0
        self.track_new_mode = None
        self.track_new_level = 0
        self.track_voice_detect = True
        self.track_turn_base = True
        self.last_voice_track = 0

        self.track_base_track_vel = 0.0
        self.track_base_track_pan_ave = None

        self.head_rot_cmd_angle = None
        self.head_rot_steps = 0
        self.head_rot_dwell_ticks = 0

        # Create an object for controlling each pan-tilt base joint
        self.servo_pan = CameraServo("pan")
        self.servo_tilt = CameraServo("tilt")
        self.servo_head_rotate = CameraServo("rotate")

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
        self.listening = False

        self.smile_timer = self.create_timer(0.1, self.smile_timer_callback)

        self.antenna_timer = self.create_timer(0.1, self.antenna_timer_callback)

        self.detections = None
        self.detections_time = None

        self.cur_pose = None
        self.cur_pose_valid = False;

        self.thread = threading.Thread(target=self.tracker_thread)
        self.thread.start()

    def stop_base_pose_tracking(self):
        if abs(self.track_base_track_vel) > 0.0:
            self.track_base_track_vel = 0.0
            msg = Twist()
            msg.angular.z = self.track_base_track_vel
            self.pub_cmd_vel.publish(msg)

    def update_base_pose_tracking(self):
        #print("Update base pose, state= %s" % self.track_base_track_state)

        pan = self.servo_pan.get_servo_degrees();
        if self.track_base_track_pan_ave == None:
            self.track_base_track_pan_ave = pan
        else:
            self.track_base_track_pan_ave = self.track_base_track_pan_ave*0.5 + pan*0.5

        if abs(self.servo_pan.get_servo_degrees()) > 20.0:
            self.track_base_track_vel = math.copysign(0.1, self.track_base_track_pan_ave)
        else:
            self.track_base_track_vel = 0.0

        msg = Twist()
        msg.angular.z = self.track_base_track_vel
        self.pub_cmd_vel.publish(msg)

    def tracker_thread(self):
        rot_cnt = 0
        track_cnt = 0
        while True:
            time.sleep(0.05)

            rot_cnt += 1
            if rot_cnt >= 2:
                rot_cnt = 0
                self.update_head_rotation()

            if self.track_new_mode != None and \
                self.track_new_mode != self.track_cmd_mode:

                self.scan_active = False
                self.servo_pan.stop_tracking()
                self.servo_tilt.stop_tracking()
                self.stop_base_pose_tracking()

                if self.track_new_mode == "LookDown":
                    self.servo_tilt.set_servo_pos(self.servo_tilt.servo_minpos)

                elif self.track_new_mode == "Scan":
                    self.servo_tilt.set_servo_pos(self.servo_tilt.servo_midpos)
                    self.init_scan(True, False, 0, 0)

                elif self.track_new_mode == "Track":
                    self.servo_tilt.set_servo_pos(self.servo_tilt.servo_midpos)

                self.track_cmd_mode = self.track_new_mode
                self.track_new_mode = None
                self.publish_scan_status()

            # Check for objects of interest
            closest_person = None
            if self.detections != None:
                cnt = len(self.detections)
                if cnt > 0:
                    for obj in self.detections:
                        if obj.name != 'person' or obj.confidence < min_track_confidence:
                            continue
                        if closest_person != None:
                            # x is according to ROS conventions (pointing away from camera)
                            if closest_person.x >  obj.x:
                                closest_person = obj
                        else:
                            closest_person = obj

            if self.track_cmd_mode == "Scan":
                self.update_scan()
                self.publish_scan_status()

            elif self.track_cmd_mode == "TrackScan":
                if closest_person == None:
                    self.update_scan()
                    # Go back to track mode if scan completed without seeing a person
                    if self.scan_active == False:
                        self.track_cmd_mode = "Track"
                else:
                    self.track_cmd_mode = "Track"

            if self.track_cmd_mode == "Track":
                track_cnt += 1
                if track_cnt >= 2:
                    track_cnt = 0

                    # If no object detect is detected but sound was detected, then
                    # scan in the direction of the sound.
                    if closest_person == None and \
                        self.track_voice_detect and \
                        self.sound_aoa != None and \
                        time.monotonic() - self.last_voice_track > 10.0:

                        # Mic AOA angle
                        #    225
                        # -45     135
                        #    45
                        # front of robot
                        #
                        scan_left = self.sound_aoa > 45 and self.sound_aoa < 225
                        left_cnt = 1 if scan_left else 0
                        right_cnt = 0 if scan_left else 1
                        self.init_scan(scan_left, True, left_cnt, right_cnt)
                        self.track_cmd_mode = "TrackScan"

                        self.last_voice_track = time.monotonic()

                    else:
                        self.servo_pan.update(closest_person)
                        self.servo_tilt.update(closest_person)

                    self.sound_aoa = None
                    self.broadcast_camera_joints()

                if self.track_turn_base:
                    self.update_base_pose_tracking()
                else:
                    self.stop_base_pose_tracking()

            if self.detections != None and time.monotonic() - self.detections_time > 1.0:
                self.detections = None

            if closest_person == None:
                self.stop_base_pose_tracking()

            self.broadcast_camera_joints()

    # Does not work since RCLPY does not support this yet (4/2021 Rolling release)
    # def publish_detected_pose(self, obj):
    #     try:
    #         self.tf_map_to_oakd = self.tfBuffer.lookup_transform("map", "oakd",
    #                                                              rclpy.time.Time(),
    #                                                              Duration(seconds=0.3))
    #     except Exception as e:
    #         self.get_logger().info(str(e))
    #         return
    #     else:
    #         self.get_logger().info('Got transform')
    #
    #     now = self.get_clock().now()
    #
    #     msg = PoseStamped()
    #     msg.header.stamp = now.to_msg()
    #     msg.header.frame_id = "oakd"
    #     msg.pose.position.x = obj.x;
    #     msg.pose.position.y = obj.y;
    #     msg.pose.position.z = obj.z;
    #     msg.pose.orientation.x = 0.0;
    #     msg.pose.orientation.y = 0.0;
    #     msg.pose.orientation.z = 0.0;
    #     msg.pose.orientation.w = 1.0;
    #
    #     map_pose = self.tfBuffer.transform(msg, "map")
    #     self.goal_pub.publish(map_pose)

    # Broadcast the pan-tilt joints so ROS TF can be used to tranform positions
    # in the camera frame to other frames such as the map frame when navigating.
    # My URDF file names the camera joints as: 'cam_tilt_joint', 'cam_pan_joint
    def broadcast_camera_joints(self):
        cam_pan_rad = self.servo_pan.get_servo_degrees()/180.0*PI
        cam_tilt_rad = -1.0*self.servo_tilt.get_servo_degrees()/180.0*PI

        #print("Joint states, pan,tilt: %f  %f" % (cam_pan_rad, cam_tilt_rad))

        now = self.get_clock().now()
        joint_state = JointState()
        joint_state.header.stamp = now.to_msg()
        joint_state.name = ['cam_tilt_joint', 'cam_pan_joint']
        joint_state.position = [cam_tilt_rad, cam_pan_rad]
        self.pub_joint.publish(joint_state)

    # Move the head in a side-to-side scanning motion
    def init_scan(self, initial_scan_left, limit_num_scans, cnt_scans_left, cnt_scans_right):
        self.scan_left = initial_scan_left
        self.scan_limit_scans = limit_num_scans
        self.scan_stop_left_cnt = self.scan_at_left_count + cnt_scans_left
        self.scan_stop_right_cnt = self.scan_at_right_count + cnt_scans_right
        self.scan_active = True

    def update_scan(self):
        if self.scan_left:
            if self.servo_pan.servo_pos < self.servo_pan.servo_maxpos:
                self.servo_pan.set_servo_pos(self.servo_pan.servo_pos + self.scan_step)
            else:
                self.scan_at_left_count += 1
                self.scan_left = False
        else:
            if self.servo_pan.servo_pos > self.servo_pan.servo_minpos:
                self.servo_pan.set_servo_pos(self.servo_pan.servo_pos - self.scan_step)
            else:
                self.scan_at_right_count += 1
                self.scan_left = True

        if self.scan_limit_scans and \
            self.scan_stop_left_cnt == self.scan_at_left_count and \
            self.scan_stop_right_cnt == self.scan_at_right_count:
            # Done scanning, go back to center
            self.servo_pan.auto_center()
            self.scan_active = False


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

    def update_head_rotation(self):
        if self.head_rot_cmd_angle != None:
            self.head_rot_steps = self.head_rot_cmd_angle/self.servo_head_rotate.servo_degrees_per_step
            self.head_rot_cmd_angle = None

            if self.head_rot_steps < 0:
                self.head_rot_dir = -1
            else:
                self.head_rot_dir = 1

            self.head_rot_steps *= self.head_rot_dir
            self.head_rot_dwell_ticks = int((self.head_rot_cmd_dwell_dur + 99)/100)

            #print("head_rot_steps= %d, dir= %d" % (self.head_rot_steps, self.head_rot_dir))

        if self.head_rot_steps > 0:
            self.servo_head_rotate.set_servo_pos(self.servo_head_rotate.servo_pos + self.head_rot_dir*10)
            self.head_rot_steps -= 10
            #print("head_rot_steps= %d, dir= %d, servo_pos %d" % (self.head_rot_steps, self.head_rot_dir, self.servo_head_rotate.servo_pos))

        elif self.head_rot_dwell_ticks > 0:
            self.head_rot_dwell_ticks -= 1
            #print("head_rot_dwell_ticks= %d" % (self.head_rot_dwell_ticks))

            if self.head_rot_dwell_ticks == 0:
                self.head_rot_steps = self.servo_head_rotate.servo_midpos - self.servo_head_rotate.servo_pos

                if self.head_rot_steps < 0:
                    self.head_rot_dir = -1
                else:
                    self.head_rot_dir = 1

                self.head_rot_steps *= self.head_rot_dir
                #print("head_rot_steps= %d, dir= %d" % (self.head_rot_steps, self.head_rot_dir))

    def smile_timer_callback(self):
        self.update_smile()

    def smile_callback(self, msg):
        self.get_logger().info('Received smile msg: mode: %s, level: %d, duration: %d, def: %s' % (msg.mode, msg.level, msg.duration_ms, msg.use_as_default))
        self.smile_cmd_mode = msg.mode
        self.smile_cmd_level = msg.level
        self.smile_cmd_duration_ms = msg.duration_ms
        if msg.use_as_default == True:
            self.smile_mode_def = msg.mode
            self.smile_level_def = msg.level

    def track_callback(self, msg):
        self.get_logger().info('Received track msg: mode: %s, rate: %s' % (msg.mode, msg.rate))
        self.track_new_mode = msg.mode
        self.track_voice_detect = msg.voice_detect
        self.tracK_turn_base = msg.turn_base
        self.scan_step = int(msg.rate)

    def set_smile(self):
        for i in range(0, len(self.smile_leds)):
            pca.channels[smile_led_map[i]].duty_cycle = 0 if self.smile_leds[i] else 65535

    def update_smile(self):
        if self.smile_cmd_mode != None:
            mode = self.smile_cmd_mode
            self.smile_cmd_mode = None

            print("New Smile cmd: %s" % mode)

            if mode != self.smile_mode:
                if mode == "default":
                    self.smile_mode = self.smile_mode_def
                    self.smile_level = self.smile_level_def
                    self.smile_leds = smile_patterns[self.smile_level]
                    self.set_smile()

                elif mode == "talking":
                    self.smile_mode = mode
                    self.smile_talk_index = 0
                    self.smile_leds = talk_patterns[self.smile_talk_index]
                    self.set_smile()

                elif mode == "smile":
                    self.smile_mode = mode
                    self.smile_level = self.smile_cmd_level
                    self.smile_leds = smile_patterns[self.smile_level]
                    self.smile_delta = 0
                    self.smile_duration = 0
                    self.set_smile()

            # Already smiling but a new level?
            elif mode == "smile" and \
                self.smile_level != self.smile_cmd_level and \
                self.smile_cmd_level >= 0 and self.smile_cmd_level < len(smile_patterns):

                self.smile_delta = -1 if self.smile_level > self.smile_cmd_level else 1
                self.smile_target_level = self.smile_cmd_level

                # Convert to timer updates
                self.smile_duration = int((self.smile_cmd_duration_ms + 99)/100)
                #print("smile_duration= %d" % self.smile_duration)

            return

        if self.smile_delta != 0:
            #print("smile_delta != 0, delta= %d" % self.smile_delta)

            self.smile_level = min(self.smile_level + self.smile_delta, len(smile_patterns) - 1)
            self.smile_leds = smile_patterns[self.smile_level]
            self.set_smile()

            if self.smile_level == self.smile_target_level:
                self.smile_delta = 0

        elif self.smile_duration > 0:
            #print("self.smile_duration > 0:, smile_duration= %d" % self.smile_duration)

            self.smile_duration -= 1
            if self.smile_duration == 0 and self.smile_level != self.smile_level_def:
                self.smile_delta = -1 if self.smile_level > self.smile_level_def else 1
                self.smile_target_level = self.smile_level_def
                #print("new smile_delta, delta= %d" % self.smile_delta)

        elif self.smile_mode == "talking":
            self.smile_talk_index += 1
            if self.smile_talk_index >= len(talk_patterns):
                self.smile_talk_index = 1

            #print("new self.smile_talk_index= %d" % self.smile_talk_index)
            self.smile_leds = talk_patterns[self.smile_talk_index]
            self.set_smile()

    def antenna_timer_callback(self):
        self.update_antenna()

    def antenna_callback(self, msg):
        self.get_logger().info('Received antenna msg, left: %s, right: %s, rate: %d, intensity: %d' %
                               (msg.left_blink_pattern, msg.right_blink_pattern, msg.rate, msg.intensity))
        self.new_antenna = msg

    def update_antenna(self):
        reset = False

        if self.listening:
            pca.channels[antenna_left_ch].duty_cycle = 10000;
            pca.channels[antenna_right_ch].duty_cycle = 10000;
            return

        if self.antenna == None:
            self.antenna = Antenna()
            self.antenna.left_blink_pattern =  '1010100000'
            self.antenna.right_blink_pattern = '0000010101'
            self.antenna.intensity = 50
            self.antenna.rate = 1
            reset = True

        elif self.new_antenna != None:
            self.antenna = self.new_antenna
            self.new_antenna = None
            reset = True

        if reset:
            self.antenna_left_idx = 0
            self.antenna_right_idx = 0
            self.antenna_update_cnt = 0
            self.antenna_left_state = None
            self.antenna_right_state = None

        self.antenna_update_cnt -= 1
        if self.antenna_update_cnt < 0:
            self.antenna_update_cnt = self.antenna.rate
            def set_antenna(side, pattern, idx, state):
                idx -= 1
                if idx < 0:
                    idx = len(pattern) - 1
                new_state = 65535 if pattern[idx] == '0' else self.antenna.intensity*500
                if new_state != state:
                    pca.channels[side].duty_cycle = new_state
                    state = new_state
                #print("Antenna updated: %s, idx= %u" % (side, idx))
                return idx, state

            self.antenna_left_idx, self.antenna_left_state = set_antenna(antenna_left_ch, self.antenna.left_blink_pattern, self.antenna_left_idx, self.antenna_left_state)
            self.antenna_right_idx, self.antenna_right_state = set_antenna(antenna_right_ch, self.antenna.right_blink_pattern, self.antenna_right_idx, self.antenna_right_state)

        #else:
        #    print("Antenna cnt: %u" % self.antenna_update_cnt)

    def listening_callback(self, msg):
        self.get_logger().info('Received listening active msg: speaking: %d' % msg.data)
        self.listening = msg.data

    def speaking_callback(self, msg):
        self.get_logger().info('Received speaking active msg: speaking: %d' % msg.data)
        if msg.data:
            self.smile_cmd_mode = "talking"
        # Go back to default smile when talking stops
        elif self.smile_mode == "talking":
            self.smile_cmd_mode = "default"
        else:
            return
        self.update_smile()

    def speech_aoa_callback(self, msg):
        self.get_logger().info('Received speech AOA msg: angle: %d' % msg.data)
        self.sound_aoa = msg.data

    def speech_vad_callback(self, msg):
        self.get_logger().info('Received speech VAD msg: detected: %d' % msg.data)
        self.speech_detected = msg.data

    def obj_detection_callback(self, msg):
        #self.get_logger().info('Received object detection msg')
        self.detections = msg.objects
        self.detections_time = time.monotonic()

    def pose_callback(self, msg):
        #self.get_logger().info('Received pose')
        self.cur_pose = msg.pose
        self.cur_pose_valid = True;

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


