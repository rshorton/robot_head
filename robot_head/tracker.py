import threading
import sys
import time

import os
import os.path
from os import path
from math import copysign

import rclpy
from rclpy.node import Node
import rclpy.time
from rclpy.duration import Duration

from std_msgs.msg import String
from std_msgs.msg import Bool
from std_msgs.msg import Int32

from face_control_interfaces.msg import Smile, HeadTilt, Track, ScanStatus

# Custom object detection messages
from object_detection_msgs.msg import ObjectDescArray
from object_detection_msgs.msg import ObjectDesc

# Used for publishing the camera joint positions
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile

#from geometry_msgs.msg import PoseStamped
#import tf2_ros
#from tf2_ros.transform_listener import TransformListener

# Servo control - uses Adafruit ServoKit to drive
# ADAFRUIT PCA9685 16-channel servo driver
import time
from adafruit_servokit import ServoKit

min_track_confidence = 0.70

PI = 3.14159

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

            self.servo_step = 0.75

            # Approx angle measurements for the above positions
            self.servo_degrees_per_step = (58.0 + 59.0)/(self.servo_maxpos_cal - self.servo_minpos)
            #self.servo_mid_degrees = 57.0

        elif joint == "tilt":
            self.chan = 13

            self.servo_minpos = 5
            self.servo_maxpos_cal = 142
            self.servo_maxpos = 80
            self.servo_midpos = 26

            self.servo_step = 0.75

            self.servo_degrees_per_step = (15.0 + 90.0)/(self.servo_maxpos_cal - self.servo_minpos)
            #self.servo_mid_degrees = 14.0

        elif joint == "rotate":
            self.chan = 14

            self.servo_minpos = 0
            self.servo_maxpos = 150
            self.servo_midpos = 90

            self.servo_step = 0.75

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
        return  self.servo_pos <= self.servo_minpos + self.servo_step or self.servo_pos >= self.servo_maxpos - self.servo_step

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

    def stop_tracking(self):
        self.obj_last_dir = 0
        self.auto_center_time = 0.0
        self.set_servo_pos(self.servo_midpos)

    # Update the pan-tilt base
    def update(self, obj, voice_angle):
        if obj != None:
            #print('Joint: %s, Object to track: %f, %f [%f, %f]' % (self.joint, obj.x, obj.y, obj.x_min, obj.x_max))

            factor = 0.0

            if self.joint == "pan":
                pos = (obj.x_min + obj.x_max)/2
                self.obj_ave = self.obj_ave*0.7 + pos*0.3

                # Try to object in center of left-right view
                if self.obj_ave > 0.9:
                    factor = -8.0
                elif self.obj_ave > 0.8:
                    factor = -6.0
                elif self.obj_ave > 0.7:
                    factor = -5.0
                elif self.obj_ave > 0.6:
                    factor = -1.0
                elif self.obj_ave < 0.1:
                    factor = 8.0
                elif self.obj_ave < 0.2:
                    factor = 6.0
                elif self.obj_ave < 0.3:
                    factor = 5.0
                elif self.obj_ave < 0.4:
                    factor = 1.0
                factor *= -1.0
            else:
                pos = obj.y_min
                self.obj_ave = self.obj_ave*0.2 + pos*0.8

                # Try to keep top of object (person) in view
                if self.obj_ave > 0.6:
                    factor = 2.0
                elif self.obj_ave > 0.4:
                    factor = 1.0
                elif self.obj_ave > 0.3:
                    factor = 1.0
                elif self.obj_ave < 0.1:
                    factor = -1.0
                elif self.obj_ave < 0.2:
                    factor = -1.0

            self.obj_last_pos = pos

            if self.joint == 'pan':
                print('Joint: %s, ave= %f, pos_in= %f factor= %f add= %f, servo= %f' % \
                      (self.joint, self.obj_ave, pos, factor, factor*self.servo_step, self.servo_pos - factor*self.servo_step))

            self.set_servo_pos(self.servo_pos - factor*self.servo_step)
            self.obj_last_dir = -1*factor

            self.target_pos = None
            self.auto_center_time = time.monotonic()
            self.last_adj_by_voice = 0.0

        elif self.target_pos != None:
            diff = self.target_pos - self.servo_pos
            if abs(diff) < self.move_steps:
                self.move_steps = abs(diff)
            self.set_servo_pos(self.servo_pos + copysign(self.move_steps, diff))

            #print("Moving %s %f" % (self.joint, self.servo_pos))

            if self.servo_pos == self.target_pos or \
                self.servo_pos <= self.servo_minpos or \
                self.servo_pos >= self.servo_maxpos:
                self.target_pos = None;

        else:
            if self.joint == "pan" and voice_angle != None:
                if time.monotonic() - self.last_adj_by_voice < 2.0:
                    return

                angle = float(voice_angle) - 90.0
                print('angle: %f, %d' % (angle, voice_angle))
                if angle > 180.0:
                    angle = -90.0
                elif angle > 90.0:
                    angle = 90.0
                pos = int(self.get_servo_pos_from_degrees(angle))
                print("pos from voice: %d, angle: %f" % (pos, angle))
                self.target_pos = pos
                self.move_steps = 5
                self.last_adj_by_voice = time.monotonic()
                self.auto_center_time = time.monotonic()
                return

            # No object visible now.
            # If was tracking on last update, then continue moving in that
            # direction to try and catch up.  If the limit is reached, then
            # return to center after a timeout. If wasn't tracking before,
            # then return to center after a timeout.
            if False:
            #if self.obj_last_dir != 0:
                self.set_servo_pos(self.servo_pos + self.obj_last_dir*self.servo_step*1)
                self.auto_center_time = time.monotonic()
                if self.is_at_servo_limit():
                    self.obj_last_dir = 0
            elif self.auto_center_time != 0.0 and time.monotonic() - self.auto_center_time > 6.0:
                self.auto_center_time = 0.0
                self.obj_last_dir = 0
                self.target_pos = self.servo_midpos
                self.move_steps = 2

class CameraTracker(Node):
    def __init__(self):
        super().__init__('camera_tracker')

#   def __init__(self, node):
#      self.node = node

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

        self.sub_track = self.create_subscription(
            Track,
            '/head/track',
            self.track_callback,
            2)

        self.sub_head_tilt = self.create_subscription(
            HeadTilt,
            '/head/tilt',
            self.head_tilt_callback,
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

        self.track_cmd_mode = "Track"
#        self.track_cmd_mode = "None"
        self.track_rate = 0
        self.track_new_mode = None
        self.track_new_level = 0
        self.track_voice_detect = False
        self.last_track_mode = None

        self.head_tilt_cmd_angle = None
        self.head_tilt_steps = 0
        self.head_tilt_dwell_ticks = 0

        # Create an object for controller each pan-tilt base joint
        self.servo_pan = CameraServo("pan")
        self.servo_tilt = CameraServo("tilt")
        self.servo_head_tilt = CameraServo("rotate")

        self.tilt_scan_pos = self.servo_tilt.servo_midpos - 5
        self.scan_left = True
        self.scan_step = 4

        self.scan_at_left_count = 0
        self.scan_at_right_count = 0

        self.speech_angle = None
        self.speech_detected = None

        self.smile_timer = self.create_timer(0.1, self.smile_timer_callback)

        self.thread = threading.Thread(target=self.tracker_thread)
        self.thread.start()

    def tracker_thread(self):
        tilt_cnt = 0
        while True:
            time.sleep(0.05)

            tilt_cnt += 1
            if tilt_cnt >= 2:
                tilt_cnt = 0
                self.head_tilt_timer_callback()

            self.head_scan_timer_callback()
            self.broadcast_camera_joints()

    # Does not work since RCLPY does not support this yet
    def publish_detected_pose(self, obj):
        try:
            self.tf_map_to_oakd = self.tfBuffer.lookup_transform("map", "oakd",
                                                                 rclpy.time.Time(),
                                                                 Duration(seconds=0.3))
        except Exception as e:
            self.get_logger().info(str(e))
            return
        else:
            self.get_logger().info('Got transform')

        now = self.get_clock().now()

        msg = PoseStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "oakd"
        msg.pose.position.x = obj.x;
        msg.pose.position.y = obj.y;
        msg.pose.position.z = obj.z;
        msg.pose.orientation.x = 0.0;
        msg.pose.orientation.y = 0.0;
        msg.pose.orientation.z = 0.0;
        msg.pose.orientation.w = 1.0;

        map_pose = self.tfBuffer.transform(msg, "map")
        self.goal_pub.publish(map_pose)

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

    # def update_camera_pos(self, obj):
    #     pan_pos = self.servo_pan.servo_pos
    #     tilt_pos = self.servo_tilt.servo_pos
    #
    #     self.servo_pan.update(obj)
    #     self.servo_tilt.update(obj)
    #     self.broadcast_camera_joints()

    def process_detections(self, objListTrack):
        if self.track_new_mode != None:
            if self.track_new_mode == 'Off':
                self.servo_pan.stop_tracking()
                self.servo_tilt.stop_tracking()
            elif self.track_new_mode == "LookDown":
                self.servo_pan.stop_tracking()
                self.servo_tilt.stop_tracking()
                self.servo_tilt.set_servo_pos(self.servo_tilt.servo_minpos)
            elif self.track_new_mode == "Scan":
                self.servo_tilt.set_servo_pos(self.servo_tilt.servo_midpos)
            elif self.track_new_mode == "Track":
                self.servo_tilt.set_servo_pos(self.servo_tilt.servo_midpos)

            self.track_cmd_mode = self.track_new_mode
            self.track_new_mode = None
            self.publish_scan_status()

        if self.track_cmd_mode == "Track":
            self.last_track_mode = self.track_cmd_mode

            pan_pos = self.servo_pan.servo_pos
            tilt_pos = self.servo_tilt.servo_pos

            closest_obj = None

            cnt = len(objListTrack)
            if cnt > 0:
                for obj in objListTrack:
                    if obj.name != 'person' or obj.confidence < min_track_confidence:
                        continue

                    if closest_obj != None:
                        # x is according to ROS conventions (pointing away from camera)
                        if closest_obj.x >  obj.x:
                            closest_obj = obj
                    else:
                        closest_obj = obj

            if self.track_voice_detect == False:
                self.speech_angle = None

            self.servo_pan.update(closest_obj, self.speech_angle)
            self.servo_tilt.update(closest_obj, None)
            self.broadcast_camera_joints()

            self.speech_angle = None

            #if closest_obj != None:
            #    self.publish_detected_pose(closest_obj)

    def obj_detection_callback(self, msg):
        #self.get_logger().info('Received object detection msg')
        self.process_detections(msg.objects)


    # Move the hand in a left-to-right scanning motion
    def update_scan(self, init):
        if init:
            self.scan_left = True
            self.servo_tilt.set_servo_pos(self.tilt_scan_pos)

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

    def publish_scan_status(self):
        scan_status = ScanStatus()
        scan_status.scanning = self.track_cmd_mode == "Scan"
        scan_status.angle = int(self.servo_pan.get_servo_degrees())
        scan_status.direction = 1 if self.scan_left else -1
        scan_status.at_left_count = self.scan_at_left_count
        scan_status.at_right_count = self.scan_at_right_count
        self.pub_scan_status.publish(scan_status)

    def head_scan_timer_callback(self):
        if self.track_cmd_mode == "Scan":
            self.update_scan(self.last_track_mode != "Scan")
            self.last_track_mode = self.track_cmd_mode
            self.publish_scan_status()

    def head_tilt_timer_callback(self):
        self.update_head_tilt()

    def head_tilt_callback(self, msg):
        self.get_logger().info('Received head tilt msg: angle: %s, transition_duration: %d, dwell_duration: %d' % \
            (msg.angle, msg.transition_duration, msg.dwell_duration))
        self.head_tilt_cmd_angle = msg.angle
        self.head_tilt_cmd_trans_dur = msg.transition_duration
        self.head_tilt_cmd_dwell_dur = msg.dwell_duration

    def update_head_tilt(self):
        if self.head_tilt_cmd_angle != None:
            self.head_tilt_steps = self.head_tilt_cmd_angle/self.servo_head_tilt.servo_degrees_per_step
            self.head_tilt_cmd_angle = None

            if self.head_tilt_steps < 0:
                self.head_tilt_dir = -1
            else:
                self.head_tilt_dir = 1

            self.head_tilt_steps *= self.head_tilt_dir
            self.head_tilt_dwell_ticks = int((self.head_tilt_cmd_dwell_dur + 99)/100)

            #print("head_tilt_steps= %d, dir= %d" % (self.head_tilt_steps, self.head_tilt_dir))

        if self.head_tilt_steps > 0:
            self.servo_head_tilt.set_servo_pos(self.servo_head_tilt.servo_pos + self.head_tilt_dir*10)
            self.head_tilt_steps -= 10
            #print("head_tilt_steps= %d, dir= %d, servo_pos %d" % (self.head_tilt_steps, self.head_tilt_dir, self.servo_head_tilt.servo_pos))

        elif self.head_tilt_dwell_ticks > 0:
            self.head_tilt_dwell_ticks -= 1
            #print("head_tilt_dwell_ticks= %d" % (self.head_tilt_dwell_ticks))

            if self.head_tilt_dwell_ticks == 0:
                self.head_tilt_steps = self.servo_head_tilt.servo_midpos - self.servo_head_tilt.servo_pos

                if self.head_tilt_steps < 0:
                    self.head_tilt_dir = -1
                else:
                    self.head_tilt_dir = 1

                self.head_tilt_steps *= self.head_tilt_dir
                #print("head_tilt_steps= %d, dir= %d" % (self.head_tilt_steps, self.head_tilt_dir))

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
        if msg.mode == "Scan":
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

    def speaking_callback(self, msg):
        self.get_logger().info('Received speaking active msg: speaking: %d' % msg.data)
        if msg.data:
            self.smile_cmd_mode = "talking"
        elif self.smile_mode == "talking":
            self.smile_cmd_mode = "default"
        else:
            return
        self.update_smile()

    def speech_aoa_callback(self, msg):
        self.get_logger().info('Received speech AOA msg: angle: %d' % msg.data)
        self.speech_angle = msg.data

    def speech_vad_callback(self, msg):
        self.get_logger().info('Received speech VAD msg: detected: %d' % msg.data)
        self.speech_detected = msg.data

def main(args=None):
    rclpy.init(args=args)
    depthai_publisher = CameraTracker()

    rclpy.spin(depthai_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    depthai_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


