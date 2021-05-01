import os
import os.path
from os import path

import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from face_control_interfaces.msg import Smile, HeadTilt

# Used for publishing the camera joint positions
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile

# Servo control - uses Adafruit ServoKit to drive
# ADAFRUIT PCA9685 16-channel servo driver
import time
from adafruit_servokit import ServoKit

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
            self.servo_minpos = 0
            self.servo_maxpos = 170
            self.servo_midpos = 78

            # Approx angle measurements for the above positions
            self.servo_degrees_per_step = (57.0 + 65.0)/(self.servo_maxpos - self.servo_minpos)
            self.servo_mid_degrees = 57.0

        elif joint == "tilt":
            self.chan = 13

            self.servo_minpos = 0
            self.servo_maxpos = 80
            self.servo_midpos = 40

            self.servo_degrees_per_step = (14.0 + 51.0)/(self.servo_maxpos - self.servo_minpos)
            self.servo_mid_degrees = 14.0

        elif joint == "rotate":
            self.chan = 14

            self.servo_minpos = 0
            self.servo_maxpos = 170
            self.servo_midpos = 85

            self.servo_degrees_per_step = (57.0 + 65.0)/(self.servo_maxpos - self.servo_minpos)
            self.servo_mid_degrees = 57.0

        else:
            print("Invalid camera joint: %s" % joint)
            return

        self.servo_pos = 0
        # Steps per position adjustment
        self.servo_step = 1
        self.set_servo_pos(self.servo_midpos)

        self.obj_ave = 0.0
        self.obj_last_dir = 0
        self.obj_last_pos = 0.0
        self.obj_timeout_cnt = -1

    def is_at_servo_limit(self):
        return  self.servo_pos == self.servo_minpos or self.servo_pos == self.servo_maxpos

    def set_servo_pos(self, pos):
        if pos < self.servo_minpos:
            pos = self.servo_minpos
        elif pos > self.servo_maxpos:
            pos = self.servo_maxpos

        global servo_kit
        servo_kit.servo[self.chan].angle = pos
        self.servo_pos = pos
        #print("Set servo pos %d" % pos)

    def get_servo_degrees(self):
        deg = self.servo_pos*self.servo_degrees_per_step - self.servo_mid_degrees
        #print("servo pos: %d -> degrees: %f" % (self.servo_pos, deg))
        return deg

    # Update the pan-tilt base
    def update(self, obj):
        if obj != None:
            #print('Joint: %s, Object to track: %f, %f [%f, %f]' % (self.joint, obj['x'], obj['y'], obj['x_min'], obj['x_max']))

            if self.joint == "pan":
                pos = (obj['x_min'] + obj['x_max'])/2
            else:
                pos = obj['y_min']

            self.obj_ave = self.obj_ave*0.5 + pos*0.5
            self.obj_last_pos = pos

            if self.joint == "pan":
                # Try to object in center of left-right view
                if self.obj_ave > 0.6:
                    self.set_servo_pos(self.servo_pos - self.servo_step)
                    self.obj_last_dir = -1
                elif self.obj_ave < 0.4:
                    self.set_servo_pos(self.servo_pos + self.servo_step)
                    self.obj_last_dir = 1
                else:
                    self.obj_last_dir = 0
            else:
                # Try to keep top of object (person) in view
                if self.obj_ave > 0.3:
                    self.set_servo_pos(self.servo_pos - self.servo_step)
                    self.obj_last_dir = -1
                elif self.obj_ave < 0.1:
                    self.set_servo_pos(self.servo_pos + self.servo_step)
                    self.obj_last_dir = 1
                else:
                    self.obj_last_dir = 0

            self.obj_timeout_cnt = 0
        else:
            # No object visible now.
            # If was tracking on last update, then continue moving in that
            # direction to try and catch up.  If the limit is reached, then
            # return to center after a timeout. If wasn't tracking before,
            # then return to center after a timeout.
            go_center = False
            if self.obj_last_dir != 0:
                self.set_servo_pos(self.servo_pos + self.obj_last_dir*self.servo_step)
                self.obj_timeout_cnt = 0
                if self.is_at_servo_limit():
                    self.obj_last_dir = 0
            elif self.obj_timeout_cnt >= 0:
                self.obj_timeout_cnt += 1
                if self.obj_timeout_cnt > 30:
                    self.obj_last_dir = 0
                    self.obj_timeout_cnt = -1
                    go_center = True

            if go_center:
                self.set_servo_pos(self.servo_midpos)

class CameraTracker:
    def __init__(self, node):
        self.node = node

        init_servo_driver()

        self.sub_smile = node.create_subscription(
            Smile,
            'smile',
            self.smile_callback,
            10)
        self.sub_smile  # prevent unused variable warning

        self.sub_head_tilt = node.create_subscription(
            HeadTilt,
            'head_tilt',
            self.head_tilt_callback,
            10)
        self.sub_head_tilt  # prevent unused variable warning

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

        self.head_tilt_cmd_angle = None
        self.head_tilt_steps = 0
        self.head_tilt_dwell_ticks = 0

        # Create an object for controller each pan-tilt base joint
        self.servo_pan = CameraServo("pan")
        self.servo_tilt = CameraServo("tilt")
        self.servo_head_tilt = CameraServo("rotate")

         # Publisher for states of the pan-tilt joints
        qos_profile = QoSProfile(depth=10)
        self.joint_pub = node.create_publisher(JointState, 'joint_states', qos_profile)

#        self.timer = self.create_timer(0.2, self.timer_callback)

        self.smile_timer = self.node.create_timer(0.1, self.smile_timer_callback)

        self.head_tilt_timer = self.node.create_timer(0.1, self.head_tilt_timer_callback)

    # Broadcast the pan-tilt joints so ROS TF can be used to tranform positions
    # in the camera frame to other frames such as the map frame when navigating.
    # My URDF file names the camera joints as: 'cam_tilt_joint', 'cam_pan_joint
    def broadcast_camera_joints(self):
        cam_pan_rad = self.servo_pan.get_servo_degrees()/180.0*PI
        cam_tilt_rad = -1.0*self.servo_tilt.get_servo_degrees()/180.0*PI

        now = self.node.get_clock().now()
        joint_state = JointState()
        joint_state.header.stamp = now.to_msg()
        joint_state.name = ['cam_tilt_joint', 'cam_pan_joint']
        joint_state.position = [cam_tilt_rad, cam_pan_rad]
        self.joint_pub.publish(joint_state)

    def update_camera_pos(self, obj):
        pan_pos = self.servo_pan.servo_pos
        tilt_pos = self.servo_tilt.servo_pos

        self.servo_pan.update(obj)
        self.servo_tilt.update(obj)
        self.broadcast_camera_joints()

    def process_detections(self, objListTrack):
        pan_pos = self.servo_pan.servo_pos
        tilt_pos = self.servo_tilt.servo_pos

        closest_obj = None

        cnt = len(objListTrack)
        if cnt > 0:
            for obj in objListTrack:
                if closest_obj != None:
                    # x is according to ROS conventions (pointing away from camera)
                    if closest_obj['x'] >  obj['x']:
                        closest_obj = obj
                else:
                    closest_obj = obj

        self.servo_pan.update(closest_obj)
        self.servo_tilt.update(closest_obj)
        self.broadcast_camera_joints()

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

    def set_smile(self):
        for i in range(0, len(self.smile_leds)):
            pca.channels[smile_led_map[i]].duty_cycle = 0 if self.smile_leds[i] else 65535

    def update_smile(self):
        if self.smile_cmd_mode != None:
            mode = self.smile_cmd_mode
            self.smile_cmd_mode = None

            print("New Smile cmd: %s" % mode)

            if mode != self.smile_mode:
                print("New mode")

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

            self.smile_level += self.smile_delta
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

