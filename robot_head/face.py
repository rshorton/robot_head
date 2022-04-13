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
import sys
import time
import math

import rclpy
from rclpy.node import Node
import rclpy.time
from rclpy.duration import Duration

from std_msgs.msg import String
from std_msgs.msg import Bool
from std_msgs.msg import Int32
from std_msgs.msg import Float32

from robot_head_interfaces.msg import Smile, Antenna

# Servo control - uses Adafruit ServoKit to drive
# ADAFRUIT PCA9685 16-channel servo driver
import time
from adafruit_servokit import ServoKit

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

class FacialExpression(Node):
    def __init__(self):
        super().__init__('facial_expression')

        init_servo_driver()

        # Topic for receiving smile display commands.
        # These commands specify the smile level and/or temporary
        # increased smile.
        self.sub_smile = self.create_subscription(
            Smile,
            '/head/smile',
            self.smile_callback,
            2)

        # Topic for receiving message indicating when speech output is active.
        # When active, the smile display is shows a 'talking' pattern instead
        # of the current smile level.
        self.sub_speaking = self.create_subscription(
            Bool,
            '/head/speaking',
            self.speaking_callback,
            2)

        # Topic for receiving message indicating when speech-to-text is active.
        # When active, the antennae LEDs pattern is temporarily changed
        # to a 'listening' pattern
        self.sub_listening = self.create_subscription(
            Bool,
            '/speech_detect/listening',
            self.listening_callback,
            2)

        # Topic for receiving command to control Antennae pattern
        self.sub_antenna = self.create_subscription(
            Antenna,
            '/head/antenna',
            self.antenna_callback,
            2)

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
        self.listening = False

        self.smile_timer = self.create_timer(0.1, self.smile_timer_callback)
        self.antenna_timer = self.create_timer(0.1, self.antenna_timer_callback)

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
            self.smile_level = min(self.smile_level + self.smile_delta, len(smile_patterns) - 1)
            self.smile_leds = smile_patterns[self.smile_level]
            self.set_smile()

            if self.smile_level == self.smile_target_level:
                self.smile_delta = 0

        elif self.smile_duration > 0:
            self.smile_duration -= 1
            if self.smile_duration == 0 and self.smile_level != self.smile_level_def:
                self.smile_delta = -1 if self.smile_level > self.smile_level_def else 1
                self.smile_target_level = self.smile_level_def
                #print("new smile_delta, delta= %d" % self.smile_delta)

        elif self.smile_mode == "talking":
            self.smile_talk_index += 1
            if self.smile_talk_index >= len(talk_patterns):
                self.smile_talk_index = 1

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
            pca.channels[antenna_left_ch].duty_cycle = 10000
            pca.channels[antenna_right_ch].duty_cycle = 10000
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

def main(args=None):
    rclpy.init(args=args)
    expresso = FacialExpression()

    rclpy.spin(expresso)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    expresso.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


