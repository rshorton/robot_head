
import numpy as np
import copy
import math
import sys
import os

pose_calc_debug = False

def calc_joint_angle(base_pt, joint_pt, end_pt):
    b_j = np.linalg.norm(base_pt - joint_pt)
    j_e = np.linalg.norm(joint_pt - end_pt)
    e_b = np.linalg.norm(end_pt - base_pt)

    nom = b_j*b_j + j_e*j_e - e_b*e_b
    denom = 2*b_j*j_e

    a = 90.0
    #print('nom %f, denom %f, %f' % (nom, denom, nom/denom))
    if nom != 0.0:
        try:
            a = np.arccos(nom/denom)
            a *= 180.0/np.pi
            a = 180.0 - a
        except Exception as e:
            print(e)
            print('nom %f, denom %f, %f' % (nom, denom, nom/denom))
            return None

    # Determine sign of angle. Positive if point w above line ne_s.
    # Slope of ne_s
    rise = joint_pt[1] - base_pt[1]
    run = joint_pt[0] - base_pt[0]
    if run == 0.0:
        if pose_calc_debug:
            print("base segment is vert")
        # base segment is vertical, use x for angle direction
        if end_pt[0] < joint_pt[0]:
            a *= -1;
    else:
        m = rise/run
        # y intercept
        b = base_pt[1] - m*base_pt[0]

        val = m*end_pt[0] + b
        #print("slope %f, yinter %f, val %f  end_pt_y %f" % (m, b, val, end_pt[1]))
        if end_pt[1] < (m*end_pt[0] + b):
            a *= -1.0

    # fix - needed *-1 after moving to blazepose
    return int(a*-1.0)


def pose_from_angles(s, e):
    if s != None and e != None:
        if s > -70 and s < -25 and e > -100 and e < -40:
            return "OnHip"
        #elif s > -15 and s < 15 and e > 130 and e < 150:
        #    return "TouchingEar"
        elif s > -30 and s < 30 and e > -45 and e < 45:
            return "ArmOut"
        elif s > -100 and s < -55 and e > -30 and e < 30:
            return "ArmToSide"
        elif s > -75 and s < 0 and e > 130 and e < 160:
            return "TouchingShoulder"
        elif s > -80 and s < -45 and e > -130 and e < -90:
            return "TouchingStomach"
        elif s > 20 and s < 120 and e > -10 and e < 80:
            return "Abovehead"
        elif s > 0 and s < 60 and e > 70 and e < 150:
            return "OnHead"
        elif s > -100 and s < -45 and e > -180 and e < -140:
            return "TouchingNeck"

    return "none"

def detect_pose_side(side, w, e, s, os):
    w = np.array(w)
    e = np.array(e)
    s = np.array(s)
    os = np.array(os)

    if pose_calc_debug:
        print("                   os %s,  s %s,  e %s, w %s" % (os, s, e, w))

    joint_s = calc_joint_angle(os, s, e);
    joint_e = calc_joint_angle(s, e, w);

    pose = pose_from_angles(joint_s, joint_e)

    if pose_calc_debug:
        print("%s: S %s,  E %s, Pose %s" % (side, joint_s, joint_e, pose))
    return pose

def get_empty_pose():
    pose = {}
    pose["detected"] = False
    pose["left"] = "none"
    pose["right"] = "none"
    pose["num_points"] = 0
    return pose

def analyze_pose(bp_region):

    detected = False
    pose_r = "none"
    pose_l = "none"
    cnt = 0

    pose = {}

    try:
        pose["detected"] = True
        # Fix
        pose["num_points"] = 33

        if pose_calc_debug:
            print("------------------")

        pose_r = detect_pose_side("Right",
                                  bp_region.landmarks_abs[16,:2]*[-1.0,1.0], # Right wrist
                                  bp_region.landmarks_abs[14,:2]*[-1.0,1.0], # Right elbow
                                  bp_region.landmarks_abs[12,:2]*[-1.0,1.0], # Right Shoulder
                                  bp_region.landmarks_abs[11,:2]*[-1.0,1.0]) # Left Shoulder
        pose_l = detect_pose_side("Left",
                                  bp_region.landmarks_abs[15,:2], # Left wrist
                                  bp_region.landmarks_abs[13,:2], # Left elbow
                                  bp_region.landmarks_abs[11,:2], # Left Shoulder
                                  bp_region.landmarks_abs[12,:2]) # Right Shoulder
    except Exception as e:
        #print(e)
        pose["num_points"] = 0

    pose["left"] = pose_l
    pose["right"] = pose_r
    return pose