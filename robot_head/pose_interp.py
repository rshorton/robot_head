
import numpy as np
import copy
import math
import sys
import os

nPoints = 18

PI = 3.1415
def calc_joint_angle(base_pt, joint_pt, end_pt):
    if base_pt[0] == invalid_coord:
        print("Missing base_pt")
        return None
    if joint_pt[0] == invalid_coord:
        print("Missing joint_pt")
        return None
    if end_pt[0] == invalid_coord:
        print("Missing end_pt")
        return None

    b_j = np.linalg.norm(base_pt - joint_pt)
    j_e = np.linalg.norm(joint_pt - end_pt)
    e_b = np.linalg.norm(end_pt - base_pt)

    nom = b_j*b_j + j_e*j_e - e_b*e_b
    denom = 2*b_j*j_e

    a = 90.0
    #print('nom %f, denom %f, %f' % (nom, denom, nom/denom))
    if nom != 0.0:
        try:
            a = math.acos(nom/denom)
            a *= 180.0/PI
            a = 180.0 - a
        except:
            print('nom %f, denom %f, %f' % (nom, denom, nom/denom))
            return None

    # Determine sign of angle. Positive if point w above line ne_s.
    # Slope of ne_s
    #print("s1 %f, ne1 %f,  s0 %f, ne0 %f" % (s[1], ne[1], s[0], ne[0]))

    rise = joint_pt[1] - base_pt[1]
    run = joint_pt[0] - base_pt[0]
    if run == 0.0:
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

    return int(a)

def pose_from_angles(s, e):
    if s != None and e != None:
        if s > -70 and s < -25 and e > -100 and e < -40:
            return "OnHip"
        #elif s > -15 and s < 15 and e > 130 and e < 150:
        #    return "TouchingEar"
        elif s > -30 and s < 30 and e > -20 and e < 20:
            return "ArmOut"
        elif s > -100 and s < -55 and e > -30 and e < 30:
            return "ArmToSide"
        elif s > -60 and s < 0 and e > 130 and e < 155:
            return "TouchingShoulder"
        elif s > -80 and s < -45 and e > -130 and e < -90:
            return "TouchingStomach"
        elif s > 20 and s < 120 and e > -10 and e < 80:
            return "Abovehead"
        elif s > 0 and s < 50 and e > 100 and e < 150:
            return "OnHead"
        elif s > -100 and s < -45 and e > -180 and e < -140:
            return "TouchingNeck"

    return "Unknown"

def detect_pose_side(side, w, e, s, no, ne):
    w = np.array(w)
    e = np.array(e)
    s = np.array(s)
    no = np.array(no)
    ne = np.array(ne)

    #print("                   ne %s,  s %s,  e %s, w %s" % (ne, s, e, w))

    joint_s = calc_joint_angle(ne, s, e);
    joint_e = calc_joint_angle(s, e, w);

    pose = pose_from_angles(joint_s, joint_e)

    print("%s: S %s,  E %s, Pose %s" % (side, joint_s, joint_e, pose))
    #print("%s: E %s" % (side, joint_e))

    return pose

invalid_coord = 10000.0

def get_empty_pose():
    pose = {};
    pose["left"] = "none"
    pose["right"] = "none"
    return pose

def detect_pose(kp, pp_by_name):

    detected = "false"
    pose_r = "none"
    pose_l = "none"
    if kp[pp_by_name['Nose']][0] != invalid_coord and kp[pp_by_name['Neck']][0] != invalid_coord:
        detected = "true"
        print("------------------")
        #print("%s, %s, %s" % (kp[pp_by_name['WrR']], kp[pp_by_name['ElbR']], kp[pp_by_name['ShoR']]))
        pose_r = detect_pose_side("Right", kp[pp_by_name['WrR']]*[-1.0,1.0], kp[pp_by_name['ElbR']]*[-1.0,1.0], kp[pp_by_name['ShoR']]*[-1.0,1.0],
                         kp[pp_by_name['Nose']], kp[pp_by_name['Neck']]);

        pose_l = detect_pose_side("Left", kp[pp_by_name['WrL']], kp[pp_by_name['ElbL']], kp[pp_by_name['ShoL']],
                         kp[pp_by_name['Nose']], kp[pp_by_name['Neck']]);


    pose = {};
    pose["left"] = pose_l
    pose["right"] = pose_r
    return pose

#    pose = detected + "," + pose_l + "," + pose_r
#    f = open("/home/ubuntu/tmp/pose_wr", "x");
#    f.write(pose)
#    f.close()
#    os.rename("/home/ubuntu/tmp/pose_wr", "/home/ubuntu/tmp/pose")


def show_kp(kp, pp_by_name):
    print("              Nose = %s" % kp[pp_by_name['Nose']])
    print("              Neck = %s" % kp[pp_by_name['Neck']])
    print(" ShoR = %s                    ShoL = %s" % (kp[pp_by_name['ShoR']], kp[pp_by_name['ShoL']]))
    print(" ElbR = %s                    ElbL = %s" % (kp[pp_by_name['ElbR']], kp[pp_by_name['ElbL']]))
    print(" WrR = %s                     WrL = %s" % (kp[pp_by_name['WrR']], kp[pp_by_name['WrL']]))

def show_openpose(keypoints_limbs, frame, **kwargs):
    frame = np.uint8(frame.copy())
    if len(keypoints_limbs) == 3:
        detected_keypoints = keypoints_limbs[0]
        personwiseKeypoints = keypoints_limbs[1]
        keypoints_list = keypoints_limbs[2]

        dump = False
        if dump == True:
            print("detected_keypoints")
            print(detected_keypoints)

            print("personwiseKeypoints")
            print(personwiseKeypoints)

            print("keypoints")
            print(keypoints_list)

            #keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
            # detected_keypoints
            # [ [(211, 92, 0.7802639, 0)], [(204, 155, 0.7402401, 1)], [(170, 148, 0.73155975, 2)], [(130, 203, 0.673028, 3)], [], [(250, 156, 0.6964865, 4)],
            #   [(275, 221, 0.62695503, 5)], [], [], [], [], [],
            #   [], [], [(203, 90, 0.84602356, 6)], [(219, 91, 0.80147266, 7)], [(189, 100, 0.81802654, 8)],
            #   [(227, 100, 0.86163807, 9)]]
            #
            # personwiseKeypoints
            # [[ 0.         1.         2.         3.        -1.         4.
            #    5.        -1.        -1.        -1.        -1.        -1.
            #   -1.        -1.         6.         7.         8.         9.
            #   15.4935609]]
            #
            # keypoints_list
            # [[211.          92.           0.7802639 ]
            #  [204.         155.           0.7402401 ]
            #  [170.         148.           0.73155975]
            #  [130.         203.           0.67302799]
            #  [250.         156.           0.69648647]
            #  [275.         221.           0.62695503]
            #  [203.          90.           0.84602356]
            #  [219.          91.           0.80147266]
            #  [189.         100.           0.81802654]
            #  [227.         100.           0.86163807]]
            #
            # Nose(211, 92), Neck(204, 155),       R-Sho(170, 148), R-Elb(130, 203),     L-Sho(250, 156), L-Elb(275, 221),
            # R-Eye(203, 90), L-Eye(219, 91),      R-Ear(189, 100), L-Ear(227, 100),

        pose = ""

        pp_by_name = {'Nose': 0,  'Neck': 1,
                        'ShoR': 2,  'ElbR': 3,  'WrR': 4,
                        'ShoL': 5,  'ElbL': 6,  'WrL': 7,
                        'HipR': 8,  'KnR':  9,  'AnkR': 10,
                        'HipL': 11, 'KnL':  12, 'AnkL': 13,
                        'EyeR': 14, 'EyeL': 15,
                        'EarR': 16, 'EarL': 17};


        #print(pose)

        for i in range(17):
            for n in range(len(personwiseKeypoints)):
                if n > 0:
                    continue

                #print("np.array(POSE_PAIRS[i])")
                #print(np.array(POSE_PAIRS[i]))
                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                #print(index)
                if -1 in index:
                    continue
                B = np.int32(keypoints_list[index.astype(int), 0])
                A = np.int32(keypoints_list[index.astype(int), 1])

                radius = 5
                cv2.circle(frame, (B[0], A[0]), radius, colors[i], -1, cv2.LINE_AA)
                cv2.circle(frame, (B[1], A[1]), radius, colors[i], -1, cv2.LINE_AA)

                cv2.line(frame, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

    return frame

dump = False

def analyze_pose(detected_keypoints, keypoints_list, personwiseKeypoints):

    pose = get_empty_pose()
    if keypoints_list is not None and detected_keypoints is not None and personwiseKeypoints is not None:
        if dump == True:
            print("detected_keypoints")
            print(detected_keypoints)

            print("personwiseKeypoints")
            print(personwiseKeypoints)

            print("keypoints")
            print(keypoints_list)

            #keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
            # detected_keypoints
            # [ [(211, 92, 0.7802639, 0)], [(204, 155, 0.7402401, 1)], [(170, 148, 0.73155975, 2)], [(130, 203, 0.673028, 3)], [], [(250, 156, 0.6964865, 4)],
            #   [(275, 221, 0.62695503, 5)], [], [], [], [], [],
            #   [], [], [(203, 90, 0.84602356, 6)], [(219, 91, 0.80147266, 7)], [(189, 100, 0.81802654, 8)],
            #   [(227, 100, 0.86163807, 9)]]
            #
            # personwiseKeypoints
            # [[ 0.         1.         2.         3.        -1.         4.
            #    5.        -1.        -1.        -1.        -1.        -1.
            #   -1.        -1.         6.         7.         8.         9.
            #   15.4935609]]
            #
            # keypoints_list
            # [[211.          92.           0.7802639 ]
            #  [204.         155.           0.7402401 ]
            #  [170.         148.           0.73155975]
            #  [130.         203.           0.67302799]
            #  [250.         156.           0.69648647]
            #  [275.         221.           0.62695503]
            #  [203.          90.           0.84602356]
            #  [219.          91.           0.80147266]
            #  [189.         100.           0.81802654]
            #  [227.         100.           0.86163807]]
            #
            # Nose(211, 92), Neck(204, 155),       R-Sho(170, 148), R-Elb(130, 203),     L-Sho(250, 156), L-Elb(275, 221),
            # R-Eye(203, 90), L-Eye(219, 91),      R-Ear(189, 100), L-Ear(227, 100),

        pp_by_name = {'Nose': 0,  'Neck': 1,
                        'ShoR': 2,  'ElbR': 3,  'WrR': 4,
                        'ShoL': 5,  'ElbL': 6,  'WrL': 7,
                        'HipR': 8,  'KnR':  9,  'AnkR': 10,
                        'HipL': 11, 'KnL':  12, 'AnkL': 13,
                        'EyeR': 14, 'EyeL': 15,
                        'EarR': 16, 'EarL': 17};

        person = 0
        kp = [[invalid_coord, invalid_coord]]*nPoints
        if person < len(personwiseKeypoints):
            for n in range(len(personwiseKeypoints[person]) - 1):
                index = int(personwiseKeypoints[person][n])
                if index != -1 and index < nPoints:
                    #kp[n] = np.int32(keypoints_list[index][0:2])
                    kp[n] = copy.copy(keypoints_list[index][0:2])
                    #print("n = %d, index= %d, %s" % (n, index, kp[n]))

            #show_kp(kp, pp_by_name)
            neck = copy.copy(kp[pp_by_name['Neck']])
            #print("neck %s" % neck)
            try:
                if neck[0] != invalid_coord:
                    for n in range(len(kp)):
                        if (kp[n][0] != invalid_coord):
                            kp[n] -= neck
                            kp[n] *= [1.0, -1.0]
                    #show_kp(kp, pp_by_name)
            except:
                print("Unexpected error:", sys.exc_info()[0])
                print("neck %s" % neck)
                print("detected_keypoints")
                print(detected_keypoints)

                print("personwiseKeypoints")
                print(personwiseKeypoints)

                print("keypoints")
                print(keypoints_list)

            kp = np.array(kp)
            pose = detect_pose(kp, pp_by_name)
    return pose
