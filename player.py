import time
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import os
import sys
import argparse
import matplotlib.pyplot as plt
from sys import platform
from scipy.optimize import curve_fit
import json
from math import pi
from ball import balls



tf.disable_v2_behavior()

##### ball detection function is in the one_ball class
##### get openpose data function is independant function, here i just wrote a read json file function

##### player_list should be updated in the ReID process and here we will use the most updated player_list



class player:
    def __init__(self, person_id):
        self.id = person_id
        self.img_path = []
        self.model_path = []
        self.time_frame = []
        self.current_img_position = np.zeros([1,2], dtype='float32')
        self.current_model_position = np.zeros([1,2], dtype='float32')
        self.previous_img_position = np.zeros([1,2], dtype='float32')
        self.previous_model_position = np.zeros([1,2], dtype='float32')
        self.skip_frames = int
        self.statistics = {
            'attempts': 0,
            'made': 0,
            'miss': 0,
            'duration': 0,
            'attempt_time':[],
            'made_position': [],
            'miss_position': [],
            'attempt_position': []
        }

        # pose data
        self.wrists_positions = [] # list of lists
        self.nose_position = []
        self.body_center_position = []

        self.current_wrists_positions = []
        self.current_nose_position = []
        self.current_body_center_position = []

        # relationship with ball
        self.ball_in_hand = False
        self.previous_hold_position = []
        self.previous_hold_model_position = []
        self.shooting_now = True


#### create player and player list using ReID and openpose
# here the code is for test
player_A = player('player_0')
player_B = player('player_1')
player_list = [player_A, player_B]


#### read openpose data into player class

#### In reality, the cropped openpose body box will be matched with ReID dictionary,
# find the match one and at the same time update the player class
# or create a new one and then update the player list
# always make sure to run openpose and ReID just one time to same computation.

def distance(x, y):
    return (np.linalg.norm(x - y))


def read_json_with_video(frame_index, file_path, court_class):

    global player_list

    # calculate the angle between three points abc
    def calculate_angle(a, b, c):
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return angle

    def determine_if_p_in_quanlilateral(point_array, p):
        angle = 0
        for i in range(point_array.shape[0]):
            if i < 3:
                angle = angle + calculate_angle(point_array[i, :], p, point_array[i + 1, :])
            else:
                angle = angle + calculate_angle(point_array[i, :], p, point_array[0, :])
        lower_bound = 2 * pi * 0.95
        upper_bound = 2 * pi * 1.05
        if (angle > lower_bound) & (angle < upper_bound):
            return True
        else:
            return False

    """
    // {0, "Nose"},
    // {1, "Neck"},
    // {2, "RShoulder"},
    // {3, "RElbow"},
    // {4, "RWrist"},
    // {5, "LShoulder"},
    // {6, "LElbow"},
    // {7, "LWrist"},
    // {8, "MidHip"},
    // {9, "RHip"},
    // {10, "RKnee"},
    // {11, "RAnkle"},
    // {12, "LHip"},
    // {13, "LKnee"},
    // {14, "LAnkle"},
    // {15, "REye"},
    // {16, "LEye"},
    // {17, "REar"},
    // {18, "LEar"},
    // {19, "LBigToe"},
    // {20, "LSmallToe"},
    // {21, "LHeel"},
    // {22, "RBigToe"},
    // {23, "RSmallToe"},
    // {24, "RHeel"},
    // {25, "Background"}
    """

    total_frames = 2000
    foot_x_index = [33, 42, 57, 66]
    foot_y_index = [34, 43, 58, 67]
    foot_probability_index = [35, 44, 59, 68]

    left_wrist_xy_index = [12, 13]
    right_wrist_xy_index = [21, 22]
    head_xy_index = [0, 1]
    body_x_index = [3, 24]
    body_y_index = [4, 25]


    counter_json = 0
    time_frame_list = []

    person_list = []

    # generate the json file name
    frame_index_str = str(frame_index)
    # part_name = 'v4_000000000000'
    part_name = 'img_0000'
    json_filepath = file_path + "/" + part_name[:-len(frame_index_str)] + frame_index_str + '.json'

    with open(json_filepath) as f:
        person_dict = json.load(f)
        # people = person_dict
        people = person_dict

        # using person_id to get the right person trajectory
        # using person_id to get the right person trajectory
        # using person_id to get the right person trajectory
        if len(people) == 0:
            pass
        else:
            for person in people:

                # find the right player
                this_player_index = next((index for index, player in enumerate(player_list) if player.id == person["person_id"]), None)
                # print(this_player_index)

                if this_player_index == None:
                    print("No this player")
                else:
                    # print(player_list[this_player_index])
                    pose_keypoints_2d = person["pose_keypoints_2d"]
                    flat_pose_list = [item for sublist in pose_keypoints_2d for item in sublist]

                    # feet
                    feet_position_x = np.mean([flat_pose_list[i] for i in foot_x_index])
                    feet_position_y = np.mean([flat_pose_list[j] for j in foot_y_index])
                    # feet_position_x = np.mean([pose_keypoints_2d[i] for i in foot_x_index])
                    # feet_position_y = np.mean([pose_keypoints_2d[j] for j in foot_y_index])
                    feet_position = np.array([[feet_position_x, feet_position_y]], dtype='float32')

                    # wrist
                    left_wrist_position = np.array([flat_pose_list[i] for i in left_wrist_xy_index])
                    right_wrist_position = np.array([flat_pose_list[i] for i in right_wrist_xy_index])
                    wrists_positions = [left_wrist_position, right_wrist_position]

                    # nose
                    nose_position = np.array([flat_pose_list[i] for i in head_xy_index])

                    # body
                    body_position_x = np.mean([flat_pose_list[i] for i in body_x_index])
                    body_position_y = np.mean([flat_pose_list[j] for j in body_y_index])
                    body_position = np.array([[body_position_x, body_position_y]], dtype='float32')



                    # print(feet_position[0,:])

                    # condition, the feet position should be inside the court
                    if determine_if_p_in_quanlilateral(court_class.img_corners, feet_position[0,:]):
                        # print("inside the court")

                        # transformed to model coordinates:
                        feet_image_positions = feet_position[:, np.newaxis, :]

                        # finally, get the mapping
                        feet_model_position = cv2.perspectiveTransform(feet_image_positions, court_class.H)
                        # print(feet_model_position[0, 0, :])

                        # player_list[this_player_index].img_path.append(feet_position[0, :])
                        # player_list[this_player_index].model_path.append(feet_model_position[0, 0, :])
                        # player_list[this_player_index].time_frame.append(frame_index)

                        if len(player_list[this_player_index].time_frame) == 0:

                            player_list[this_player_index].previous_img_position = feet_position[0,:]
                            dist = 0

                        else:
                            dist = distance(feet_position[0,:], player_list[this_player_index].previous_img_position) / (player_list[this_player_index].skip_frames + 1)
                            # print(dist)

                        if dist < 80:
                            # print("added")

                            # update previous_feet_position
                            player_list[this_player_index].previous_img_position = player_list[this_player_index].current_img_position
                            player_list[this_player_index].previous_model_position = player_list[this_player_index].current_model_position

                            # update the player's positions
                            player_list[this_player_index].img_path.append(feet_position[0,:])
                            player_list[this_player_index].model_path.append(feet_model_position[0, 0, :])
                            player_list[this_player_index].time_frame.append(frame_index)
                            player_list[this_player_index].skip_frames = 0
                            player_list[this_player_index].current_img_position = feet_position[0,:]
                            player_list[this_player_index].current_model_position = feet_model_position[0, 0, :]

                            # updata current pose position
                            player_list[this_player_index].current_wrists_positions = wrists_positions
                            player_list[this_player_index].current_nose_position = nose_position
                            player_list[this_player_index].current_body_center_position= body_position

                            player_list[this_player_index].wrists_positions.append(wrists_positions)
                            player_list[this_player_index].nose_position.append(nose_position)
                            player_list[this_player_index].body_center_position.append(body_position)



                        else:
                            player_list[this_player_index].skip_frames = player_list[this_player_index].skip_frames + 1

                            # append the same value as before to path
                            player_list[this_player_index].img_path.append(player_list[this_player_index].previous_img_position)
                            player_list[this_player_index].model_path.append(player_list[this_player_index].previous_model_position)
                            player_list[this_player_index].time_frame.append(frame_index)

                            # append current pose position to path
                            player_list[this_player_index].wrists_positions.append(player_list[this_player_index].current_wrists_positions)
                            player_list[this_player_index].nose_position.append(player_list[this_player_index].current_nose_position)
                            player_list[this_player_index].body_center_position.append(player_list[this_player_index].current_body_center_position)

                            # # update previous_feet_position
                            # player_list[this_player_index].current_img_position = player_list[this_player_index].previous_img_position
                            # player_list[this_player_index].previous_img_position = player_list[this_player_index].previous_img_position
                            # player_list[this_player_index].previous_model_position = player_list[this_player_index].previous_model_position

                        counter_json = counter_json + 1




#### use player list and one ball data to judge the gestures
# assume only one ball during the game

        # # pose data
        # self.wrists_positions = []
        # self.nose_position = []
        # self.body_center_position = []
        #
        # # relationship with ball
        # self.ball_in_hand = False
        # self.previous_hold_position = []


def match_player_with_ball(frame, trace, balls):

    global player_list

    # if court != None:
    #     previous_hold_model_position

    #### who is holding the ball?
    ## Either wrist is close the ball
    # change all other player's ball in hand to False and change this player's ball_in_hand = True
    # always record the position when ball is not in hands. previous_hold_position = [], which will be the shooting position

    # one_ball.position should be a list of positions in at this frame.

    min_ball_hand_distance = 2000
    ball_player_ID = str


    ## fail to detect ball or no one holds the ball, then keep the ball_in_hand same as before

    # find the minimum distance between hand and ball, find the player_id with the minimum distance.
    if len(balls.positions_at_frame) != 0:
        # print(len(balls.positions_at_frame))
        for one_ball_position in balls.positions_at_frame:
            # print(one_ball_position)
            for one_player in player_list:
                # reset the shooting_now variable:
                one_player.shooting_now = False
                for one_wrist_position in one_player.current_wrists_positions:
                    # print(one_wrist_position)
                    dist_hand_ball = distance(one_ball_position, one_wrist_position)
                    if dist_hand_ball < min_ball_hand_distance:
                        min_ball_hand_distance = dist_hand_ball
                        ball_player_ID = one_player.id

                    # print("draw the wrists!")
                    #
                    # cv2.circle(img=frame, center=(one_player.current_img_position[0],
                    #                               one_player.current_img_position[1]), radius=3,
                    #            color=(0, 0, 255), thickness=3)
                    # cv2.circle(img=trace, center=(one_player.current_img_position[0],
                    #                               one_player.current_img_position[1]), radius=3,
                    #            color=(0, 0, 255), thickness=3)

                    # display the player's wrist
                    # cv2.putText(frame, str("player's wrist!"), one_wrist_position,
                    #             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                    # cv2.putText(trace, str("player's wrist!"), one_wrist_position,
                    #             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)


    # change all other player's ball in hand to False and change this player's ball_in_hand = True
    if min_ball_hand_distance < 20:
        # print(min_ball_hand_distance)
        for one_player in player_list:
            if one_player.ball_in_hand == False and one_player.id == ball_player_ID:
                one_player.ball_in_hand = True
                one_player.previous_hold_position = one_player.current_img_position
                # one_player.previous_hold_model_position = court.transformed_img_2_model_point(one_player.previous_hold_position)

                # display the text
                cv2.putText(frame, str("{} is holding the ball".format(one_player.id)), (50,50),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)

            elif one_player.ball_in_hand == True and one_player.id != ball_player_ID:

                one_player.previous_hold_position = one_player.previous_img_position
                one_player.ball_in_hand = False

                # display the text
                cv2.putText(frame, str("{} stole the ball".format(one_player.id)), (50,50),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)


            elif one_player.ball_in_hand == True and one_player.id == ball_player_ID:

                one_player.previous_hold_position = one_player.current_img_position
                # one_player.previous_hold_model_position = court.transformed_img_2_model_point(one_player.current_img_position)

                # display the text
                cv2.putText(frame, str("* {} is holding the ball".format(one_player.id)), (50,50),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)

            else:
                one_player.ball_in_hand = False


    #### check ball.made_or_not if shot is made
    ## if made, then locate the player with ball_in_hand is True.
    # update the made number, miss number and the shoot position(previous_hold_position)

    if balls.made_or_not_at_frame == True:
        # find the right player
        this_player_index = next((index for index, player in enumerate(player_list) if player.ball_in_hand == True), None)
        player_list[this_player_index].shooting_now = True

        if this_player_index != None:
            player_list[this_player_index].statistics['made_position'].append(player_list[this_player_index].previous_hold_position)
            player_list[this_player_index].statistics['attempts'] += 1
            player_list[this_player_index].statistics['made'] += 1
            player_list[this_player_index].statistics['attempt_position'].append(
                player_list[this_player_index].previous_hold_position)

            # display the text
            cv2.putText(frame, str("Shot from here!"), (int(player_list[this_player_index].previous_hold_position[0]+50),int(player_list[this_player_index].previous_hold_position[1])),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(trace, str("Shot from here!"), (int(player_list[this_player_index].previous_hold_position[0]+50),int(player_list[this_player_index].previous_hold_position[1])),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)

            cv2.circle(img=frame, center=(int(player_list[this_player_index].previous_hold_position[0]),int(player_list[this_player_index].previous_hold_position[1])), radius=3,
                       color=(0, 255, 255), thickness=3)
            cv2.circle(img=trace, center=(int(player_list[this_player_index].previous_hold_position[0]),int(player_list[this_player_index].previous_hold_position[1])), radius=3,
                       color=(0, 255, 255), thickness=3)


    elif balls.missing_or_not_at_frame == True:
        # find the right player
        this_player_index = next((index for index, player in enumerate(player_list) if player.ball_in_hand == True),None)
        player_list[this_player_index].shooting_now = True

        if this_player_index != None:
            player_list[this_player_index].statistics['miss_position'].append(
                player_list[this_player_index].previous_hold_position)
            player_list[this_player_index].statistics['attempts'] += 1
            player_list[this_player_index].statistics['miss'] += 1
            player_list[this_player_index].statistics['attempt_position'].append(
                player_list[this_player_index].previous_hold_position)


            # display the text
            cv2.putText(frame, str("Miss from here!"), (player_list[this_player_index].previous_hold_position[0],
                                                        player_list[this_player_index].previous_hold_position[1]),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(trace, str("Miss from here!"), (player_list[this_player_index].previous_hold_position[0],
                                                        player_list[this_player_index].previous_hold_position[1]),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)

            cv2.circle(img=frame, center=(player_list[this_player_index].previous_hold_position[0],
                                          player_list[this_player_index].previous_hold_position[1]), radius=3,
                       color=(0, 255, 255), thickness=3)
            cv2.circle(img=trace, center=(player_list[this_player_index].previous_hold_position[0],
                                          player_list[this_player_index].previous_hold_position[1]), radius=3,
                       color=(0, 255, 255), thickness=3)




    combined = np.concatenate((frame, trace), axis=1)
    return combined, trace

    ## change all player's ball_in_hand to False



def display_trajectory_on_model(court, balls):

    global player_list
    # blank_image = np.zeros((height, width, 3), np.uint8)

    def array_2_int_turple(point_array):
        return (int(point_array[0]), int(point_array[1]))

    colors = [(0,0,255), (255,0,255)]
    model_image = court.court_model

    for player, clr in zip(player_list, colors):

        # skip the empty value

        # draw lines to connect current foot point and previous foot point
        compare = player.current_img_position == np.zeros([1,2],dtype='float32')
        equal_arrays = compare.all()

        if equal_arrays != True:
            player_model_position = court.transformed_img_2_model_point(player.current_img_position)

            # draw foot point
            cv2.circle(img=model_image, center=player_model_position, radius=3,
                       color=clr, thickness=3)

            # draw made position
            if player.shooting_now == True and balls.made_or_not_at_frame == True:

                # get player model position
                player_made_model_position = court.transformed_img_2_model_point(player.previous_hold_position)

                cv2.circle(img=model_image, center=player_made_model_position, radius=3,
                           color=(0, 255, 255), thickness=3)
                cv2.putText(model_image, str("{} Shot from here and made".format(player.id)), (player_made_model_position[0] + 50, player_made_model_position[1]),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)

            # draw missed position
            elif player.shooting_now == True and balls.missing_or_not_at_frame == True:

                # get player model position
                player_missed_model_position = court.transformed_img_2_model_point(player.previous_hold_position)

                cv2.circle(img=model_image, center=player_missed_model_position, radius=3,
                           color=(0, 255, 255), thickness=3)
                cv2.putText(model_image, str("{} Shot from here and missed".format(player.id)), (player_missed_model_position[0] + 50, player_missed_model_position[1]),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)

            # draw lines to connect current foot point and previous foot point
            compare = player.previous_img_position == np.zeros([1,2],dtype='float32')
            equal_arrays = compare.all()

            if equal_arrays != True:
                player_model_previous_position = court.transformed_img_2_model_point(player.previous_img_position)
                cv2.line(model_image, player_model_position, player_model_previous_position, color=clr, thickness=1, lineType=8)


    return model_image










### TO DO LATER: input a frame,  get ReID and openpose results , update player_list and player_dictionary
### TO DO LATER: input a frame,  get ReID and openpose results , update player_list and player_dictionary
### TO DO LATER: input a frame,  get ReID and openpose results , update player_list and player_dictionary

def openpose_init():
    try:
        if platform == "win32":
            sys.path.append('./OpenPose/Release')
            import pyopenpose as op
        else:
            sys.path.append('./OpenPose')
            from Release import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "./OpenPose/models"

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    return datum, opWrapper




# datum, opWrapper = openpose_init()

def read_openpose_update_player(frame, datum, opWrapper):

    def calculateAngle(a, b, c):
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return round(np.degrees(angle), 2)


    def get_openpose_data():

        # getting openpose keypoints
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])
        try:
            headX, headY, headConf = datum.poseKeypoints[0][0]
            handX, handY, handConf = datum.poseKeypoints[0][4]
            elbowAngle, kneeAngle, elbowCoord, kneeCoord = getAngleFromDatum(datum)
        except:
            print("Something went wrong with OpenPose")
            headX = 0
            headY = 0
            handX = 0
            handY = 0
            elbowAngle = 0
            kneeAngle = 0
            elbowCoord = np.array([0, 0])
            kneeCoord = np.array([0, 0])


    def getAngleFromDatum(datum):
        hipX, hipY, _ = datum.poseKeypoints[0][9]
        kneeX, kneeY, _ = datum.poseKeypoints[0][10]
        ankleX, ankleY, _ = datum.poseKeypoints[0][11]

        shoulderX, shoulderY, _ = datum.poseKeypoints[0][2]
        elbowX, elbowY, _ = datum.poseKeypoints[0][3]
        wristX, wristY, _ = datum.poseKeypoints[0][4]

        kneeAngle = calculateAngle(np.array([hipX, hipY]), np.array([kneeX, kneeY]), np.array([ankleX, ankleY]))
        elbowAngle = calculateAngle(np.array([shoulderX, shoulderY]), np.array([elbowX, elbowY]),
                                    np.array([wristX, wristY]))

        elbowCoord = np.array([int(elbowX), int(elbowY)])
        kneeCoord = np.array([int(kneeX), int(kneeY)])
        return elbowAngle, kneeAngle, elbowCoord, kneeCoord