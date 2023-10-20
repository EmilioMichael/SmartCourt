import time
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import os
import sys
import argparse
import matplotlib.pyplot as plta
from sys import platform
from scipy.optimize import curve_fit
from ball import balls, one_ball, tensorflow_init
from statistics import mean

from player import player_list, read_json_with_video, match_player_with_ball, display_trajectory_on_model

from court import the_court

tf.disable_v2_behavior()



##### start
##### start
##### start

video_file = "sample/test_8.mp4"
pre_model_corners=np.array([[10,10],
                      [990,10],
                      [990,930],
                      [10,930]], dtype='float32')

# import the court class
# initiate the court class here in this file

# import ball and player variables directly from other modules

court_model_filepath = './court/court_model.jpg'
this_court = the_court(court_model_filepath, model_corners=pre_model_corners)
img_corners, hoop = this_court.court_basket_calibration("./court", video_file_path = video_file)

balls_in_frame = balls()
one_ball_in_frame = one_ball()



os.environ['KMP_DUPLICATE_LIB_OK']='True'
resize_factor = 0.3

# datum, opWrapper = openpose_init()
detection_graph, image_tensor, boxes, scores, classes, num_detections = tensorflow_init()
frame_batch = 5

cap = cv2.VideoCapture(video_file)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# out = cv2.VideoWriter("sample/output_8.mp4", fourcc, fps / frame_batch, (int(width * 2 * resize_factor), int(height * resize_factor)))
out = cv2.VideoWriter("sample/output_12.mp4", fourcc, fps / frame_batch, (int(width * 2 * resize_factor), int(height * resize_factor)))
out_model = cv2.VideoWriter("sample/output_model_12.mp4", cv2.VideoWriter_fourcc(*'avc1'), fps / frame_batch, (this_court.court_model.shape[0], this_court.court_model.shape[1]))

trace = np.full((int(height), int(width), 3), 255, np.uint8)

# fig = plt.figure()

#### objects to store detection status
# shooting_result = {
#     'attempts': 0,
#     'made': 0,
#     'miss': 0,
#     'avg_elbow_angle': 0,
#     'avg_knee_angle': 0,
#     'avg_release_angle': 0,
#     'avg_ballInHand_time': 0
# }
# previous = {
# 'ball': np.array([0, 0]),  # x, y
# 'hoop': np.array([0, 0, 0, 0]),  # xmin, ymax, xmax, ymin
#     'hoop_height': 0
# }
# during_shooting = {
#     'isShooting': False,
#     'balls_during_shooting': [],
#     'release_angle_list': [],
#     'release_point': []
# }
# shooting_pose = {
#     'ball_in_hand': False,
#     'elbow_angle': 370,
#     'knee_angle': 370,
#     'ballInHand_frames': 0,
#     'elbow_angle_list': [],
#     'knee_angle_list': [],
#     'ballInHand_frames_list': []
# }
# shot_result = {
#     'displayFrames': 0,
#     'release_displayFrames': 0,
#     'judgement': ""
# }



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8


skip_count = 0
counter_index = 0
with tf.Session(graph=detection_graph, config=config) as sess:
    while True:
        ret, img = cap.read()
        if ret == False:
            break
        skip_count += 1
        if(skip_count < frame_batch):
            continue
        skip_count = 0

        # for reading the json files
        counter_index += 1
        frame_index = counter_index * frame_batch + 3

        # only for testing
        if frame_index > 484:
            break

        # detection, trace = detect_shot(img, trace, width, height, sess, image_tensor, boxes, scores, classes,
        #                                 num_detections, previous, during_shooting, shot_result, fig, shooting_result, datum, opWrapper, shooting_pose)

        #### balls mode
        # detection, trace = balls_in_frame.detect_ball(img, trace, width, height, sess, image_tensor, boxes, scores, classes, num_detections, hoop)

        # detection, trace = one_ball_in_frame.update_one_ball(img, trace, sess, boxes, scores, classes, num_detections, image_tensor)


        #### one_ball mode
        # updata one_ball class
        # one_ball_in_frame...
        frame_new, trace_ball = balls_in_frame.detect_ball(img, trace, width, height, sess, image_tensor, boxes, scores, classes, num_detections, hoop)

        # update players YUANDA and My player class
        read_json_with_video(frame_index, "/Users/WeiJB/Desktop/Harvard_MDE/SmartCourt/basketball_detection/basketball-shot-detection-master-2/sample/result/json", this_court)
        # player and ball relationship detection and update player statistics
        detection, trace_wrist = match_player_with_ball(frame_new, trace_ball, balls_in_frame)

        # plot the trajectory on the model image
        trajectory = display_trajectory_on_model(this_court, balls_in_frame)


        detection = cv2.resize(detection, (0, 0), fx=resize_factor, fy=resize_factor)
        cv2.imshow("detection", detection)
        cv2.imshow("trajectory", trajectory)
        out.write(detection)
        out_model.write(trajectory)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()


print(player_list[0].statistics)
print(player_list[1].statistics)

# getting average shooting angle
# getting average shooting angle
# getting average shooting angle

# shooting_result['avg_elbow_angle'] = round(mean(shooting_pose['elbow_angle_list']), 2)
# shooting_result['avg_knee_angle'] = round(mean(shooting_pose['knee_angle_list']), 2)
# shooting_result['avg_release_angle'] = round(mean(during_shooting['release_angle_list']), 2)
# shooting_result['avg_ballInHand_time'] = round(mean(shooting_pose['ballInHand_frames_list']) * (frame_batch / fps), 2)
#
# print("avg", shooting_result['avg_elbow_angle'])
# print("avg", shooting_result['avg_knee_angle'])
# print("avg", shooting_result['avg_release_angle'])
# print("avg", shooting_result['avg_ballInHand_time'])

# print(balls_in_frame.statistic['attempts'], balls_in_frame.statistic['miss'])

print(one_ball_in_frame.statistic['attempts'], one_ball_in_frame.statistic['miss'])

# plt.title("Trajectory Fitting", figure=fig)
# plt.ylim(bottom=0, top=height)

# trajectory_path = os.path.join(os.getcwd(), "trajectory_fitting.jpg")
# fig.savefig(trajectory_path)
# fig.clear()

# trace_path = os.path.join(os.getcwd(), "basketball_trace.jpg")
# cv2.imwrite(trace_path, trace)
