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

tf.disable_v2_behavior()




def fit_func(x, a, b, c):
    return a * (x ** 2) + b * x + c


def trajectory_fit(balls, height, width, shotJudgement, fig):
    x = []
    y = []
    for ball in balls:
        x.append(ball[0])
        y.append(height - ball[1])

    try:
        params = curve_fit(fit_func, x, y)
        [a, b, c] = params[0]
    except:
        print("fiiting error")
        a = 0
        b = 0
        c = 0
    x_pos = np.arange(0, width, 1)
    y_pos = []
    for i in range(len(x_pos)):
        x_val = x_pos[i]
        y_val = (a * (x_val ** 2)) + (b * x_val) + c
        y_pos.append(y_val)

    if (shotJudgement == "MISS"):
        plt.plot(x, y, 'ro', figure=fig)
        plt.plot(x_pos, y_pos, linestyle='-', color='red',
                 alpha=0.4, linewidth=5, figure=fig)
    else:
        plt.plot(x, y, 'go', figure=fig)
        plt.plot(x_pos, y_pos, linestyle='-', color='green',
                 alpha=0.4, linewidth=5, figure=fig)


def distance(x, y):
    return ((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2) ** (1 / 2)

def point_in_rectangle(xmin, ymin, xmax, ymax, x, y):
    if (x > xmin and x < xmax and y > ymin and y < ymax):
        return True
    else:
        return False



def tensorflow_init():
    MODEL_NAME = 'inference_graph'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    return detection_graph, image_tensor, boxes, scores, classes, num_detections


def calculateAngle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return round(np.degrees(angle), 2)



# when many balls in the court. only detect when the shot is made, scored or missed.
# This is for trimming the video to get replay

class balls():
    def __init__(self):
        self.status = "dribbling"
        self.direction = float
        self.position = [] # should be a list of lists
        self.position_path = [] # should be a list of lists
        self.positions_at_frame = []
        self.last_position = []

        # present frame
        self.detected_balls = []
        self.detected_ball_number = 0
        self.shooting_balls = 0
        self.dribble_balls = 0
        self.made_balls = 0
        self.balls_in_basket = 0

        self.made_or_not_at_frame = False
        self.missing_or_not_at_frame = False

        self.diff_balls  = 0

        # last frame
        self.previous_shooting = 0
        self.previous_balls_in_basket = 0
        self.previous_detected_ball_number = 0
        self.previous_dribble = 0

        # should change according to the fps of the streaming video
        self.basket_freezing_time = 0

        self.statistic = {
            'attempts': 0,
            'made': 0,
            'miss': 0,
        }

    def detect_balls(self, frame, sess, boxes, scores, classes, num_detections, image_tensor):


        frame_expanded = np.expand_dims(frame, axis=0)

        # main tensorflow detection, detect all videos
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # probability threshold
        prob_thre = 0.8
        # detected_balls
        ball_indice = np.where(scores[0] > prob_thre)[0]
        self.detected_ball_number = classes[0][ball_indice].tolist().count(1)



    def detect_ball(self, frame, trace, width, height, sess, image_tensor, boxes, scores, classes, num_detections, hoop):

        frame_expanded = np.expand_dims(frame, axis=0)

        # hoop = court_class.img_basket

        self.positions_at_frame = []
        self.made_or_not_at_frame = False
        self.missing_or_not_at_frame = False

        # main tensorflow detection, detect all videos
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: frame_expanded})


        self.previous_shooting = self.shooting_balls
        self.previous_balls_in_basket = self.balls_in_basket
        self.previous_detected_ball_number = self.detected_ball_number
        self.previous_dribble = self.dribble_balls

        # print(type(classes[0]))

        # probability threshold
        prob_thre = 0.8

        # detected_balls
        ball_indice = np.where(scores[0] > prob_thre)[0]
        self.detected_ball_number = classes[0][ball_indice].tolist().count(1)
        self.dribble_balls = 0
        self.shooting_balls = 0
        self.balls_in_basket = 0

        """
        # # ball in basket time
        # if self.previous_basket_balls > 0:
        #     self.made_balls = self.made_balls
        # else:
        #     self.made_balls = 0



        # # basket_freezing_time
        # if self.basket_freezing_time > 0:
        #     self.basket_freezing_time -= 1
        #     # when the basket zone freezes, remember the made_balls
        #     self.made_balls = self.made_balls
        # else:
        #     # otherwise, reset the made_balls
        #     self.made_balls = 0
        """

        # iterate through all the boxes
        for i, box in enumerate(boxes[0]):

            # display the basket
            cv2.rectangle(frame, (hoop['xmin'], hoop['ymax']),
                          (hoop['xmax'], hoop['ymin']), (80, 124, 0), 5)
            cv2.rectangle(trace, (hoop['xmin'], hoop['ymax']),
                          (hoop['xmax'], hoop['ymin']), (80, 124, 0), 5)

            if (scores[0][i] > prob_thre):
                ymin = int((box[0] * height))
                xmin = int((box[1] * width))
                ymax = int((box[2] * height))
                xmax = int((box[3] * width))
                xCoor = int(np.mean([xmin, xmax]))
                yCoor = int(np.mean([ymin, ymax]))


                # record position of the ball
                ball_position = np.array([xCoor, yCoor])
                self.positions_at_frame.append(ball_position)
                self.last_position = ball_position


                # draw yellow circles to all the basketball in the video
                # draw yellow circles to all the basketball in the video
                # draw yellow circles to all the basketball in the video
                if (classes[0][i] == 1):  # Basketball (not head)

                    # calculate ball radius
                    ball_radius = (ymax - ymin + xmax - xmin) / 2

                    # display the probability
                    cv2.putText(frame, str(scores[0][i]), (xCoor - 65, yCoor - 65),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),3)

                    # if the ball is in the basket window
                    ball_in_basket = point_in_rectangle(hoop['xmin'], hoop['ymin'], hoop['xmax'], hoop['ymax'],
                                                        xCoor, yCoor)

                    # divide the whole space into three areas, the ball will be colored differently.
                    # divide the whole space into three areas, the ball will be colored differently.
                    # divide the whole space into three areas, the ball will be colored differently.

                    # if there are a shoot or not
                    if yCoor < hoop['ymin']:
                        self.status = "shooting"
                        print("shooting!")
                        self.shooting_balls += 1
                        cv2.circle(img=frame, center=(xCoor, yCoor), radius=7,
                                   color=(255, 219, 0), thickness=3)
                        cv2.circle(img=trace, center=(xCoor, yCoor), radius=7,
                                   color=(255, 154, 0), thickness=3)

                    # shooting judgement, only capture the in-basket moment
                    # if self.status == "shooting" and self.basket_freezing_time == 0:
                    elif ball_in_basket == True:
                        self.balls_in_basket += 1

                        if self.previous_balls_in_basket == 0:

                            self.statistic['attempts'] += 1
                            self.statistic['made'] += 1
                            self.status = "dribble"

                            # should change according to the fps of the streaming video
                            # self.basket_freezing_time = 5
                            self.made_balls += 1
                            self.made_or_not_at_frame = True
                            print("made!!!")
                            # display the probability
                            cv2.putText(frame, str("Score!"), (xCoor - 20, yCoor - 20),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                            cv2.putText(trace, str("Score!"), (xCoor - 20, yCoor - 20),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)

                        cv2.circle(img=frame, center=(xCoor, yCoor), radius=7,
                                   color=(0, 0, 255), thickness=3)
                        cv2.circle(img=trace, center=(xCoor, yCoor), radius=7,
                                   color=(0, 0, 255), thickness=3)


                    else:
                        self.dribble_balls += 1

                        cv2.circle(img=frame, center=(xCoor, yCoor), radius=7,
                                   color=(100, 100, 0), thickness=3)
                        cv2.circle(img=trace, center=(xCoor, yCoor), radius=7,
                                   color=(100, 100, 0), thickness=3)

                        """
                        # # another missing or made judgement method
                        # # made (should have the same standard as the ball in the basket window)
                        # if (ymax >= (hoop['ymean']) and (distance([xCoor, yCoor], (hoop['xmean'], hoop['ymean'])) < ball_radius)):
                        #     self.statistic['attempts'] += 1
                        #     self.statistic['made'] += 1
                        #     self.status = "dribble"
                        #
                        #     # should change according to the fps of the streaming video
                        #     # self.basket_freezing_time = 5
                        #     self.made_balls += 1
                        #     print("made!!!")
                        """

            # update the self.statistics
            # update the self.statistics
            # update the self.statistics


        if self.shooting_balls == 0:
            self.status = "dribble"

        # add current position list to path
        self.position_path.append(self.positions_at_frame) # should be a list of lists

        self.diff_balls = self.previous_detected_ball_number - self.detected_ball_number

        # print(self.previous_shooting, self.previous_detected_ball_number, self.detected_ball_number, self.balls_in_basket,
        #       self.shooting_balls, self.diff_balls)

        # if the shot scores
        if self.previous_balls_in_basket - self.balls_in_basket == -1:
            made_balls = 1
        else:
            made_balls = 0

        if self.diff_balls != 0:
            missing_balls = self.previous_shooting - made_balls - self.shooting_balls - abs(self.diff_balls)
        else:
            missing_balls = self.previous_shooting - made_balls - self.shooting_balls

        # 在最严格的条件下，能检测到missing就记上去，检测不到就算了
        if missing_balls > 0:

            self.statistic['attempts'] += missing_balls
            self.statistic['miss'] += missing_balls
            self.missing_or_not_at_frame = True
            print("missing!!! * {}".format(missing_balls))

            # display the probability
            cv2.putText(frame, str("Missing!"), (100, 100),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(trace, str("Missing!"), (100, 100),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)


        # combined = np.concatenate((frame, trace), axis=1)
        return frame, trace


# when only one ball in the court. This ball will be matched with players.
# this is for recording players' statistics and game score.
class one_ball():
    def __init__(self):
        self.status = "dribbling"
        self.direction = float
        self.position = []
        self.last_position = []

        # present frame
        self.detected_balls = []
        self.detected_ball_number = 0
        self.shooting_balls = 0
        self.dribble_balls = 0
        self.made_balls = 0
        self.balls_in_basket = 0

        self.diff_balls  = 0

        # last frame
        self.previous_shooting = 0
        self.previous_balls_in_basket = 0
        self.previous_detected_ball_number = 0
        self.previous_dribble = 0

        # should change according to the fps of the streaming video
        self.basket_freezing_time = 0

        self.statistic = {
            'attempts': 0,
            'made': 0,
            'miss': 0,
        }


    def update_one_ball(self, frame, trace, sess, boxes, scores, classes, num_detections, image_tensor):

        frame_expanded = np.expand_dims(frame, axis=0)

        # main tensorflow detection, detect this frame
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        prob_thre = 0.8

        height = frame.shape[0]
        width = frame.shape[1]

        # iterate through all the boxes to get the detected ball
        for i, box in enumerate(boxes[0]):

            if (scores[0][i] > prob_thre):
                ymin = int((box[0] * height))
                xmin = int((box[1] * width))
                ymax = int((box[2] * height))
                xmax = int((box[3] * width))
                xCoor = int(np.mean([xmin, xmax]))
                yCoor = int(np.mean([ymin, ymax]))

                # record position of the ball
                ball_position = np.array([xCoor, yCoor])
                self.position.append(ball_position)
                self.last_position = ball_position


                cv2.circle(img=frame, center=(xCoor, yCoor), radius=7,
                           color=(0, 0, 255), thickness=3)
                cv2.circle(img=trace, center=(xCoor, yCoor), radius=7,
                           color=(0, 0, 255), thickness=3)


        # combined = np.concatenate((frame, trace), axis=1)
        return frame, trace