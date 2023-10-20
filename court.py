import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib import cm
import copy
# from select_points import clicker_class

import matplotlib
matplotlib.use('TkAgg')

# import draw_rectangle_on_images
from scipy.spatial import distance

from player import player_list
from math import pi

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables

    global refPt, is_drawing, img, cahe
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed


    if event == cv2.EVENT_LBUTTONDOWN:
        # clean the img
        img = copy.deepcopy(cache)
        refPt = [(x, y)]
        is_drawing = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
    # record the ending (x, y) coordinates and indicate that
    # the cropping operation is finished
        refPt.append((x, y))
        is_drawing = False
        # draw a rectangle around qthe region of interest
        cv2.rectangle(img, refPt[0], refPt[1], color = (0, 255, 0), thickness = 2)

        print("press q to save the boundary or reselect the basket")

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            img = copy.deepcopy(cache)
            cv2.rectangle(img=img, pt1=refPt[0], pt2=(x, y), color=(255, 0, 0), thickness=1)


def execute_court_basket_marking(video_file_path=None, image_file_path=None):

    global  img, refPt, cache
    refPt = []
    if video_file_path != None:
        cap = cv2.VideoCapture(video_file_path)
        success, image = cap.read()

        if success:
            # cv2.imwrite("first_frame.jpg", image)
            img = image
    else:
        img = cv2.imread(image_file_path)

    cv2.namedWindow("basket")
    cache = copy.deepcopy(img)
    # cache = copy.deepcopy(img)

    while True:
        cv2.imshow("basket", img)
        cv2.setMouseCallback("basket", click_and_crop)

        # if the 'r' key is pressed, reset the cropping region
        if cv2.waitKey(1) & 0xFF == ord("r"):
            img = copy.deepcopy(cache)

        # wait 5 seconds then wait for ESC key
        if cv2.waitKey(5) & 0xFF == ord("q"): # == 27 means esc
            break

    cv2.destroyAllWindows()

    # print(refPt)

    hoop_coordinate = {
        'xmin': min(refPt[0][0], refPt[1][0]),
        'xmax': max(refPt[0][0], refPt[1][0]),
        'ymin': min(refPt[0][1], refPt[1][1]),
        'ymax': max(refPt[0][1], refPt[1][1]),
        'xmean': sum((refPt[0][0], refPt[1][0]))/2,
        'ymean': sum((refPt[0][1], refPt[1][1]))/2
    }
    return hoop_coordinate




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
            angle = angle + calculate_angle(point_array[i, :], p, point_array[i+1, :])
        else:
            angle = angle + calculate_angle(point_array[i, :], p, point_array[0, :])
    lower_bound = 2 * pi * 0.95
    upper_bound = 2 * pi * 1.05
    if (angle > lower_bound) & (angle < upper_bound):
        return True
    else:
        return False




class clicker_class(object):
    def __init__(self, ax, pix_err=1, point_number=4):
        self.canvas = ax.get_figure().canvas
        self.cid = None
        self.pt_lst = []
        self.pt_plot = ax.plot([], [], marker='o',linestyle='none', zorder=5, color='r')[0]
        self.pix_err = pix_err
        self.connect_sf()
        self.point_number = point_number

    def set_visible(self, visible):
        '''sets if the curves are visible '''
        self.pt_plot.set_visible(visible)

    def clear(self):
        '''Clears the points'''
        self.pt_lst = []
        self.redraw()

    def connect_sf(self):
        if self.cid is None:
            self.cid = self.canvas.mpl_connect('button_press_event',
                                               self.click_event)

    def disconnect_sf(self):
        if self.cid is not None:
            self.canvas.mpl_disconnect(self.cid)
            self.cid = None

    def click_event(self, event):
        ''' Extracts locations from the user'''

        if event.key == 'shift':
            self.pt_lst = []
            return
        if event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            self.pt_lst.append((event.xdata, event.ydata))
        elif event.button == 3:
            self.remove_pt((event.xdata, event.ydata))

        # guidance
        if (self.point_number - len(self.pt_lst)) > 0:
            print('{} points left, right lick to remove the point'.format(self.point_number - len(self.pt_lst) ))
        elif (self.point_number - len(self.pt_lst)) < 0:
            print('{} points more, right lick to remove the point'.format(len(self.pt_lst)-self.point_number))
        else:
            print('0 points left, please press {} to quit'.format("q"))

        self.redraw()

    def remove_pt(self, loc):
        if len(self.pt_lst) > 0:
            dist_array = list(map(lambda x: np.sqrt((x[0] - loc[0]) ** 2 + (x[1] - loc[1]) ** 2), self.pt_lst))
            index = np.argmin(dist_array)
            self.pt_lst.pop(index)

    def redraw(self):
        if len(self.pt_lst) > 0:
            x, y = zip(*self.pt_lst)
        else:
            x, y = [], []
        self.pt_plot.set_xdata(x)
        self.pt_plot.set_ydata(y)

        self.canvas.draw()

    def return_points(self):
        '''Returns the clicked points in the format the rest of the
        code expects'''
        return (np.vstack(self.pt_lst)).astype('float32')



class the_court():
    def __init__(self, court_model_filepath, court_img_filepath= "Not given", model_corners="Not given"):

        if court_img_filepath != "Not given":
            self.court_filepath = court_img_filepath
            self.court_img = cv2.cvtColor(cv2.imread(court_img_filepath), cv2.COLOR_BGR2RGB)
        else:
            self.court_filepath = []


        self.court_model = cv2.cvtColor(cv2.imread(court_model_filepath), cv2.COLOR_BGR2RGB)

        self.img_caliP = np.zeros([4, 2], dtype='float32')
        self.model_caliP = np.zeros([4, 2], dtype='float32')

        if model_corners != "Not given":
            self.model_corners = model_corners
        else:
            self.model_corners = None
            # self.model_corners = np.zeros([4, 2], dtype='float32')  # model_corners,  np.zero([2,2], dtype='float32')

        self.img_corners = np.zeros([4, 2], dtype='float32')

        self.H = np.zeros([3, 3], dtype='float32')

        self.img_basket = {
            'xmin': 0,
            'xmax': 0,
            'ymin': 0,
            'ymax': 0,
            'xmean': 0,
            'ymean': 0

        }

    # Convert from inhomogeneous to homogeneous coordinates
    def in2hom(X):
        return np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)

    # Convert from homogeneous to inhomogeneous coordinates
    def hom2in(X):
        return X[:, :2] / X[:, 2:]

    def in2hom2(X):
        X_new = X[:, np.newaxis, :]
        return X_new

    def hom22in(X):
        # delete a dimension?
        return X[:, 0, :]

    def court_img_corner(self, homogenous_matrix):
        model_corners = self.model_corners[:, np.newaxis, :]
        court_img_corner = cv2.perspectiveTransform(model_corners, homogenous_matrix)
        return court_img_corner[:, 0, :]

    def get_img_calibrationPoint(self):
        # interactively select point

        print("please select 4 points in clockwise consequence")
        fig, ax = plt.subplots()
        ax.imshow(self.court_img)
        point_array = clicker_class(ax)
        plt.show()

        sorted_point = self.sortpts_clockwise(point_array.return_points())
        self.img_caliP = sorted_point
        # self.img_caliP = point_array.return_points()


    def get_model_calibrationPoint(self):
        # interactively select point
        print("please select 4 points in the same sequence as in the court image")
        fig, ax = plt.subplots()
        ax.imshow(self.court_model)
        point_array = clicker_class(ax)
        plt.show()

        sorted_point = self.sortpts_clockwise(point_array.return_points())
        self.model_caliP = sorted_point

        # self.model_caliP = point_array.return_points()

        # self.model_caliP = np.array([[65, 9],
        #                              [938, 9],
        #                              [938, 286],
        #                              [65, 295]], dtype='float32')

    def get_model_corners(self):
        # interactively select point

        fig, ax = plt.subplots()
        ax.imshow(self.court_model)
        point_array = clicker_class(ax)
        plt.show()

        self.model_corners = point_array.return_points()


        # self.img_caliP = np.array([[158, 617],
        #                            [1637, 611],
        #                            [1735, 766],
        #                            [20, 784]], dtype='float32')


    def get_img_basket(self):
        # interactively select point

        basket_point = np.array([[65, 9],
                                 [938, 9],
                                 [938, 286],
                                 [65, 295]], dtype='float32')

        # self.img_basket['xmin'] =
        # self.img_basket['xmax'] =
        # self.img_basket['ymin'] =
        # self.img_basket['ymax'] =
        # self.img_basket['xmean'] =
        # self.img_basket['ymean'] =

    def transformed_img_2_model_point(self, image_p):
        # transformed to model coordinates:
        image_2d_array = np.array([image_p])
        image_position = image_2d_array[:, np.newaxis, :]

        # print(image_position)

        # finally, get the mapping
        model_position = cv2.perspectiveTransform(image_position, self.H)
        # print(model_position)
        return tuple(model_position[0, 0, :].astype(int))

    def homography_matrix(self):
        # Enter the corner coordinates in this numpy array, one (x,y) per row. Note the
        # order you use (we suggest clockwise order from document top-left).
        src_corners = self.img_caliP
        out_corners = self.model_caliP

        H, _ = cv2.findHomography(src_corners, out_corners)
        self.H = H

        H_inv, _ = cv2.findHomography(out_corners, src_corners)
        self.img_corners = self.court_img_corner(H_inv)
        # self.img_corners = court_img_corner(np.linalg.inv(H), self.model_corners)
        # return H



    def sortpts_clockwise(self, A):
        # Sort A based on Y(col-2) coordinates
        sortedAc2 = A[np.argsort(A[:, 1]), :]

        # Get top two and bottom two points
        top2 = sortedAc2[0:2, :]
        bottom2 = sortedAc2[2:, :]

        # Sort top2 points to have the first row as the top-left one
        sortedtop2c1 = top2[np.argsort(top2[:, 0]), :]
        top_left = sortedtop2c1[0, :]

        # Use top left point as pivot & calculate sq-euclidean dist against
        # bottom2 points & thus get bottom-right, bottom-left sequentially
        sqdists = distance.cdist(top_left[None], bottom2, 'sqeuclidean')
        rest2 = bottom2[np.argsort(np.max(sqdists, 0))[::-1], :]

        # Concatenate all these points for the final output
        return np.concatenate((sortedtop2c1, rest2), axis=0)



    # mainly for visualize the trajectories, not for live stream analysis
    def generate_foot_point_sequence(self, file_path):

        global player_list

        total_frames = 484
        foot_x_index = [33, 42, 57, 66]
        foot_y_index = [34, 43, 58, 67]
        probability_index = [35, 44, 59, 68]

        counter_json = 0
        time_frame_list = []

        person_list = []

        for frame_index in range(0, total_frames):
            if frame_index % 10 == 0:

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
                                feet_position_x = np.mean([flat_pose_list[i] for i in foot_x_index])
                                feet_position_y = np.mean([flat_pose_list[j] for j in foot_y_index])
                                # feet_position_x = np.mean([pose_keypoints_2d[i] for i in foot_x_index])
                                # feet_position_y = np.mean([pose_keypoints_2d[j] for j in foot_y_index])
                                feet_position = np.array([[feet_position_x, feet_position_y]], dtype='float32')

                                # print(feet_position[0,:])

                                # condition, the feet position should be inside the court
                                if determine_if_p_in_quanlilateral(self.img_corners, feet_position[0,:]) == True:
                                    # print("inside the court")

                                    # transformed to model coordinates:
                                    feet_image_positions = feet_position[:, np.newaxis, :]

                                    # finally, get the mapping
                                    feet_model_position = cv2.perspectiveTransform(feet_image_positions, self.H)
                                    # print(feet_model_position[0, 0, :])

                                    # player_list[this_player_index].img_path.append(feet_position[0, :])
                                    # player_list[this_player_index].model_path.append(feet_model_position[0, 0, :])
                                    # player_list[this_player_index].time_frame.append(frame_index)

                                    if len(player_list[this_player_index].time_frame) == 0:

                                        player_list[this_player_index].previous_img_position = feet_position[0,:]
                                        dist = 0

                                    else:
                                        dist = np.linalg.norm(feet_position[0,:] - player_list[this_player_index].previous_img_position) / (player_list[this_player_index].skip_frames + 1)
                                        print(dist)

                                    if dist < 50:
                                        print("added")

                                        player_list[this_player_index].img_path.append(feet_position[0,:])
                                        player_list[this_player_index].model_path.append(feet_model_position[0, 0, :])
                                        player_list[this_player_index].time_frame.append(frame_index)
                                        player_list[this_player_index].skip_frames = 0

                                        # update feet_position
                                        player_list[this_player_index].previous_img_position = feet_position[0,:]
                                    else:
                                        player_list[this_player_index].skip_frames = player_list[this_player_index].skip_frames + 1

                                    counter_json = counter_json + 1


                        # test only select the first person
                        # test only select the first person
                        # test only select the first person
                        # person = people[0]
                        # pose_keypoints_2d = person["pose_keypoints_2d"]
                        # feet_position_x = np.mean([pose_keypoints_2d[i] for i in foot_x_index])
                        # feet_position_y = np.mean([pose_keypoints_2d[j] for j in foot_y_index])
                        # feet_position = np.array([[feet_position_x, feet_position_y]], dtype='float32')
                        #
                        # # condition, the feet position should be inside the court
                        # if (feet_position > court_img_corner[0, :]).all() and (
                        #     feet_position < court_img_corner[2, :]).all():
                        #
                        #     if counter_json == 0:
                        #         foot_positions = feet_position
                        #     else:
                        #         foot_positions = np.concatenate([foot_positions, feet_position], axis=0)
                        #
                        #     counter_json = counter_json + 1
                        #     time_frame_list.append(frame_index)
                        #
                        # else:
                        #     pass
            else:
                pass

        # # cv2.perspectiveTransform(pointsIn, H) to have shape [numpoints,1,numdims]
        # foot_image_positions = foot_positions[:, np.newaxis, :]
        #
        # # finally, get the mapping
        # foot_model_position = cv2.perspectiveTransform(foot_image_positions, self.H)
        #
        # return foot_image_positions, foot_model_position, time_frame_list

    def plot_point_oncourt(self, img_point, model_point):
        # if you want to plot the calibration points
        # plot_point_oncourt(self.img_caliP, self.model_caliP)

        # display the results
        plt.figure(figsize=(20, 10))
        plt.subplot(121)
        plt.imshow(self.court_img)
        plt.axis('off')
        plt.plot(img_point[:, 0], img_point[:, 1], 'go', markersize=12)
        plt.subplot(122)
        plt.imshow(self.court_model)
        plt.plot(model_point[:, 0], model_point[:, 1], 'go', markersize=12)
        plt.axis('off')

        plt.show()


    def plot_trajectory(self, player_list):
        # # cv2.perspectiveTransform(pointsIn, H) to have shape [numpoints,1,numdims]
        # foot_positions = foot_points[:, np.newaxis, :]

        # # finally, get the mapping
        # pointsOut = cv2.perspectiveTransform(foot_points, homograpy_matrix)

        colors = cm.rainbow(np.linspace(0, 1, len(player_list)))

        # display the results
        plt.figure(figsize=(20, 10))
        plt.subplot(121)
        plt.imshow(self.court_img)
        plt.axis('off')

        for player, clr in zip(player_list, colors):
            img_point_array = np.vstack(player.img_path)
            plt.plot(img_point_array[:, 0], img_point_array[:, 1], 'go', c=clr, markersize=2)

        plt.subplot(122)
        plt.imshow(self.court_model)
        plt.axis('off')

        for player, clr in zip(player_list, colors):
            model_point_array = np.vstack(player.model_path)
            plt.plot(model_point_array[:, 0], model_point_array[:, 1], 'go',c=clr, markersize=2)
            # connect points as lines
            plt.plot(model_point_array[:, 0], model_point_array[:, 1], c=clr)



        # # connect points to lines
        # plt.plot(pointsOut[0:-1, 0:-1, 0], pointsOut[1:, 1:, 0])

        plt.show()



    def court_basket_calibration(self, working_dir, courtimg_file_path=None, video_file_path=None):
        if video_file_path != None:
            cap = cv2.VideoCapture(video_file_path)
            success, image = cap.read()

            if success:
                cv2.imwrite(working_dir + "/first_frame.jpg", image)
                self.court_filepath = working_dir + "/first_frame.jpg"
                self.court_img = cv2.cvtColor(cv2.imread(self.court_filepath), cv2.COLOR_BGR2RGB)

        else:
            self.court_filepath = courtimg_file_path

        if self.model_corners is None:
            self.get_model_corners()

        self.get_img_calibrationPoint()

        self.get_model_calibrationPoint()

        self.homography_matrix()

        self.img_basket = execute_court_basket_marking(video_file_path=video_file_path,
                                                                                   image_file_path=courtimg_file_path)

        self.plot_point_oncourt(self.img_corners, self.model_corners)

        # print (court_1.img_corners)
        #
        # json_file_path = "/Users/WeiJB/Desktop/Harvard_MDE/SmartCourt/json"
        # court_1.generate_foot_point_sequence(json_file_path)
        #
        # print(player_list[0].img_path)
        #
        # court_1.plot_trajectory(player_list)

        return self.img_corners, self.img_basket

        # foot_points, time_sequence = court_1.generate_foot_point_sequence(file_path, 0, court_img_corner)
        # print(foot_points.shape)







def dependent_court_basket_calibration(court_model_filepath, working_dir, courtimg_file_path=None, video_file_path=None, preset_model_corner="Not given"):


    if video_file_path != None:
        cap = cv2.VideoCapture(video_file_path)
        success, image = cap.read()

        if success:
            cv2.imwrite(working_dir + "/first_frame.jpg", image)
            court_img_filepath = working_dir + "/first_frame.jpg"

    else:
        court_img_filepath = courtimg_file_path


    if preset_model_corner != "Not given":
        court_1 = the_court(court_model_filepath, court_img_filepath=court_img_filepath, model_corners=preset_model_corner)
        print("Coners ready")

    else:
        court_1 = the_court(court_img_filepath, court_model_filepath)
        court_1.get_model_corners()

    court_1.get_img_calibrationPoint()

    court_1.get_model_calibrationPoint()

    court_1.homography_matrix()

    court_1.img_basket = execute_court_basket_marking(video_file_path=video_file_path, image_file_path=courtimg_file_path)

    court_1.plot_point_oncourt(court_1.img_corners, court_1.model_corners)

    # print (court_1.img_corners)

    json_file_path = "/Users/WeiJB/Desktop/Harvard_MDE/SmartCourt/basketball_detection/basketball-shot-detection-master-2/sample/result/json"
    court_1.generate_foot_point_sequence(json_file_path)

    # print(player_list[0].img_path)

    court_1.plot_trajectory(player_list)

    return court_1.img_corners, court_1.img_basket

    # foot_points, time_sequence = court_1.generate_foot_point_sequence(file_path, 0, court_img_corner)
    # print(foot_points.shape)


# test
# test
# test


"""

pre_model_corners=np.array([[10,10],
                      [990,10],
                      [990,930],
                      [10,930]], dtype='float32')


# court_img_filepath = './court/court_img_2.png'
court_model_filepath = './court/court_model.jpg'

a, b = dependent_court_basket_calibration(court_model_filepath, "./court", video_file_path = "/Users/WeiJB/Desktop/Harvard_MDE/SmartCourt/basketball_detection/basketball-shot-detection-master-2/sample/test_8.mp4", preset_model_corner=pre_model_corners)
print(a, b, player_A.img_path, player_A.model_path)


# plot_trajectory(court_img, json_file_path, foot_points, H, time_sequence)

# """