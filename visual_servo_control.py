#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
from PIL import Image
import argparse
import sys

import gym
import numpy as np
import pyglet
from pyglet.window import key
import cv2
from gym_duckietown.envs import DuckietownEnv

# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default=None)
parser.add_argument("--map-name", default="udem1")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--camera_rand", default=False, action="store_true")
parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
parser.add_argument("--seed", default=1, type=int, help="seed")
args = parser.parse_args()

if args.env_name and args.env_name.find("Duckietown") != -1:
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
        camera_rand=args.camera_rand,
        dynamics_rand=args.dynamics_rand,
        accept_start_angle_deg=20,
        full_transparency=True
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()


class Trajectory:
    def __init__(self):
        self.distance_to_target = np.inf
        self.angle_to_target = 0.0
        self.angle_to_goal_pose = 0.0
        self.last_update_time = None
        self.last_commands = None
        self.done = False
        self.target_in_sight = False

    def update(self, relative_pose):
        if not self.done:
            x_dist = relative_pose[0][2]
            y_dist = relative_pose[0][0]
            self.distance_to_target = np.sqrt(x_dist**2 + y_dist**2)
            print(x_dist, y_dist, self.distance_to_target)
            # TODO real angle is 90 - arctan(y_dist / x_dist)

            # Temporary solution before conversion of relative pose
            #self.angle_to_target = relative_pose[1]
            # if x_dist > 0:
            #     self.angle_to_target = -90 + relative_pose[1] + np.rad2deg(np.arctan(y_dist / x_dist))
            # else:
            #     self.angle_to_target = 90 + relative_pose[1] + np.rad2deg(np.arctan(y_dist / x_dist))
            # self.angle_to_goal_pose = relative_pose[1]
            self.angle_to_target = - np.rad2deg(np.arctan(y_dist/x_dist))
            self.angle_to_goal_pose = relative_pose[1]
            self.target_in_sight = True

    def get_commands(self):
        v = 0.0
        w = 0.0
        if not self.done:
            if np.abs(self.distance_to_target) < 0.01:
                print("close to goal")
                print("angle_to_goal", self.angle_to_goal_pose)
                if np.abs(self.angle_to_goal_pose) < 1:
                    print("reached goal")
                    self.done = True
                elif self.angle_to_goal_pose < 0:
                    w = -0.3
                else:
                    w = 0.3

            elif np.abs(self.angle_to_target) < 1:
                v = 0.4
            elif self.angle_to_target < 0:
                w = -0.4
            else:
                w = 0.4
        commands = np.array([v, w])
        self.last_commands = commands
        return commands

    def predict(self, dt):
        if not self.done:
            # TODO Jerome adjust this to have better estimate (find why bot moves less than predicted)
            v_correction = 0.7
            w_correction = 0.5
            self.distance_to_target -= self.last_commands[0] * dt * v_correction
            self.angle_to_target -= np.rad2deg(self.last_commands[1] * dt) * w_correction
            self.angle_to_goal_pose -= np.rad2deg(self.last_commands[1] * dt) * w_correction
            self.target_in_sight = False

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)


# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

# TODO we should implement things in a class and put this in class attribute later
trajectory = None

# TODO these were the default values in vehicle detection package, so if we need to change something
params = cv2.SimpleBlobDetector_Params()
params.minArea = 5
params.minDistBetweenBlobs = 1
detector = cv2.SimpleBlobDetector_create(params)


def calc_circle_pattern(height, width):
    """
    Calculates the physical locations of each dot in the pattern.
    Args:
        height (`int`): number of rows in the pattern
        width (`int`): number of columns in the pattern
    """
    # check if the version generated before is still valid, if not, or first time called, create

    circlepattern_dist = 0.0125
    circlepattern = np.zeros([height * width, 3])
    for i in range(0, width):
        for j in range(0, height):
            circlepattern[i + j * width, 0] = circlepattern_dist * i - \
                                              circlepattern_dist * (width - 1) / 2
            circlepattern[i + j * width, 1] = circlepattern_dist * j - \
                                              circlepattern_dist * (height - 1) / 2
    return circlepattern


def get_pose(obs, distance, option):
    '''option 0 a 3 pour differentes combinaisons de camera_matrix et distortion coefficients'''
    camera_matrix = np.array([
        305.5718893575089,
        0,
        303.0797142544728,
        0,
        308.8338858195428,
        231.8845403702499,
        0,
        0,
        1,
    ]).reshape((3, 3))
    distortion_coefs = np.array([-0.2, 0.0305, 0.0005859930422629722, -0.0006697840226199427, 0]).reshape((1, 5))
    new_camera_matrix, b = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefs, (640, 480), 0)

    (detection, centers) = cv2.findCirclesGrid(obs,
                                               patternSize=(8, 3),
                                               flags=cv2.CALIB_CB_SYMMETRIC_GRID,
                                               blobDetector=detector)
    object_points = calc_circle_pattern(3, 8)
    if detection and option == 0:
        image_points = centers[:, 0, :]
        success, rotation_vector, translation_vector = cv2.solvePnP(objectPoints=object_points,
                                                                    imagePoints=image_points,
                                                                    cameraMatrix=new_camera_matrix,
                                                                    distCoeffs=np.array([0, 0, 0, 0, 0]))
    elif detection and option == 1:
        image_points = centers[:, 0, :]
        success, rotation_vector, translation_vector = cv2.solvePnP(objectPoints=object_points,
                                                                    imagePoints=image_points,
                                                                    cameraMatrix=new_camera_matrix,
                                                                    distCoeffs=distortion_coefs)
    elif detection and option == 2:
        image_points = centers[:, 0, :]
        success, rotation_vector, translation_vector = cv2.solvePnP(objectPoints=object_points,
                                                                    imagePoints=image_points,
                                                                    cameraMatrix=camera_matrix,
                                                                    distCoeffs=np.array([0,0,0,0,0]))
    elif detection and option == 3:
        image_points = centers[:, 0, :]
        success, rotation_vector, translation_vector = cv2.solvePnP(objectPoints=object_points,
                                                                    imagePoints=image_points,
                                                                    cameraMatrix=camera_matrix,
                                                                    distCoeffs=distortion_coefs)
    else:
        return (detection, np.array([]))

    theta = rotation_vector[1][0]

    x_global = translation_vector[2][0]
    y_global = translation_vector[0][0]
    z_global = translation_vector[1][0]

    x_but_global = x_global - distance
    y_but_global = y_global
    z_but_global = z_global

    x_but_robot = x_but_global * np.cos(theta)
    y_but_robot = y_but_global * np.cos(theta)
    z_but_robot = z_but_global


    return (detection, [np.array([y_but_robot, z_but_robot, x_but_robot]), np.rad2deg(theta)])




def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    wheel_distance = 0.102
    min_rad = 0.08

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action += np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action -= np.array([0.44, 0])
    if key_handler[key.LEFT]:
        action += np.array([0, 1])
    if key_handler[key.RIGHT]:
        action -= np.array([0, 1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])


    v1 = action[0]
    v2 = action[1]
    # # Limit radius of curvature
    # if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
    #     # adjust velocities evenly such that condition is fulfilled
    #     delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
    #     v1 += delta_v
    #     v2 -= delta_v

    action[0] = v1
    action[1] = v2

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    # TODO get observation before taking step
    # For now, we do nothing, get image, compute action, execute it. So 2 steps for one action
    # It should be get observations, compute action, take step, repeat. (1 action/step)

    obs, reward, done, info = env.step(action)

    # obs is a np.array, im is PIL image, use whatever you need
    # im = Image.fromarray(obs)
    # TODO im is a np.array of the image, get relative pose from that it could also be obtained using a built-in package

    (detect, trans_vec) = get_pose(obs, 0.1, 3)
    if detect:
        print(f'option0 : {trans_vec}')

    # (detect, trans_vec) = get_pose(obs, 1)
    # print(f'option1 : {trans_vec}')
    #
    # (detect, trans_vec) = get_pose(obs, 2)
    # print(f'option2 : {trans_vec}')
    #
    # (detect, trans_vec) = get_pose(obs, 3)
    # print(f'option3 : {trans_vec}')

    # Only for debugging, slows things down considerably and is not necessary
    # if detection:
    #     cv2.drawChessboardCorners(obs,
    #                               (8, 3), centers, detection)
    #     im = Image.fromarray(obs)
    #     im.save("circle_grid.png")

    # found_object, relative_pose = method_that_do_vision_processing(im)

    # This is a temporary workaround to test only trajectory
    goal_position = np.array([2.2, 0, 2.5])  # Because duckie is at [2.5, 0. 2.5] in visual_servo.yaml env file
    goal_angle = 0  # Because duckie faces 0 angle in visual_servo.yaml env file
    cur_position = np.array(info["Simulator"]["cur_pos"])
    cur_angle_rad = info["Simulator"]["cur_angle"]
    # Here angle is rounded for printing but probabibly should not really be.
    cur_angle_deg = int(np.rad2deg(cur_angle_rad))
    if cur_angle_deg > 179:
        cur_angle_deg -= 360
    cur_pose = np.array([cur_position, cur_angle_deg])
    relative_position = goal_position - cur_position
    relative_angle = goal_angle - cur_angle_deg
    relative_pose = [relative_position, relative_angle]
    np.set_printoptions(precision=2)
    # print("cur_pose:", cur_pose, "rel_pose:", relative_pose)

    if detect:
        global trajectory
        if trajectory is None:
            trajectory = Trajectory()
        trans_vec[1] = -trans_vec[1]
        print(trans_vec, relative_pose)
        trajectory.update(trans_vec)
    else:
        if trajectory is not None:
            trajectory.predict(dt)

    # TODO take next step in defined trajectory
    if trajectory is not None:
        action = trajectory.get_step()
    else:
        print("object not found, cannot compute initial trajectory")

    obs, reward, done, info = env.step(action)

    if key_handler[key.RETURN]:
        im = Image.fromarray(obs)

        im.save("screen.png")

    if done:
        print("done!")
        env.reset()
        env.render()

    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
