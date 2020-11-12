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
    im = Image.fromarray(obs)
    # TODO im is a np.array of the image, get relative pose from that it could also be obtained using a built-in package

    # found_object, relative_pose = method_that_do_vision_processing(im)

    # This is a temporary workaround to test only trajectory
    found_object = True
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
    print("cur_pose:", cur_pose, "rel_pose:", relative_pose)
    # TODO update trajectory if object is in field of view
    #if found_object:
    #    trajectory = get_trajectory(relative_pose)
    # We should probably put a condition on relative pose and stop if we are really close.



    # TODO take next step in defined trajectory
    # if trajectory is not None:
    #     action = trajectory.get_step()
    # else:
    #     print("object not found, cannot compute initial trajectory")

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
