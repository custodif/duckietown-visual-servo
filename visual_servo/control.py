"""
This module is computing duckiebot commands from estimated pose to a target
"""
from typing import Tuple

import numpy as np

from config import (ANGLE_THRESHOLD, DISTANCE_THRESHOLD, FINAL_ANGLE_THRESHOLD,
                    PURSUIT_DISTANCE_THRESHOLD, V_CONSTANT, V_CORRECTION,
                    W_CONSTANT, W_CORRECTION)


class Trajectory:
    """
    Object that handles the trajectory logic
    """
    def __init__(self):
        self.distance_to_target = 0.0
        self.angle_to_target = 0.0
        self.angle_to_goal_pose = 0.0
        self.last_update_time = None
        self.last_commands = None
        self.done = False
        self.target_in_sight = False
        self.initialized = False
        self.final_step = False

    def is_initialized(self) -> bool:
        """
        getter method to know if trajectory was initialized
        Returns:
            True if we have seen the object at least once to compute initial trajectory
        """
        return self.initialized

    def reset(self):
        """
        Resets trajectory
        """
        self.distance_to_target = 0.0
        self.angle_to_target = 0.0
        self.angle_to_goal_pose = 0.0
        self.last_update_time = None
        self.last_commands = None
        self.done = False
        self.target_in_sight = False
        self.initialized = False
        self.final_step = False

    def update(self, relative_pose: Tuple[np.array, float]):
        """
        Update our belief on target location from estimated pose
        Args:
            relative_pose: A tuple ([y, z, x], theta) giving pose relative to target
        """
        if not self.done:
            x_dist = relative_pose[0][2]
            y_dist = relative_pose[0][0]
            self.distance_to_target = np.sqrt(x_dist**2 + y_dist**2)
            self.angle_to_target = - np.rad2deg(np.arctan(x_dist/y_dist))
            self.angle_to_goal_pose = relative_pose[1]
            self.target_in_sight = True
            self.initialized = True

    def get_commands(self) -> np.array:
        """
        Get next duckiebot commands from current belief
        Returns:
            a np.array [v, w]
        """
        v = 0.0
        w = 0.0
        if not self.done:
            if self.distance_to_target < DISTANCE_THRESHOLD or self.final_step:
                # in the final step, we only care about facing in the same direction as the target
                if np.abs(self.angle_to_goal_pose) < FINAL_ANGLE_THRESHOLD:
                    self.done = True
                elif self.distance_to_target > PURSUIT_DISTANCE_THRESHOLD:
                    # We go back in "pursuit" mode if the target estimation is far.
                    # This allows to pursue a moving target
                    self.final_step = False
                # We enter the final step if we are very close to target. A larger distance is needed to get out of
                # that final step to avoid oscillating between the different objectives
                elif self.angle_to_goal_pose < 0:
                    w = -W_CONSTANT
                    self.final_step = True
                else:
                    w = W_CONSTANT
                    self.final_step = True
            # If we are aligned with target, we go straight toward it
            elif np.abs(self.angle_to_target) < ANGLE_THRESHOLD:
                v = V_CONSTANT
            # IF not aligned, we turn in the right direction to reduce the angle to the target
            elif self.angle_to_target < 0:
                w = -W_CONSTANT
            else:
                w = W_CONSTANT
        commands = np.array([v, w])
        self.last_commands = commands
        return commands

    def predict(self, dt: float):
        """
        Update belief according to kinematics when no object is detected.
        This is a coarse approximation but works well enough. On the bot, we should predict the new position using
        the wheels encoders.
        Args:
            dt: time elapsed since last commands where given
        """
        if not self.done:
            v_correction = V_CORRECTION
            w_correction = W_CORRECTION
            self.distance_to_target -= self.last_commands[0] * dt * v_correction
            self.angle_to_target -= np.rad2deg(self.last_commands[1] * dt) * w_correction
            self.angle_to_goal_pose -= np.rad2deg(self.last_commands[1] * dt) * w_correction
            self.target_in_sight = False
