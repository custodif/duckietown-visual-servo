"""
This file contains the parameters used for estimation and control in our visual servo
"""

# estimation params
BUMPER_TO_CENTER_DIST = 0.17  # approximation of the distance between the center of the duckie object and the bumper
CAMERA_MODE = 2  # We tried different we get camera parameters. Mode 2 seems to work best but is still not so good
CIRCLE_MIN_AREA = 5
CIRCLE_MIN_DISTANCE = 1
CIRCLE_PATTERN_DIST = 0.0125
CIRCLE_PATTERN_HEIGHT = 3
CIRCLE_PATTERN_WIDTH = 8
TARGET_DIST = 0.15  # This is the distance to the bumper that we want aim at.

# Control params
ANGLE_THRESHOLD = 1.0
DISTANCE_THRESHOLD = 0.02
FINAL_ANGLE_THRESHOLD = 1.0
PURSUIT_DISTANCE_THRESHOLD = 0.2
V_CONSTANT = 0.2
V_CORRECTION = 1.0
W_CONSTANT = 0.2
W_CORRECTION = 0.44  # This is a manually tuned constant to approximate change in angle from last angle
