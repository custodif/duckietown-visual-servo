# VISUAL SERVO FINAL PROJECT
## Fernanda, Francois, Jerome

## Description
Add project description here


## Architecture
Here are the main components of this project.
* estimation.py contains everything that is related to estimating the relative 
pose of the target point to the robot.
* control.py contains everything that is related to generating the
duckiebot commands from the pose estimation
* visual_servo.py is the main script that puts the two components together 
* config.py contains the configuration values

## Usage
Before running for the first time, create a virtual environment and from the root folder if this repository
run:
```bash
pip install -e .
```

to run the visual servo script, move inside the visual_servo directory and run:
```bash
python ./visual_servo.py
```

you can set the logging level to debug by adding the --debug flag. This will allow you to see additional information generated for debugging purposes
