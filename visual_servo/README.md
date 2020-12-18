# VISUAL SERVO FINAL PROJECT
Fernanda Carmo, Francois Hebert, Jerome Labonte

Duckietown class of Universite de Montreal Fall 2020
## Description
This script is meant as a first exploration of using visual servoing 
to control the duckiebot. The visual servo package that can be installed on the duckiebot
contains some improvements on this script but we decided to provide these files 
as a sandbox to experiment new ideas.

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

## Expected behaviour
Once the gym window is opened, your duckie but will move toward
the stationnary bot and try to park 15cm behind it and look in the same direction. 
You can change this value in the config.py file. If your bot doesn't not move, in means
 it has not detected the pattern. You can either press backspace to reinitialize the bot in 
a new location or move your duckiebot until it detects the pattern.

## Known problems
The estimation of the pose is not very precise at the moment. This is due to 
bad camera parameters. We did not succeed in finding the right parameters to use. 
You can see some of our tries in the estimation.py file, where
each camera_mode was a guess about a way to get the right values.