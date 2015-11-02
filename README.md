# Traffic Sensors Algorithms

The repository constains quick **implementations of vehicle detectoin and speed estimation algorithms** and **tools for data processing and visualization**. 

Concretely, the repository contains the following components:

- `datasets/`
    - `tsa_090315/` data collected on september 3, 2015
    - `tsa_old/` an archive of old data
- `tools/`
    - `_parser.py`
    - `colormap.py`
    - `crosscorrelation.py`
    - `hough_line_transform.py`
    - `hex_converter.py`
    - `cookb_signalsmooth.py`
- `vehicle_detection.py`
- `speed_estimation.py`
- `logger.py`
- `visualization`
- `TODO.txt`
- `deprecated` an archive of old implementations

# Installation

The project is mainly developed in Python 2. In order to run most of the scripts, users would need Python 2 as well as Numpy and Matplotlib libraries. Specifically, to run the Hough line transformation, one should also install OpenCV Python libray.

# Vehicle Detection

Run `vehicle_detection.py`.

# Speed Estimation

Run `speed_estimation.py`.

# Data Collection

Run `logger.py`.

# Future Works

To write a shell script that can automatically compile videos from specified datasets.
