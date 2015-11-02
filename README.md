# Traffic Sensors Algorithms

The repository constains quick **implementations of vehicle detectoin and speed estimation algorithms** and **tools for data processing and visualization**. 

Concretely, the repository contains the following components:

- `datasets/`
    - `tsa_090315/` data collected on september 3, 2015
    - `tsa_old/` an archive of old data
- `tools/`
    - `_parser.py` a parser that parse original sensors csv data into Python lists
    - `colormap.py` a visualization tool to generate colormaps
    - `crosscorrelation.py` cross correlation model
    - `hough_line_transform.py` an experimental Hough line transformation model
    - `hex_converter.py` a script to convert raw hexadecimal PIR data to decimal format
    - `cookb_signalsmooth.py` Gaussian filter
- `vehicle_detection.py` vehicle detection models
- `speed_estimation.py` speed estimation models
- `logger.py` a small traffic logger for creating manual traffic labels
- `visualization`
    - `colormaps_row` visualization of colormaps in row major
    - `colormaps_col` visualization of colormaps in column major
- `TODO.txt` a file of to-do tasks
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
