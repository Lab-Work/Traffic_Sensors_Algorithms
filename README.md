# Traffic Sensors Algorithms

The repository constains quick **implementations of vehicle detectoin and speed estimation algorithms** and **tools for data processing and visualization**. 

Concretely, the repository contains the following components:

- `datasets/`
    - All field data sets are now saved in Google Drive: 
       https://drive.google.com/open?id=0B1EgxrmaGlFaN29sOUNmNl9INUE
    - `data_~_Noise_Analysis/` folders containing the dataset and results for analyzing the PIR noise. 
- `tools/`
    - `_parser.py` a parser that parse original sensors csv data into Python lists
    - `colormap.py` a visualization tool to generate colormaps
    - `crosscorrelation.py` cross correlation model
    - `hough_line_transform.py` an experimental Hough line transformation model
    - `hex_converter.py` a script to convert raw hexadecimal PIR data to decimal format
    - `cookb_signalsmooth.py` Gaussian filter
- `code/`
    - `TrafficDataClass.py` a universal traffic data class which handles different PIR configuration data in the format specified in Data_Collection_Manual.pdf
    - `analyze~.py` short scripts which performs simple plotting and analysis of the data
    - `test~.py` a script for debugging the TrafficDataClass
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

Run the test with `vehicle_detection.py`. The functionality has yet to be fully developed.

# Speed Estimation

Run the test with `speed_estimation.py`. The functionality has yet to be fully developed.

# Data Collection

Run the program with `logger.py`. Enter anything and press enter to make a log. Type 'exit' to terminate the program. The log file will be automatically stored in the working directory.

# Future Works

To write a shell script that can automatically compile videos from specified datasets.
