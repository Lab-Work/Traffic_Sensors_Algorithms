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
    - `TrafficDataClass.py` a universal traffic data class which handles different PIR configuration data in the format specified in data_collection\manual.pdf
	- `TrafficModelsClass.py` a model-packed class used for vehicle detection and speed estimation.
    - `analyze~.py` short scripts which performs simple plotting and analysis of the data
    - `test~.py` a script for debugging the TrafficDataClass
- `data_collection/`
    - `manual.pdf` the protocol by which the input data is stored
    - `tables.pdf` the tables to be filled by hardware tester(s) and field work surveyer(s)
    - `logger.py` a small traffic logger for creating manual traffic labels
- `visualization`
    - `colormaps_row` visualization of colormaps in row major
    - `colormaps_col` visualization of colormaps in column major
- `deprecated` an archive of old implementations

# Installation

The project is mainly developed in Python 2. In order to run most of the scripts, users would need Python 2 as well as Numpy, Matplotlib, and scikit-learn libraries.

# Data Collection

Run the program with `logger.py`. Enter anything and press enter to make a log. Type 'exit' to terminate the program. The log file will be automatically stored in the working directory.

Below is a list of YouTube videos that provide visualization for collected dataset:
- June 25th: https://www.youtube.com/watch?v=U-MzDkoVVvI
- September 3rd: https://www.youtube.com/watch?v=AKLSiwKt6Aw

# Future Works

To use off-the-shell tools to do camera vehicle detection to expand the contents of label data. 
