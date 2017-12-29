# Traffic_Sensor_Algorithms
Yanning Li, Fangyu Wu, December, 2017

## 1) Overview
This repository contains the source code developed for traffic detection and vehicle speed estimation using PIR sensors. This results are reported in a journal paper *"Traffic detection via energy efficient passive infrared sensing"* by Yanning Li, Christopher Chen, Fangyu Wu, Christian Claudel, and Daniel Work, which was submitted to *IEEE Transactions on Intelligent Transportation Systems*.

## 2) License

This software is licensed under the *University of Illinois/NCSA Open Source License*:

**Copyright (c) 2017 The Board of Trustees of the University of Illinois. All rights reserved**

**Developed by: Department of Civil and Environmental Engineering University of Illinois at Urbana-Champaign**

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal with the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimers in the documentation and/or other materials provided with the distribution. Neither the names of the Department of Civil and Environmental Engineering, the University of Illinois at Urbana-Champaign, nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.

## 3) Structure

- `/src/` The source code folder. 
	- `main_v3.py'` the main function for running the adaptive regression method for vehicle detection and speed estimation
	- `TrafficSensorAlg_V3.py` the file containing all the classes needed for the adaptive regression method.
	- `alternative_MidpointSmoothingAlg.py` an alternative heuristic based method for vehicle detection and speed estimation.

- `/datasets/` The datasets folder, which contains all the sensor data reported in the journal article. 

- `/workspace/` The workspace folder, which saves all the intermediate data, detection results, and figures. 

## 4) How to use

- **Installation:** Install Python 2 or 3, and packages Numpy, Matplotlib, openCV, pandas, scipy, sklearn.

- **Run the code:**
	- *Adaptive regression method*: Comment/uncomment or set True/False in `main_v3.py` to process desired dataset. Then in the terminal, run `cd SRC_DIRECTORY && python main_v3.py`. 
	- *Heuristic midpoint smoothing method*: Comment/uncomment or set True/False in `alternative_MidpointSmoothingAlg.py` to process desired dataset. Then in the terminal, run `cd SRC_DIRECTORY && alternative_MidpointSmoothingAlg.py`. 




