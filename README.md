# CVPR - exam
This repository contains the exam project for the Computer Vision and Pattern Recognition course. The project consists in the implementation of an image classifier using the bag-of-words approach.


## Project organisation

The source code is located in the `souce/` folder, the project report in the `report/` folder and the accuracy plots in the `plots/` folder.

The code is organised as follows:
* `main.ipynb` presents the implemented. It is presented in a notebook format, to provide a more convenient interaction.
* `vocabulary.py` implements the methods to read the images and build a visual vocabulary from them.
* `classifiers.py` provides the implementation for a number of different classifiers.
* `utils.py` implements some utility functions to measure code execution and plot confusion matrices.


## Notes
`main.ipynb` contains indications for expected execution time on demanding cells. This expectations are referred to and Intel i7-8550U CPU and might be unreliable, depending on the available computing power.
