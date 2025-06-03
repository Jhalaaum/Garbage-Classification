# Waste Classification

## Overview

This project focuses on classifying waste into six categories: Cardboard, Plastic, Trash, Glass, Metal, and Paper using multiple approaches:

1. A custom-built Convolutional Neural Network (CNN)

2. Transfer Learning with MobileNet

3. Object Detection using YOLOv8

### Citations
Dataset and CNN code adapted from TrashNet by Gary Thung and Mindy Yang. Please cite the original repository if used.

YOLOv8 detection trained on TACO: Trash Annotations in Context Dataset by Pedro Proença and Pedro Simões.

Transfer learning implemented using MobileNet (Howard et al., 2017).

### Methods

1. Custom CNN - 
A baseline CNN was implemented and trained from scratch using the TrashNet dataset.
Model performance was benchmarked using standard metrics (accuracy, precision, recall).
Refer to model.ipynb for architecture and training process.

2. Transfer Learning - 
Leveraged MobileNet (Howard et al., 2017) for feature extraction and fine-tuning.
Enabled faster training and improved accuracy with limited data.
See mobilenet.ipynb for implementation details.

3. Object Detection (YOLOv8) - 
Used YOLOv8 for real-time object detection of waste in varied environments.
Trained on the TACO Dataset for diverse annotations and realistic scenarios.
Code and inference examples available in pretrainedlive.py.

## Collaborators

1. Aum