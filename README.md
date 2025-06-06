# Waste Classification

This is a waste classification model made by Aum in his free time. 

## Overview

This project focuses on classifying waste into six categories: Cardboard, Plastic, Trash, Glass, Metal, and Paper using multiple approaches:

1. A custom-built Convolutional Neural Network (CNN)

2. Transfer Learning with MobileNet

3. Object Detection using YOLOv8

4. Transfer learning from Roboflow

## Datasets

There are 2 datasets - from Trashnet and Roboflow.

## Methods

1. Custom CNN - 
A baseline CNN was implemented and trained from scratch using the TrashNet dataset.
Model performance was benchmarked using standard metrics (accuracy, precision, recall).
Refer to model.ipynb for architecture and training process.

2. Transfer Learning - 
Leveraged ResNet18 (He et al., 2015) for feature extraction and fine-tuning.
Enabled efficient training and strong performance even with limited data.
See transfer_learning.ipynb for implementation details.


3. Object Detection (YOLOv8) - 
Used YOLOv8 for real-time object detection of waste in varied environments.
Trained on the TACO Dataset for diverse annotations and realistic scenarios.
Code and inference examples available in pretrainedlive.py.

4. Transfer learning from Roboflow - 
used an online roboflow model(same as one used for dataset)
Code found in roboflow.py

## Collaborators

1. Aum

## Citations
Dataset and CNN code adapted from TrashNet by Gary Thung and Mindy Yang. Please cite the original repository if used.

YOLOv8 detection trained on TACO: Trash Annotations in Context Dataset by Pedro Proença and Pedro Simões.

Transfer learning implemented using ResNet18 (He et al., 2015).

Roboflow dataset - 

        waste-classifier-louut_dataset,
        title = { Waste Classifier Dataset },
        type = { Open Source Dataset },
        author = { Classification },
        howpublished = { \url{ https://universe.roboflow.com/classification-vu3ol/waste-classifier-louut } },
        url = { https://universe.roboflow.com/classification-vu3ol/waste-classifier-louut },
        journal = { Roboflow Universe },
        publisher = { Roboflow },
        year = { 2024 },
        month = { apr },
        note = { visited on 2025-06-03 }