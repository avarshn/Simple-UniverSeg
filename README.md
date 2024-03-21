# Simple-UniverSeg
Course Project For ENG EC 500 - Biomedical Images for AI

# Task
To train a simpler variant of UniverSeg, which trains on several labels of 24 seg-protocol of 2D brain MRI and generalize to holdout labels.
![Few Shot Segmentation Task on a Query Image using a Support Set](result/task.jpg)


# Pre-trained UniverSeg Model
The pre-trained UniverSeg model is described at project/models/original_universeg/model.py

# Evaluation Script (project/main.py)
This script evaluates the pre-trained UniverSeg model on [Neurite OASIS Sample Data](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md) with 24 seg protocol.

# Utilities scripts (project/utils folder)
- dataset.py: Loads the Neurite OASIS data.
- visualization.py: For visualizing the Original Image, Ground truths, Soft Predictions and Predictions

# Plots for visualization (project/Plots)
- choose_labels.ipynb - Comparison of size of different ROIs in the Brain MRI images
- plot_selected_labels.ipynb - Highlight the labels to get an idea of the various categories of labelled data

# Licenses
Code is released under the [Apache 2.0 license](LICENSE).
