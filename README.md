# Simple Variant of UniverSeg on 2D Brain MRI
This is the code repository for the implementation of the training and inference scripts of the [UniverSeg model](https://arxiv.org/abs/2304.06131)

# Installation
Install the packages from `requirements.txt` file using `pip`. 
```
pip install -r requirements.txt
```
# Download a dataset
Download the data into the `data` directory and specify its location in the `project/utils/const.py` for variable `DATA_FOLDER`

# Training Configurations
## To add data augmentation
Set the value of JSON keys ("do_in_task_augmentation" or/and "do_task_augmentation") to true otherwise keep false.

# Training
Train a model with Augmentation + Combined Losses.

Modify the configurations file, go to the `project` directory, and use the following command -
```
python main_Augmentation_.py --config ./configurations/configs_final.json
```


# Inference
To run inference of the trained model, go to the `src` directory, use the following command -
```
python inference.py --config ./configurations/configs1.json
```

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
- choose_labels.ipynb
- plot_selected_labels.ipynb

# Licenses
Code is released under the [Apache 2.0 license](LICENSE).


# Code Details
Adapted Code:
- dataset.py and visualization.py: This code has been modified from the following sources: https://github.com/JJGO/UniverSeg/blob/main/example_data/oasis.py for processing the OASIS dataset.
- OASIS dataset processing details are available at https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md.
- main.py: The inference pipeline has been adapted from https://colab.research.google.com/drive/1TiNAgCehFdyHMJsS90V9ygUw0rLXdW0r?usp=sharing.
- main_Augmentation.py - Training script for UniverSeg model with Augmentation
- main_different_loss.py - Training script for UniverSeg model with Augmentation + Different losses
  
Additional Code:
- main.py: Included Dice Score and HD95 evaluation metrics.
- choose_labels.ipynb: Provides a comparison of the sizes of different Regions of Interest (ROIs) in Brain MRI images.
- plot_selected_labels.ipynb: Highlights labels to provide an overview of the various categories of labelled data.
