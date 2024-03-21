# Simple-UniverSeg
Course Project For EC 500

# Task
To train a simpler variant of UniverSeg, which trains on several labels of 24 seg-protocol of 2D brain MRI and generalize to holdout labels.
![Few Shot Segmentation Task on a Query Image using a Support Set](relative/path/to/your_image.png)


# Pre-trained UniverSeg Model
The pre-trained UniverSeg model is described at project/models/original_universeg/model.py

# Evaluation Script (project/main.py)
This script evaluates the pre-trained UniverSeg model on [Neurite OASIS Sample Data](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md) with 24 seg protocol.

# Utilities scripts (project/utils folder)
- dataset.py: Loads the Neurite OASIS data.
- visualization.py: For visualizing the Original Image, Ground truths, Soft Predictions and Predictions

# path:
```sh
module load python3/3.8.10
export PYTHONPATH=/projectnb/ace-ig/jw_python/lib/python3.8.10/site-packages:$PYTHONPATH
module load pytorch/1.13.1

cd /projectnb/ec500kb/projects/UniverSeg/code
cd /projectnb/ec500kb/projects/UniverSeg/data

module load python3/3.10.12
module load pytorch/1.13.1
# virtualenv /projectnb/ec500kb/students/jueqiw/venvs/monai
source /projectnb/ec500kb/students/jueqiw/venvs/project/bin/activated

qrsh -P ec500kb -l h_rt=03:00:00 -l mem_per_core=3G -l gpus=1 -l gpu_c=7

source /projectnb/ec500kb/projects/UniverSeg/code/project/HWenv/bin/activate
 pip install torch torchvision torchaudio
```

# Licenses
Code is released under the [Apache 2.0 license](LICENSE).
