# Simple-UniverSeg
Course Project For EC 500

# Pre-trained UniverSeg Model
The pre-trained UniverSeg model is described at project/models/original_universeg/model.py

# Evaluation Script (project/main.py)
This script evaluates the pre-trained UniverSeg model on [Neurite OASIS Sample Data](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md) with 24 seg protocol.

# path:
```sh
module load python3/3.8.10
export PYTHONPATH=/projectnb/ace-ig/jw_python/lib/python3.8.10/site-packages:$PYTHONPATH
module load pytorch/1.13.1

/projectnb/ec500kb/projects/UniverSeg/code

/projectnb/ec500kb/projects/UniverSeg/data

source /projectnb/ec500kb/students/jueqiw/venvs/monai/bin/activate
module load pytorch/1.13.1


module load python3/3.10.12
module load pytorch/1.13.1
# virtualenv /projectnb/ec500kb/students/jueqiw/venvs/monai
source /projectnb/ec500kb/students/jueqiw/venvs/project/bin/activated

source ./HWenv/bin/activate
```
