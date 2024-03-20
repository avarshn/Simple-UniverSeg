# Simple-UniverSeg
Course Project For EC 500


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

# To run the code:
source /projectnb/ec500kb/projects/UniverSeg/code/project/HWenv/bin/activate

qrsh -P ec500kb -l h_rt=03:00:00 -l mem_per_core=3G -l gpus=1 -l gpu_c=7
```