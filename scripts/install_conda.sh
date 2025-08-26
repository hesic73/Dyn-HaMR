#!/usr/bin/env bash
set -e

export CONDA_ENV_NAME=dynhamr

conda create -n $CONDA_ENV_NAME python=3.10 -y

conda activate $CONDA_ENV_NAME


pip install -r requirements.txt

# install source
pip install -e .

# install DROID-SLAM/DPVO
cd third-party/DROID-SLAM
python setup.py install

cd thirdparty/lietorch

export TORCH_CUDA_ARCH_LIST="7.5;8.6;8.9;9.0"
pip install --no-build-isolation .
cd ../..

cd ../..

# install HaMeR
cd third-party/hamer
pip install -e .[all]
pip install -v -e third-party/ViTPose
