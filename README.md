# EaRNet_DOA

## Introduction
An implementation for EaRNet(Extract and Reconstruction Network)


## Requirments
```
pip install -r requirements.txt
```

## Download
We have upload our pretrained model on Google Drive, you can download [**here**](https://drive.google.com/file/d/1rBNYJXvrtmYAemD5UJIt2G6rMlezIH3d/view?usp=drive_link)


## Usage
1. git clone this repo
```
git clone https://github.com/CHMimilanlan/EaRNet_DOA.git
```
2. Run code to generate your dataset
```
cd Dataset
# set your expected environment args in file makeData_labelNoDown_snrAll.py
python makeData_labelNoDown_snrAll.py
```

3. After generate your dataset, train EaRNet
```
cd ../trainEaRNet-Stage1
python train_1d_tcn_KL.py
# train stage2 after stage1
cd ../trainEaRNet-Stage2
python train_1d_tcn_KL.py
```

4. Evaluate
```
cd ../eval
python sonic_infer_read.py
```

## Todo (Priority-sorted)
- [ ] Add SIR dataset generation code.

## :dart: Update Log
[24/11/22] Create repo and update partial code
