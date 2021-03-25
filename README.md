# Video Feature extractor
This repository is based on ![https://github.com/HS-YN/pytorch-i3d](https://github.com/HS-YN/pytorch-i3d)

## Environment
```bash
# For pip
pip install -r requirements.txt
# For conda
conda install -f environment.yml
```

## Feature Extarction
`extract_features.py` contains the code to load a pre-trained model (I3D or Resnet) and extract the features and save the features as numpy arrays.
