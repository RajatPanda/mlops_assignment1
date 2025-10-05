# MLOps Assignment 1: Boston Housing Price Prediction

This repository contains a complete machine learning workflow for predicting house prices using the Boston Housing
dataset with DecisionTreeRegressor and KernelRidge models.

## Repository Structure

```
mlops-assignment1-boston-housing/
├── README.md
├── requirements.txt
├── misc.py              # Utility functions for ML pipeline
├── train.py            # DecisionTreeRegressor training script
├── train2.py           # KernelRidge training script
└── .github/
    └── workflows/
        └── ml-pipeline.yml
```

## Branch Structure

- **main**: Contains README.md and merged code from both branches
- **dtree**: Contains DecisionTreeRegressor implementation with requirements.txt and misc.py
- **kernelridge**: Contains KernelRidge implementation and GitHub Actions workflow

## Installation

1. Clone the repository:

```bash
git clone https://github.com/RajatPanda/mlops_assignment1.git
cd mlops_assignment1
```

2. Create a conda environment:

```bash
conda create -n mlops-env python=3.8
conda activate mlops-env
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training DecisionTreeRegressor

```bash
python train.py
```

### Training KernelRidge

```bash
python train2.py
```