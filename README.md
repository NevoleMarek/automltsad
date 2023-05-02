# AutoMLTSAD: AutoML Time Series Anomaly Detection

## Overview

This repository contains the code and experiments for the master's thesis on using AutoML for anomaly detection in a semi or unsupervised setting on time series. Simultaneously with thesis a framework for anomaly detection in time series data was developed.

## Contents
This repository contains two directories:
- [AutoMLTSAD](automlstad/)
    - A Python package developed for anomaly detection
    - Contains all models used in the thesis and more
    - Easy to use framework to handle time series data of many formats
    - Contains utility and transform functions
    - Recent evaluation schemas and metrics are implementeds
- [Experiments](experiments/)
    - Contains all scripts and configurations used to run the experiments
    - Results are explored in Jupyter notebooks

## Installation
To install the package, follow these instructions:
1. Clone the repository to your local machine.
2. The package was developed using Python [3.10.2](https://www.python.org/downloads/release/python-3102/), which is also required for the Poetry environment to work.
3. [Poetry](https://python-poetry.org/) is used for dependency management. Install it by running `pip install poetry`.
4. In the directory with `pyproject.toml`, run `poetry install`. This may take a while since many packages are used.
5. After successfully installing all packages, you can run the code.
