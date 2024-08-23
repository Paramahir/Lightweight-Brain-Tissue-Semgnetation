# Lightweight Brain Tissue Segmentation

This repository contains code for lightweight and efficient segmentation of brain tissues using deep learning models. The aim is to provide a streamlined approach for the automatic segmentation of brain tissue from MRI scans or similar medical images.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Overview

Brain tissue segmentation is a crucial step in medical image analysis, particularly for diagnosing and monitoring various neurological conditions. This repository provides a deep learning-based method for segmenting different types of brain tissues in an efficient and lightweight manner.

## Features

- **Lightweight Model**: A deep learning model optimized for quick and accurate brain tissue segmentation.
- **Preprocessing Scripts**: Tools to handle and prepare input data.
- **Training Scripts**: Easily train the model from scratch or fine-tune it on your data.
- **Evaluation Scripts**: Measure the model's performance on test data with minimal effort.
- **Python Interface**: User-friendly and easy to integrate into existing workflows.

## Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/Paramahir/Lightweight-Brain-Tissue-Segmentation.git
cd Lightweight-Brain-Tissue-Segmentation
pip install -r requirements.txt
```

## Usage

To segment brain tissue from your own MRI images:

1. Place your MRI images in a directory.
2. Run the following command:

```bash
python main.py --input_dir path/to/images --output_dir path/to/save/segmented_images
```
## Training

To train the model on your own dataset:

1. Prepare your dataset and update the configuration file (`config.py`).
2. Start the training process:

```bash
python train.py
```
## Evaluation

To evaluate the trained model on a test dataset:

```bash

python evaluate.py --test_data path/to/test_data
```
