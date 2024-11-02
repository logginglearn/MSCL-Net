# MSCL-Net
Multi-Stream and Contrastive Learning Networks for Lithology Classification from Well Logging Data

We will showcase the complete work in the future
## Overview

This project implements a neural network model for processing and classifying logging data. The code includes data preprocessing, feature extraction using GRU and CNN architectures, and contrastive learning for improved classification performance. Complete code and datasets will be provided in subsequent updates.

## Code Explanation

### Imports

The code utilizes several libraries:

- **PyTorch** for building and training the neural network.
- **Pandas** for data manipulation and preprocessing.
- **Scikit-learn** for scaling features.
- **NumPy** for numerical operations.

### Dataset Class

The `LoggingDataset` class is responsible for loading and preprocessing the logging data from a CSV file.

#### Key Steps in Preprocessing:

1. **Handling Missing Values**:
   - Columns (`PE`, `GR`, `AC`, `CNL`, `DEN`) are checked for missing values. If the missing data is less than 5%, it is filled with the mean; otherwise, the column is dropped.

2. **Outlier Detection**:
   - Outliers are detected using a z-score method. Data points more than 3 standard deviations from the mean are replaced with the median of the column.

3. **Feature Scaling**:
   - The `StandardScaler` is applied to scale features to have zero mean and unit variance.

### DataLoader Preparation

The `prepare_data_loader` function creates a DataLoader object for batch processing of the dataset.

### MultiStreamEncoder Class

This class implements a neural network architecture combining GRU and CNN layers:

- **GRU Layer**:
  - An Attention Gated Recurrent Unit (AGRU) processes the sequence data, capturing temporal dependencies.
  
- **Convolutional Layers**:
  - Multi-layer CNN processes the input through various kernel sizes for extracting features at multiple scales.

### MultiLayerContrastiveLearning Class

This class implements a contrastive learning framework with two types of losses:

1. **Instance-level Contrastive Loss**:
   - Encourages similar instances to have closer representations in the feature space.

2. **Temporal Contrastive Loss**:
   - Focuses on the temporal relationships within sequences, enhancing the model's understanding of time-related patterns.

### Forward Method

The `forward` method combines the outputs from the encoder and computes the total contrastive loss based on both instance-level and temporal loss.

## Future Work

- Complete code and datasets will be added.
- Further optimizations and experiments will be conducted to improve model performance.

## Installation

Instructions for setting up the environment and installing necessary packages.

```bash
pip install torch pandas scikit-learn numpy
