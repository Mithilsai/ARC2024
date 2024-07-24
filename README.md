# ARC AGI Competition Submission

All the dataset can be downloaded from: https://www.kaggle.com/competitions/arc-prize-2024/data

## Project Overview
This project was developed as part of the ARC AGI competition. The primary objective is to predict the outputs of given input grids based on training examples provided. The following models preprocesses the data, trains on the provided examples, and predicts the output for new inputs. To gain a better understanding of the objective of the competition, the ARC website provides an interactive app. https://arcprize.org/play?task=00576224

## Subash Shibu:

### Model Architecture
The CNN model is built using TensorFlow and Keras. It consists of several convolutional layers followed by max pooling, upsampling, batch normalization, and dropout layers to enhance performance and prevent overfitting. The architecture is as follows:

Input Layer: Accepts the input grid of shape (32, 32, 1)
Convolutional Layers: Extracts features from the input grid
Max Pooling Layers: Reduces the spatial dimensions of the feature maps
Batch Normalization: Normalizes the activations of the previous layer
Dropout Layers: Regularizes the model to prevent overfitting
UpSampling Layers: Increases the spatial dimensions of the feature maps
Output Layer: Produces the final output grid

### Data Preprocessing
The input and output grids are resized to a target size of 32x32 and normalized by dividing by 9. This ensures that all the grids have the same dimensions and the values are scaled appropriately for training.

### Training and Validation
The data is split into training and validation sets using an 80-20 split. The model is trained for 50 epochs with a batch size of 32. The loss function used is Mean Squared Error (MSE), and the optimizer is Adam.

### Evaluation
The model is evaluated on the validation set, achieving a validation loss of 0.0079 and a validation accuracy of 0.9598. The model's performance on test inputs is visualized using matplotlib, showing the predicted outputs alongside the actual inputs and outputs.

### Acknowledgments
This project was developed with the assistance of ChatGPT, which helped in writing and debugging the code. The TensorFlow and Keras communities also provided great resources and documentation.

# ARC Challenge Deep Learning Model: Documentation by Mithil

### 1.1 Data Sources

The model utilizes four primary JSON files:
- `arc-agi_training_challenges.json`
- `arc-agi_training_solutions.json`
- `arc-agi_evaluation_challenges.json`
- `arc-agi_evaluation_solutions.json`

### 1.2 Data Processing

A custom PyTorch Dataset class, `ARCDataset`, is implemented to process the challenge and solution data. Key features include:
- Grid padding to a standard 30x30 size
- Conversion of data to PyTorch tensors
- Separation of train and test inputs/outputs for each challenge

## 2. Model Architecture

### 2.1 Network Structure

The model, named ARCNN, is a Convolutional Neural Network with the following architecture:
- Input layer: Accepts 30x30 grids
- Convolutional layer 1: 32 filters, 3x3 kernel, ReLU activation
- Convolutional layer 2: 64 filters, 3x3 kernel, ReLU activation
- Fully connected layer 1: 512 units, ReLU activation
- Fully connected layer 2: 900 units (30x30 output)
- Output layer: Reshaped to 30x30 grid

### 2.2 Design Rationale

CNNs are chosen for their ability to process grid-like data and capture spatial relationships, which is crucial for the ARC challenge.

## 3. Training Process

### 3.1 Training Configuration

- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam (learning rate = 0.001)
- Number of epochs: 10
- Batch size: 1

### 3.2 Training Loop

For each epoch:
1. Forward pass: Input data through the model
2. Loss calculation
3. Backpropagation
4. Parameter update


## 4. Model Evaluation

### 4.1 Evaluation Process

An evaluation function assesses the model's performance on the evaluation dataset.

### 4.2 Evaluation Results

Evaluation Loss: 1.882220

### 4.3 Result Analysis

The higher evaluation loss compared to the final training loss suggests potential overfitting. The model may have learned patterns specific to the training data that don't generalize well to unseen data.

## 6. Conclusions and Future Work

### 6.1 Current Status

### Errors and Issues Encountered

IndexError in pad_grid Function
During the initial stages of model development, an `IndexError` was encountered in the `pad_grid` function:

### IndexError: tuple index out of range
This error occurred because some grids had dimensions of 0, causing the indexing operation in the `pad_grid` function to fail.

### Invalid Grid Dimensions
Upon further investigation, it was discovered that many challenges had grids with 0 dimensions. This issue was identified with error messages indicating: Skipping challenge <challenge_id> due to error: Grid should have 2 dimensions, but has 0

These errors suggested that some challenges had empty or improperly formatted grid data.

### Solutions Implemented

### Improving the pad_grid Function
To address the `IndexError`, the `pad_grid` function was modified to include a check for valid grid dimensions:

### Filtering Valid Keys in Dataset
To handle challenges with invalid grids, a new method _filter_valid_keys was added to the ARCDataset class. This method filters out keys with invalid grids and skips them during data loading

