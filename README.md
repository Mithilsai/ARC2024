# ARC AGI Competition Submission

All the dataset can be downloaded from: https://www.kaggle.com/competitions/arc-prize-2024/data

## Project Overview
This project was developed as part of the ARC AGI competition. The primary objective is to predict the outputs of given input grids based on training examples provided. The following models preprocesses the data, trains on the provided examples, and predicts the output for new inputs. To gain a better understanding of the objective of the competition, the ARC website provides an interactive app. https://arcprize.org/play?task=00576224

## Subash Shibu

## Model Architecture
The CNN model is built using TensorFlow and Keras. It consists of several convolutional layers followed by max pooling, upsampling, batch normalization, and dropout layers to enhance performance and prevent overfitting. The architecture is as follows:

Input Layer: Accepts the input grid of shape (32, 32, 1)
Convolutional Layers: Extracts features from the input grid
Max Pooling Layers: Reduces the spatial dimensions of the feature maps
Batch Normalization: Normalizes the activations of the previous layer
Dropout Layers: Regularizes the model to prevent overfitting
UpSampling Layers: Increases the spatial dimensions of the feature maps
Output Layer: Produces the final output grid

## Data Preprocessing
The input and output grids are resized to a target size of 32x32 and normalized by dividing by 9. This ensures that all the grids have the same dimensions and the values are scaled appropriately for training.

## Training and Validation
The data is split into training and validation sets using an 80-20 split. The model is trained for 50 epochs with a batch size of 32. The loss function used is Mean Squared Error (MSE), and the optimizer is Adam.

## Evaluation
The model is evaluated on the validation set, achieving a validation loss of 0.0079 and a validation accuracy of 0.9598. The model's performance on test inputs is visualized using matplotlib, showing the predicted outputs alongside the actual inputs and outputs.

## Acknowledgments
This project was developed with the assistance of ChatGPT, which helped in writing and debugging the code. Special thanks to the TensorFlow and Keras communities for providing excellent resources and documentation.
