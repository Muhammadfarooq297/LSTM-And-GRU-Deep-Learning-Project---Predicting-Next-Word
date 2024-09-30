# LSTM And GRU Deep Learning Project - Predicting Next Word

![Project Banner](https://example.com/project-banner.gif)  <!-- Replace with an actual GIF link -->

This project demonstrates the use of Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) neural networks for the task of predicting the next word in a sequence of text. The model is trained on a large corpus of text and is designed to learn patterns of language to predict the next word given a sequence of previous words.

## Table of Contents

- [LSTM And GRU Deep Learning Project - Predicting Next Word](#lstm-and-gru-deep-learning-project---predicting-next-word)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Dataset](#dataset)
  - [Model Architecture](#model-architecture)
    - [LSTM Model](#lstm-model)
    - [GRU Model](#gru-model)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Usage](#usage)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Results](#results)
  - [Conclusion](#conclusion)
  - [Acknowledgments](#acknowledgments)

## Introduction

Next-word prediction is a crucial component of various applications such as text auto-completion, machine translation, and conversational AI. This project uses deep learning techniques with LSTM and GRU models to predict the next word in a sequence.

![Next Word Prediction Animation](https://example.com/next-word-prediction.gif)  <!-- Replace with an actual GIF link -->

## Dataset

The project uses a large dataset of text, which is tokenized and cleaned to train the models. The dataset is split into input sequences where each word is used to predict the next one in the sequence. Tokenization and padding are applied to make sequences uniform, and a word-to-index mapping is created for vocabulary management.

## Model Architecture

### LSTM Model

The LSTM model is designed to capture long-term dependencies in text sequences. It consists of an embedding layer that converts words into dense vectors, two LSTM layers that capture sequential patterns, a dropout layer to prevent overfitting, and a final dense layer that outputs predictions using a softmax activation function.

![LSTM Architecture](https://example.com/lstm-architecture.gif)  <!-- Replace with an actual GIF link -->

### GRU Model

The GRU model is an alternative to LSTM, designed to be computationally lighter while still capturing sequential patterns. It also includes an embedding layer, two GRU layers for processing the input sequence, a dropout layer for regularization, and a dense layer with softmax activation for word prediction.

![GRU Architecture](https://example.com/gru-architecture.gif)  <!-- Replace with an actual GIF link -->

## Training

The models are trained using categorical cross-entropy loss and the Adam optimizer. Training is done in batches with a validation split to monitor the model's performance and avoid overfitting. The number of epochs, batch size, and optimizer configuration can be tuned for optimal performance.

![Training Process Animation](https://example.com/training-animation.gif)  <!-- Replace with an actual GIF link -->

## Evaluation

The models are evaluated using accuracy metrics. The validation set is used to measure accuracy during training, and the final test set evaluation provides the model’s overall performance. The performance of both LSTM and GRU models is compared to see which architecture performs better on next-word prediction.

## Usage

To use this project, you will need to clone the repository, install the required dependencies, and train the model using the dataset provided. You can train either the LSTM or GRU model depending on your preference, and test the model by inputting a sequence of text to predict the next word.

## Requirements

- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib

These dependencies can be installed via a package manager such as pip. A `requirements.txt` file is provided in the repository to help set up the environment quickly.

## Installation

To run this project:

1. Clone the repository from GitHub.
2. Install the required dependencies using the `requirements.txt` file.
3. Train the LSTM or GRU model by running the appropriate training script.
4. After training, you can test the model by inputting sequences of text to see the predicted next word.

## Results

The project showcases the comparison between LSTM and GRU models in terms of next-word prediction. Typically, the LSTM model tends to perform better at capturing long-term dependencies, while the GRU model offers a faster and more efficient alternative with slightly lower accuracy.

- **LSTM Model Accuracy**: [81 %]
- **GRU Model Accuracy**: [82.9 %]

Both models perform well, and their accuracy depends on the complexity of the dataset and the number of training epochs.

![Results Comparison](https://example.com/results-comparison.gif)  <!-- Replace with an actual GIF link -->

## Conclusion

This project demonstrates the use of LSTM and GRU neural networks for next-word prediction. LSTM provides better accuracy for tasks that require capturing long-term dependencies, while GRU offers faster training with a reduced number of parameters. The trade-off between speed and accuracy makes both models useful depending on the specific requirements of the task.

## Acknowledgments

Special thanks to the open-source community and the creators of the tools and frameworks used in this project, including TensorFlow, Keras, and the dataset providers.
