# GIZ4_AIS2_S1
DEPI R4
# Amit Repository
## Project Overview
This repository is designed for Python projects related to Machine Learning.


 Project Overview:

This project implements a simple neural network from scratch using only NumPy.
The network consists of:

Two fully connected (Dense) layers

A Tanh activation function

A Sigmoid activation function


The goal is to understand how a forward pass works inside neural networks without relying on high-level frameworks like TensorFlow or PyTorch.

pip install numpy
---

How the Code Works

1. Dense Layer (Layer_Dense)

Each Dense layer:

Initializes random weights

Initializes zero biases

Computes:


output = inputs · weights + biases

2. Activation_Tanh

Applies the Tanh activation function:

tanh(x)

3. Activation_Sigmoid

Applies the Sigmoid activation function:

1 / (1 + e^(-x))

4. Forward Pass Flow

The data passes through the network in this order:

1. Dense Layer 1
2. Tanh Activation
3. Dense Layer 2
4. Sigmoid Activation

---

Purpose of the Project

This project helps you understand:

How Dense layers work internally

How activation functions transform data

How data flows through neural network layers

Fundamental concepts before moving to deep learning frameworks

---

Expected Output

The output will show the results of each step in the forward pass:

Output of the first Dense layer

Output after Tanh activation

Output of the second Dense layer

Output after Sigmoid activation

(Values may vary slightly due to random initialization.)

---
Future Improvements

Possible future upgrades:

Add Backpropagation
Add Loss functions
Train on real datasets
Add more layers
Add different activation functions