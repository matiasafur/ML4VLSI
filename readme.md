# Convolutional Neural Network for VLSI

This project implements a Convolutional Neural Network (CNN) designed for image classification, created during the Machine Learning for VLSI course at EAMTA (Argentine School of Micro-Nanoelectronics, Technology, and Applications) - March 2023.

## Overview

The CNN is designed using PyTorch and consists of three convolutional layers followed by max-pooling layers, and three fully connected layers. The model is trained and evaluated on the CIFAR-10 dataset.

## Network Architecture

The network architecture is defined in the `ConvNet` class and includes the following layers:

- **Conv1:** Convolutional layer with 3 input channels, 64 output channels, kernel size of 3x3, and padding of 1.
- **Conv2:** Convolutional layer with 64 input channels, 128 output channels, kernel size of 3x3, and padding of 1.
- **Conv3:** Convolutional layer with 128 input channels, 256 output channels, kernel size of 3x3, and padding of 1.
- **Pool:** MaxPooling layer with a kernel size of 2x2 and stride of 2.
- **FC1:** Fully connected layer with input size of 256 * 4 * 4 and output size of 1024.
- **FC2:** Fully connected layer with input size of 1024 and output size of 512.
- **FC3:** Fully connected layer with input size of 512 and output size of 10.
- **Dropout:** Dropout layer with a dropout probability of 0.5.

## Training and Testing

The network is trained using the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The training and testing procedure includes:

1. **Dataset Preparation:** Load and normalize the CIFAR-10 dataset.
2. **Model Initialization:** Instantiate the `ConvNet` model and define the loss function and optimizer.
3. **Training Loop:** Iterate over the training dataset to optimize the model parameters.
4. **Evaluation Loop:** Compute the model accuracy on the test dataset.

## Usage

To train and evaluate the network, run the `main.py` script:

```bash
python main.py
```

# Dimensioning and Estimating a MAC Unit

To dimension and estimate the size of a Multiply-Accumulate (MAC) unit using the provided data, we will follow a series of steps that include calculating the number of logic gates required, and estimating the area and power consumption. We will consider 8-bit quantization for weights and inputs.

## 1. Calculation of the Number of Logic Gates

A MAC unit with 8-bit precision includes two main components: the multiplier and the accumulator.

### a. 8-bit Multiplier

An 8-bit multiplier can be implemented using a combination of logic gates. We will assume an array multiplier, which is a common implementation for this precision.

- An 8-bit multiplier typically requires \(8 \times 8 = 64\) partial multiplications.
- Each partial multiplication requires an AND gate.
- Additionally, the partial sums can be managed by half and full adders.

To simplify, let's assume the multiplier requires approximately 1000 logic gates.

### b. 32-bit Accumulator

The accumulator is simpler to dimension. A 32-bit accumulator might require:
- A 32-bit adder (which can be implemented with 32 full adders).
- Each full adder can be implemented with approximately 20 logic gates.

Therefore, the 32-bit accumulator will require approximately \(32 \times 20 = 640\) logic gates.

## 2. Estimation of Area

Using the provided data, we can calculate the area of each type of logic gate and then sum the total.

### a. Area per Gate Type

- AND2: \(1.8 \mu m \times 1.0 \mu m = 1.8 \mu m^2\)
- NAND2: \(1.8 \mu m \times 0.8 \mu m = 1.44 \mu m^2\)
- INV: \(1.8 \mu m \times 0.6 \mu m = 1.08 \mu m^2\)

### b. Total Area Estimation

Assuming the 8-bit multiplier primarily uses AND2 gates and the 32-bit accumulator primarily uses NAND2 and INV gates:

- 8-bit multiplier: 1000 AND2 gates: \(1000 \times 1.8 \mu m^2 = 1800 \mu m^2\)
- 32-bit accumulator: 320 NAND2 and 320 INV gates:
  - NAND2: \(320 \times 1.44 \mu m^2 = 460.8 \mu m^2\)
  - INV: \(320 \times 1.08 \mu m^2 = 345.6 \mu m^2\)

Total area for the accumulator: \(460.8 \mu m^2 + 345.6 \mu m^2 = 806.4 \mu m^2\)

Total area for the MAC unit: \(1800 \mu m^2 + 806.4 \mu m^2 = 2606.4 \mu m^2\)

## 3. Estimation of Power Consumption

Power consumption can be estimated using the dynamic and static power data provided.

### a. Dynamic Power Consumption

Using the provided dynamic power values:

- AND2: 0.0021 µW/MHz
- NAND2: 0.00095 µW/MHz
- INV: 0.00074 µW/MHz

8-bit multiplier: \(1000 \times 0.0021 \mu W/MHz = 2.1 \mu W/MHz\)

32-bit accumulator:
- NAND2: \(320 \times 0.00095 \mu W/MHz = 0.304 \mu W/MHz\)
- INV: \(320 \times 0.00074 \mu W/MHz = 0.2368 \mu W/MHz\)

Total dynamic power: \(2.1 \mu W/MHz + 0.304 \mu W/MHz + 0.2368 \mu W/MHz = 2.6408 \mu W/MHz\)

### b. Static Power Consumption (Leakage Power)

Using the provided leakage power values:

- AND2: 14.23 pW
- NAND2: 8.2 pW
- INV: 5.14 pW

8-bit multiplier: \(1000 \times 14.23 pW = 14230 pW\)

32-bit accumulator:
- NAND2: \(320 \times 8.2 pW = 2624 pW\)
- INV: \(320 \times 5.14 pW = 1644.8 pW\)

Total leakage power: \(14230 pW + 2624 pW + 1644.8 pW = 18498.8 pW = 18.5 nW\)

## Summary

For an 8-bit MAC unit with a 32-bit accumulator:

- **Estimated Area**: 2606.4 µm²
- **Estimated Dynamic Power**: 2.6408 µW/MHz
- **Estimated Leakage Power**: 18.5 nW
