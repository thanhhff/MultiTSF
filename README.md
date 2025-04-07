## MultiSensor-Home: A Wide-area Multi-modal Multi-view Dataset for Action Recognition and Transformer-based Sensor Fusion

## Introduction
This repository contains the implementation of MultiTSF on the MultiSensor-Home dataset.

## Environment

The Python code is developed and tested in the environment specified in `requirements.txt`.
Experiments on the MultiSensor-Home dataset were conducted on four NVIDIA A100 GPUs, each with 32 GB of memory.
You can adjust the `batch_size` parameter in the code to accommodate GPUs with smaller memory.

## Dataset

Download the MultiSensor-Home dataset and place it in the `dataset/MultiSensor-Home` directory.

## Training
To train the model, execute the following command:
```
    bash ./scripts/train.sh
```

## Inference
To perform inference, use the following command:
```
    bash ./scripts/infer.sh
```