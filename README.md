## MultiSensor-Home: A Wide-area Multi-modal Multi-view Dataset for Action Recognition and Transformer-based Sensor Fusion
![Official Implementation](https://img.shields.io/badge/Official-Implementation-brightgreen)
[![ArXiv](https://img.shields.io/badge/arXiv-2504.02287-red)](https://www.arxiv.org/abs/2504.02287)

This work was presented at the 19th IEEE International Conference on Automatic Face and Gesture Recognition (FG2025). **Best Student Paper Award**.

**Authors:** [Trung Thanh Nguyen](https://scholar.google.com/citations?user=QSV452QAAAAJ), [Yasutomo Kawanishi](https://scholar.google.com/citations?user=Tdfw6WMAAAAJ), [Vijay John](https://scholar.google.co.jp/citations?user=Wv71RXYAAAAJ), [Takahiro Komamizu](https://scholar.google.com/citations?user=j4n_V44AAAAJ), [Ichiro Ide](https://scholar.google.com/citations?user=8PXJm98AAAAJ)


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

## 📄 Citation
```
@inproceedings{nguyen2025multisensor,
  author    = {Trung Thanh Nguyen and Yasutomo Kawanishi and Vijay John and Takahiro Komamizu and Ichiro Ide},
  title     = {MultiSensor-Home: A Wide-area Multi-modal Multi-view Dataset for Action Recognition and Transformer-based Sensor Fusion},
  booktitle = {Proceedings of the 19th IEEE International Conference on Automatic Face and Gesture Recognition},
  year      = {2025},
  note      = {Best Student Paper Award}
}
```
