# VR-Flow-State-Assessment-Framework

This repository is the implementation of the paper "Assessing Flow State in Virtual Reality: A Multi-Channel Physiological Framework with Self-Supervised Pre-Training".

# Data Preparation
We provide [preprocessed data and features](https://pan.baidu.com/s/1m60A38gwZQq5WrE8Vg3U1A?pwd=h3m3) as [raw data](https://pan.baidu.com/s/1xpoUgMAI5OnIMAKw0omsFA?pwd=fvty). Please download and place them in the `files/` directory.

# Data Processing
The code required for data processing and model training is under the `flow_soft` folder.

The `process/` folder contains the data processing code. If you need to reprocess the data, please follow these steps:
1. Set `Folder.root` in `filepath` to the root directory where your data is stored.
2. Load the raw data provided in the document and place it in the `Folder.rawValue` folder.
3. Configure the environment according to `flow_soft/requirements.txt`.
4. Run `process/main.py` to read the raw data and perform data cleaning, feature extraction, and other processing steps.

We recommend using the preprocessed data we provide.

# Model Training
We have provided trained model weights (including results from pre-training and supervised fine-tuning) in the `files/model/ours` folder.

If you want to rerun the experiments, please run `flow_soft/model/DL/run.sh`, which includes execution commands for self-supervised pre-training (first stage) and supervised fine-tuning (second stage). If you need to modify parameters, please refer to the parameter settings in `flow_soft/model/DL/config_init.py` to modify the execution commands.

# Beat Flow Game Framework

We provide a download link for the [Beat Flow Unity project](https://pan.baidu.com/s/1mC0dF0e4YMc00xTlLtmwJw?pwd=mb97).
The `Timer.cs` file contains the communication code between the ICO device and the PC. If you need to communicate, please modify the IP address in it.