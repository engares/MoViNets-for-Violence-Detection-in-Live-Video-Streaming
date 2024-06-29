# MoViNets for Violence Detection in Video Streaming

## Overview

This project aims to harness the capabilities of MoViNet models to accurately detect instances of violence in video streams. By employing strategies such as transfer learning and fine-tuning, the objective is to develop a high-performance model that can function efficiently on edge devices, which often have limited computational resources. Specifically, the MoViNet-A3 model has demonstrated promising results, achieving an accuracy rate of 85%. This level of precision underscores the model's potential for real-time applications in environments where quick and reliable video analysis is critical.

## Key Features

- **Model Training**: Utilizes MoViNets, an advanced architecture from Google Research (Kondratyuk et al., 2021) known for its efficiency in mobile and edge computing environments. Employs transfer-learning on this pre-trained models on human action recognition to enhance learning efficacy and reduce the necessity for extensive computational resources. The code is available in _'movinet_training.ipynb'_. The training and the evaluation metrics are availible in the folders  above, as well as a futher analysis on those results. 
- **Real-time Operation**: Optimized for real-time applications, ensuring swift and accurate violence detection, the inference can be performed through _'movinet_inference.ipynb'_

## Example of the visual interface for the inference
<p align="center">
  <img src="https://github.com/engares/MoViNets-for-Violence-Detection-in-Live-Video-Streaming/assets/126718587/35e32b64-d29a-4a80-b7d5-728577704bcb" alt="Fight_2">
  <img src="https://github.com/engares/MoViNets-for-Violence-Detection-in-Live-Video-Streaming/assets/126718587/c51f3226-9d6b-4247-958b-da6e11196c73" alt="Fight_1">

_More examples on the 'example_videos' folder_

## Requirements

- Python 3.10+
- TensorFlow 2.15+
- linux distro
- Other dependencies listed in `requirements.txt`

## Usage

_You can use directly the [Colab Notebook here](https://colab.research.google.com/drive/1O3F8X7JPkqKmz9CruN0Dr3QaXlS5LNLl?usp=sharing)    (RECOMENDED)_

Or you can run it on the python script in a virtual environment:

1. Clone the repository
   ```bash
   git clone https://github.com/engares/MoViNets-for-Violence-Detection-in-Live-Video-Streaming.git
   cd MoViNets-for-Violence-Detection-in-Live-Video-Streaming
2. Install the required packages
   ```bash
   pip install -r requeriments.txt
   sudo apt update && sudo apt install -y ffmpeg
3. Download the models (1.8 GB, this may take time)
   ```bash
   git clone https://huggingface.co/engares/MoViNet4Violence-Detection
   
4. Run 'movinet_inference.py' indicating the path to the video and one ofselecting one of the trained models based on the hyperparameters. (The best model is chosen by default)
   ```bash
   python movinet_inference.py [/path/to/video.mp4] --model_id a3 --lr 0.001 --bs 64 --dr 0.3 --trly 0
The full list of models with its performance metrics is available is on [this .csv](https://github.com/engares/MoViNets-for-Violence-Detection-in-Live-Video-Streaming/blob/main/evaluation_metrics_analysis/model_performance_metrics.csv)


