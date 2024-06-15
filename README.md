# MoViNets for Violence Detection in Video Streaming

## Overview

This project focuses on leveraging MoViNet models to effectively detect violence in video streams. Utilizing transfer learning and fine-tuning techniques, the goal is to achieve a high-performance model that operates efficiently on edge devices with limited resources.

## Key Features

- **Model Training**: Utilizes MoViNets, an advanced architecture from Google Research (Kondratyuk et al., 2021) known for its efficiency in mobile and edge computing environments. Employs transfer-learning on this pre-trained models on human action recognition to enhance learning efficacy and reduce the necessity for extensive computational resources. The code is available in _'movinet_training.ipynb'_. The training and the evaluation metrics are availible in the folders  above, as well as a futher analysis on those results. 
- **Real-time Operation**: Optimized for real-time applications, ensuring swift and accurate violence detection, the inference can be performed through _'movinet_inference.ipynb'_

## Example of the visual interface for the inference
<p align="center">
  <img src="https://github.com/engares/MoViNets-for-Violence-Detection-in-Live-Video-Streaming/assets/126718587/35e32b64-d29a-4a80-b7d5-728577704bcb" alt="Fight_2">
</p>

_More examples on the 'example_videos' folder_

## Requirements

- Python 3.10+
- TensorFlow 2.15+
- linux distro
- Other dependencies listed in `requirements.txt`

## Usage

_You can also use directly the [Colab Notebook here](https://colab.research.google.com/github/engares/MoViNets-for-Violence-Detection-in-Live-Video-Streaming/blob/main/movinet_inference.ipynb)_

1. Clone the repository:
   ```bash
   git clone https://github.com/engares/MoViNets-for-Violence-Detection-in-Live-Video-Streaming.git
2. Install the required packages:
   ```bash
   pip install -r requeriments.txt
   sudo apt update && sudo apt install -y ffmpeg
3. Download the models (1.8 GB, this may take time)
   ```bash
   git clone https://huggingface.co/engares/MoViNet4Violence-Detection
   
4. Open the 'movinet_inference.ipynb' notebook and select one of the trained models based on the hyperparameters. (The best model is chosen by default)
   ```bash
   model_id = 'a3'
   fps = 12
   bs = 64
   lr = 0.001
   dr = 0.3
   trly = 0

   inference_model, RESOLUTION = build_streaming_model(model_id, num_frames = fps, batch_size = bs, learning_rate = lr, dropout_rate = dr, trainable_layers = trly, dataset = dataname )
  
5. Upload your local video/stream fragment (Mp4, avi, etc.)
   ```bash
   video_path = "./test_videos/test2.avi"

   
   
