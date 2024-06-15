
#import tqdm
import seaborn as sns
import sys
import random
import os
from tqdm import tqdm
from pathlib import PosixPath
import argparse

# Set FFmpeg logging level to 'error'
os.environ['FFMPEG_LOG_LEVEL'] = 'error'

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

# Some modules to display an animation using imageio.
import imageio
from IPython import display
from urllib import request
from tensorflow_docs.vis import embed
import matplotlib.pyplot as plt
import matplotlib as mpl
import PIL
from PIL import Image
import mediapy as media
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip


import keras
import tensorflow as tf
import tensorflow_hub as hub

# Import the MoViNet model from TensorFlow Models (tf-models-official) for the MoViNet model
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

"""## Building the model"""

def build_streaming_model(model_id = 'a3', num_frames = 12 , batch_size = 64, learning_rate = 0.001, dropout_rate= 0.3, trainable_layers = 0, dataset = '_NoAug' ):

    use_positional_encoding = model_id in {'a3', 'a4', 'a5'}

    # Set resolution and frames per second based on model_id
    model_specs = {
    'a0': {'resolution': 172, 'fps': 5},
    'a1': {'resolution': 172, 'fps': 5},
    'a2': {'resolution': 224, 'fps': 5},
    'a3': {'resolution': 256, 'fps': 12},
    'a4': {'resolution': 290, 'fps': 8},
    'a5': {'resolution': 320, 'fps': 12}
    }

    # Retrieve settings based on model_id
    model_settings = model_specs.get(model_id)  # Returns None if model_id not found
    RESOLUTION = model_settings['resolution'] if model_settings else None
    FRAMES = model_settings['fps'] if model_settings else None

    # Create backbone and model.
    backbone = movinet.Movinet(
        model_id=model_id,
        causal=True,
        conv_type='2plus1d',
        se_type='2plus3d',
        activation='hard_swish',
        gating_activation='hard_sigmoid',
        use_positional_encoding=use_positional_encoding,
        use_external_states=True,
    )

    model = movinet_model.MovinetClassifier(
        backbone,
        num_classes=2,
        output_states=True)

    # Create your example input here.
    # Refer to the paper for recommended input shapes.
    inputs = tf.ones([1, 1, 224, 224, 3])

    # [Optional] Build the model and load a pretrained checkpoint.
    model.build(inputs.shape)

    # Load weights from the checkpoint to the rebuilt model
    models_dir = f'./MoViNet4Violence-Detection/trained_models_dropout_autolr_trlayers{dataset}/'

    checkpoint_dir = f'{models_dir}/movinet_{model_id}_{num_frames}fps_{batch_size}bs_{learning_rate}lr_{dropout_rate}dr_{trainable_layers}tl/'
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    return model, RESOLUTION, FRAMES


"""### Animating Inference on Streaming"""

CLASSES = ['Fight','No_Fight']

def get_top_k(probs, k=2, label_map=CLASSES):
    """Outputs the top k model labels and probabilities on the given video."""
    top_predictions = tf.argsort(probs, axis=-1, direction='DESCENDING')[:k]
    top_labels = tf.gather(label_map, top_predictions, axis=-1)
    top_labels = [label.decode('utf8') for label in top_labels.numpy()]
    top_probs = tf.gather(probs, top_predictions, axis=-1).numpy()
    return tuple(zip(top_labels, top_probs))


# Get top_k labels and probabilities predicted using MoViNets streaming model
def get_top_k_streaming_labels(probs, k=2, label_map=CLASSES):
  """Returns the top-k labels over an entire video sequence.

  Args:
    probs: probability tensor of shape (num_frames, num_classes) that represents
      the probability of each class on each frame.
    k: the number of top predictions to select.
    label_map: a list of labels to map logit indices to label strings.

  Returns:
    a tuple of the top-k probabilities, labels, and logit indices
  """
  top_categories_last = tf.argsort(probs, -1, 'DESCENDING')[-1, :1]
  # Sort predictions to find top_k
  categories = tf.argsort(probs, -1, 'DESCENDING')[:, :k]
  categories = tf.reshape(categories, [-1])

  counts = sorted([
      (i.numpy(), tf.reduce_sum(tf.cast(categories == i, tf.int32)).numpy())
      for i in tf.unique(categories)[0]
  ], key=lambda x: x[1], reverse=True)

  top_probs_idx = tf.constant([i for i, _ in counts[:k]])
  top_probs_idx = tf.concat([top_categories_last, top_probs_idx], 0)
  # find unique indices of categories
  top_probs_idx = tf.unique(top_probs_idx)[0][:k+1]
  # top_k probabilities of the predictions
  top_probs = tf.gather(probs, top_probs_idx, axis=-1)
  top_probs = tf.transpose(top_probs, perm=(1, 0))
  # collect the labels of top_k predictions
  top_labels = tf.gather(label_map, top_probs_idx, axis=0)
  # decode the top_k labels
  top_labels = [label.decode('utf8') for label in top_labels.numpy()]

  return top_probs, top_labels, top_probs_idx



# Plot top_k predictions at a given time step
def plot_streaming_top_preds_at_step(
    top_probs,
    top_labels,
    step=None,
    image=None,
    legend_loc='lower left',
    duration_seconds=10,
    figure_height=500,
    playhead_scale=0.8,
    grid_alpha=0.3):

  """Generates a plot of the top video model predictions at a given time step.

  Args:
    top_probs: a tensor of shape (k, num_frames) representing the top-k
      probabilities over all frames.
    top_labels: a list of length k that represents the top-k label strings.
    step: the current time step in the range [0, num_frames].
    image: the image frame to display at the current time step.
    legend_loc: the placement location of the legend.
    duration_seconds: the total duration of the video.
    figure_height: the output figure height.
    playhead_scale: scale value for the playhead.
    grid_alpha: alpha value for the gridlines.

  Returns:
    A tuple of the output numpy image, figure, and axes.
  """
  # find number of top_k labels and frames in the video
  num_labels, num_frames = top_probs.shape
  if step is None:
    step = num_frames
  # Visualize frames and top_k probabilities of streaming video
  fig = plt.figure(figsize=(6.5, 7), dpi=300)
  gs = mpl.gridspec.GridSpec(8, 1)
  ax2 = plt.subplot(gs[:-3, :])
  ax = plt.subplot(gs[-3:, :])
  # display the frame
  if image is not None:
    ax2.imshow(image, interpolation='nearest')
    ax2.axis('off')
  # x-axis (frame number)
  preview_line_x = tf.linspace(0., duration_seconds, num_frames)
  # y-axis (top_k probabilities)
  preview_line_y = top_probs

  line_x = preview_line_x[:step+1]
  line_y = preview_line_y[:, :step+1]

  for i in range(num_labels):
    ax.plot(preview_line_x, preview_line_y[i], label=None, linewidth='1.5',
            linestyle=':', color='gray')
    ax.plot(line_x, line_y[i], label=top_labels[i], linewidth='2.0')


  ax.grid(which='major', linestyle=':', linewidth='1.0', alpha=grid_alpha)
  ax.grid(which='minor', linestyle=':', linewidth='0.5', alpha=grid_alpha)

  min_height = tf.reduce_min(top_probs) * playhead_scale
  max_height = tf.reduce_max(top_probs)
  ax.vlines(preview_line_x[step], min_height, max_height, colors='red')
  ax.scatter(preview_line_x[step], max_height, color='red')

  ax.legend(loc=legend_loc)

  plt.xlim(0, duration_seconds)
  plt.ylabel('Probability')
  plt.xlabel('Time (s)')
  plt.yscale('log')

  fig.tight_layout()
  fig.canvas.draw()

  data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()

  figure_width = int(figure_height * data.shape[1] / data.shape[0])
  image = PIL.Image.fromarray(data).resize([figure_width, figure_height])
  image = np.array(image)

  return image

# Plotting top_k predictions from MoViNets streaming model
def plot_streaming_top_preds(
    probs,
    video,
    top_k=2,
    video_fps=25.,
    figure_height=500,
    use_progbar=True):

  """Generates a video plot of the top video model predictions.

  Args:
    probs: probability tensor of shape (num_frames, num_classes) that represents
      the probability of each class on each frame.
    video: the video to display in the plot.
    top_k: the number of top predictions to select.
    video_fps: the input video fps.
    figure_fps: the output video fps.
    figure_height: the height of the output video.
    use_progbar: display a progress bar.

  Returns:
    A numpy array representing the output video.
  """
  # select number of frames per second
  #video_fps = 8.
  # select height of the image
  figure_height = 500
  # number of time steps of the given video
  steps = video.shape[0]
  # estimate duration of the video (in seconds)
  duration = steps / video_fps
  # estimate top_k probabilities and corresponding labels
  top_probs, top_labels, _ = get_top_k_streaming_labels(probs, k=top_k)

  images = []
  step_generator = tqdm(range(steps)) if use_progbar else range(steps)
  for i in step_generator:
    image = plot_streaming_top_preds_at_step(
        top_probs=top_probs,
        top_labels=top_labels,
        step=i,
        image=video[i],
        duration_seconds=duration,
        figure_height=figure_height,
    )
    images.append(image)

  return np.array(images)

def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded.
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 15):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

  need_length = 1 + (n_frames - 1) * frame_step

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)

  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  result.append(format_frames(frame, output_size))

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
      frame = format_frames(frame, output_size)
      result.append(frame)
    else:
      result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)[..., [2, 1, 0]]

  return result


def video_to_gif_tensor(video_path, image_size=(RESOLUTION, RESOLUTION), fps=FRAMES):
    """
    Processes frames from a video file, saves them as a GIF in the same directory, and loads the GIF as a TensorFlow tensor.

    Args:
      video_path: String path to the input video file.
      image_size: Tuple indicating the size to which each frame should be resized.
      fps: Frames per second to be used in the GIF.

    Returns:
      A TensorFlow tensor representing the loaded GIF.
    """
    # Generate the gif_path in the same directory with a .gif extension
    gif_path = os.path.splitext(video_path)[0] + '.gif'

    # Assume frames_from_video_file is a function that extracts frames from video
    images = frames_from_video_file(video_path, n_frames=fps)  # function to be defined or replaced

    # Convert images to uint8 and save as GIF
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)  # Proper scaling to 255
    imageio.mimsave(gif_path, converted_images, fps=fps)

    # Load the GIF file into a TensorFlow tensor
    raw = tf.io.read_file(gif_path)
    video = tf.io.decode_gif(raw)
    video = tf.image.resize(video, image_size)
    video = tf.cast(video, tf.float32) / 255.0  # Normalize to [0,1]

    return video

def streaming_inference(video, init_states, inference_model):
    """
    Perform streaming inference on a video using our pre-trained model.

    Args:
        video (tf.Tensor): Video frames to perform inference on.
        states (dict): Initial states for the model.

    Returns:
        tf.Tensor: Probabilities estimated by the model.
    """
    images = tf.split(video[tf.newaxis], video.shape[0], axis=1)
    all_logits = []

    # To run on a video, pass in one frame at a time
    states = init_states
    for image in images:
      # predictions for each frame
      logits, states = inference_model({**states, 'image': image})
      all_logits.append(logits)

    # concatenating all the logits
    logits = tf.concat(all_logits, 0)
    # estimating probabilities
    probs = tf.nn.softmax(logits, axis=-1)

    final_probs = probs[-1]
    print('Top_k predictions and their probabilities\n')
    for label, p in get_top_k(final_probs):
        print(f'{label:20s}: {p:.3f}')

    return probs


def main(video_path, model_id=None, bs=None, lr = None, dr=None, trly=None):
    # If any parameter is missing, ignore all custom parameters and use defaults
    if any(param is None for param in [model_id, bs, lr, dr, trly]):
        model_id, fps, bs,lr, dr, trly = 'a3', 12, 64,0.001, 0.3, 0
        print("Missing some parameters. Using default settings.")

    model, RESOLUTION, FRAMES = build_streaming_model(model_id, bs,lr, dr, trly)
    video_frames = video_to_gif_tensor(video_path, FRAMES)
    init_states_fn = model.init_states
    init_states = init_states_fn(tf.shape(tf.ones(shape=[1, 1, RESOLUTION, RESOLUTION, 3])))
    probs = streaming_inference(video_frames, init_states, model)
    plot_video = plot_streaming_top_preds(probs, video_frames, video_fps=FRAMES)
    output_path = os.path.join(os.path.dirname(video_path), f"inference_{os.path.basename(video_path)}")
    plot_video.write_videofile(output_path, codec='libx264')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MoViNet inference on a video file.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--model_id", type=str, help="Model ID to use")
    parser.add_argument("--lr", type=int, help="Learning Rate")
    parser.add_argument("--bs", type=int, help="Batch size")
    parser.add_argument("--dr", type=float, help="Dropout rate")
    parser.add_argument("--trly", type=int, help="Number of trainable layers")
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print("Error: video file does not exist.")
        sys.exit(1)

    # Pass parameters to main, using None as default if not specified
    main(args.video_path, args.model_id, args.fps, args.bs, args.dr, args.trly)
