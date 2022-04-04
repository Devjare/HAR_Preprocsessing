import math
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

def remove_gravity(data):
  # TODO: HOW GRAVITY EXTRACTION WORKS.
  # IF IT STARTS IN 0, what's
  alpha = 0.8

  gravity = [0, 0, 0]
  accNoGravity = [[], [], []]

  for idx in data["Idx"]:
    x = data['x'][idx]
    y = data['y'][idx]
    z = data['z'][idx]

    gravity[0] = alpha * gravity[0] + (1 - alpha) * x
    gravity[1] = alpha * gravity[1] + (1 - alpha) * y
    gravity[2] = alpha * gravity[2] + (1 - alpha) * z

    
    accNoGravity[0].append(x - gravity[0])
    accNoGravity[1].append(y - gravity[1]) 
    accNoGravity[2].append(z - gravity[2])

  return accNoGravity[0], accNoGravity[1], accNoGravity[2]


def segment_data(segment_size, data):
  # Extract features from each segment.
  # Each segment will be preprocesseed and features will be saved.
  # With an array segment_nmbr  is taken by index.
  # Not necesary to complete the last segment which may not have the 
  #   same stablished  size
  segments = {}
  segment_number = 0
  for i in range(0, len(data), segment_size):
    segment_init = i
    segment = data.iloc[i:i+segment_size]
    # unique_classes = segment["gt"].unique()
    # Get the most freuent class, and select it as the segment class.
    most_freq_class = segment["gt"].value_counts().idxmax()
    segments[segment_number] = {
            'init': segment_init,
            'size': len(segment),
            'data': segment,
            'class': most_freq_class
        }

    # Preprocess segment(Extract features)
    segment_number += 1

  return segments


def get_magnitudes(data, nograv=False):
  magnitudes = []
  for i in range(len(data)):
    x = data["x_nograv" if nograv else "x"].iloc[i]
    y = data["y_nograv" if nograv else "y"].iloc[i]
    z = data["z_nograv" if nograv else "z"].iloc[i]

    magnitudes.append(math.sqrt((x ** 2) + (y ** 2) + (z ** 2)))

  return magnitudes


def get_features(data):
  # Get mean, std, percentils, etc.
  return  {
      'x_mean': np.mean(data["x"]),
      'y_mean': np.mean(data["y"]),
      'z_mean': np.mean(data["z"]),
      'x_max': np.max(data["x"]),
      'y_max': np.max(data["y"]),
      'z_max': np.max(data["z"]),
      'x_min': np.min(data["x"]),
      'y_min': np.min(data["y"]),
      'z_min': np.min(data["z"]),
      'x_std': np.std(data["x"]),
      'y_std': np.std(data["y"]),
      'z_std': np.std(data["z"]),
      'x_nograv_mean': np.mean(data["x_nograv"]),
      'y_nograv_mean': np.mean(data["y_nograv"]),
      'z_nograv_mean': np.mean(data["z_nograv"]),
      'x_nograv_max': np.max(data["x_nograv"]),
      'y_nograv_max': np.max(data["y_nograv"]),
      'z_nograv_max': np.max(data["z_nograv"]),
      'x_nograv_min': np.min(data["x_nograv"]),
      'y_nograv_min': np.min(data["y_nograv"]),
      'z_nograv_min': np.min(data["z_nograv"]),
      'x_nograv_std': np.std(data["x_nograv"]),
      'y_nograv_std': np.std(data["y_nograv"]),
      'z_nograv_std': np.std(data["z_nograv"]),
      'm_mean': np.mean(data["mag"]),
      'm_max': np.max(data["mag"]),
      'm_min': np.min(data["mag"]),
      'm_std': np.std(data["mag"])
  }
