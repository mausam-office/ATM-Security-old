import numpy as np
import os

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

# split data into training and testing set manually
'''
import os, random, shutil
try:
    os.mkdir('dataset/train')
    os.mkdir('dataset/test')
except:
    print("train test directory already exits.")


image_paths = os.listdir('dataset/images')
random.shuffle(image_paths)


for i, image_path in enumerate(image_paths):
  if i < int(len(image_paths) * 0.8):
    try:
        shutil.copy(f'dataset/images/{image_path}', 'dataset/train')
        shutil.copy(f'dataset/annotations/{image_path.replace("jpg", "xml")}', 'dataset/train')
    except:
        shutil.copy(f'dataset/images/{image_path}', 'dataset/train')
        shutil.copy(f'dataset/annotations/{image_path.replace("jpeg", "xml")}', 'dataset/train')
  else:
    try:
        shutil.copy(f'dataset/images/{image_path}', 'dataset/test')
        shutil.copy(f'dataset/annotations/{image_path.replace("jpg", "xml")}', 'dataset/test')
    except:
        shutil.copy(f'dataset/images/{image_path}', 'dataset/test')
        shutil.copy(f'dataset/annotations/{image_path.replace("jpeg", "xml")}', 'dataset/test')
'''

# select model architecture
spec = model_spec.get('efficientdet_lite0')

# Load the dataset
train_data = object_detector.DataLoader.from_pascal_voc('dataset/train', 'dataset/train', ['log'])
validation_data = object_detector.DataLoader.from_pascal_voc('dataset/test', 'dataset/test', ['log'])

# Training of tensorflow lite model 
model = object_detector.create(train_data, model_spec=spec, epochs = 10, batch_size=8, train_whole_model=True, validation_data=validation_data)

# Model evaluation on test data
model.evaluate(validation_data)

# Export tensorflow model
model.export(export_dir='./models/model_10_8_mm')
