import numpy as np
import os
import time

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)


# select model architecture
#spec = model_spec.get('efficientdet_lite0')
spec = object_detector.EfficientDetSpec(
  model_name='efficientdet-lite0',
  uri='https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1', 
  hparams={'max_instances_per_image': 50})

# Load the dataset
labels = ['hammer', 'mask', 'person', 'smoke', 'knife', 'robbery mask', 'gun', 'driller', 'keyboard', 'phone', 'screwdriver', 'laptop', 'mouse', 'helmet']
train_data = object_detector.DataLoader.from_pascal_voc(images_dir='dataset/train', annotations_dir='dataset/train', label_map=labels)
validation_data = object_detector.DataLoader.from_pascal_voc(images_dir='dataset/test', annotations_dir='dataset/test', label_map=labels)

# Training of tensorflow lite model 
model = object_detector.create(train_data, model_spec=spec, epochs = 1000, batch_size=16, train_whole_model=True, validation_data=validation_data, do_train=True)

time.sleep(10)
# Model evaluation on test data
model.evaluate(validation_data)

# Export tensorflow model
model.export(export_dir='./models/atm_best_16_lite0_mm')

