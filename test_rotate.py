# pipeline_code.py

import os
import logging
from augflow.pipeline import Pipeline
import augflow.utils.configs as config

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Define the experiment identifier
experiment_id = 'exp_rotate_example'

# Ensure output directories exist
os.makedirs('visualize/visualize_rotation_custom', exist_ok=True)
os.makedirs('raw_images/augmented_images_rotation_custom', exist_ok=True)

# Initialize the AugFlow pipeline
pipe = Pipeline()

# Configure the pipeline task for YOLO format, specifying the dataset path containing images and labels
pipe.task(
    format='yolo',
    dataset_path='/home/omar/Downloads/coco128-seg/train/'
)

# Custom Rotate Configuration
custom_rotate_config = {
    'modes': ['targeted'],  # Applying targeted mode only
    'focus_categories': ['person', 'bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light'],
    'rotation_probability': 1.0,
    'rotation_point_modes': ['center', 'random'],
    'rotation_angle_modes': ['predefined_set'],
    'angle_parameters': {
        'predefined_set': [-30, -15, 15, 30],
        'alpha': 30
    },
    'num_rotations_per_image': 3,
    'max_clipped_area_per_category': None,  # Defaults will be used
    'random_seed': 42,
    'enable_rotation': True,
    'visualize_overlays': True,
    'output_visualizations_dir': 'visualize/visualize_rotation_custom',
    'output_images_dir': 'raw_images/augmented_images_rotation_custom',
}

pipe.fuse(
    source_id='root',
    type='rotate',
    id='aug_rotate',
    config=custom_rotate_config,
    merge=True
)

# Define the output configuration for the pipeline, specifying YOLO format and the destination path
pipe.out(
    format='yolo',
    output_path=f'/home/omar/Desktop/aug100/res/{experiment_id}',
    ignore_masks=False,
    visualize_annotations=True
)
