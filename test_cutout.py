# Example usage
import os
import logging
from augflow.pipeline import Pipeline
from augflow.utils.configs import cutout_default_config

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Define the experiment identifier
experiment_id = 'exp_cutout_example'

# Ensure output directories exist
os.makedirs('visualize/visualize_cutout_custom', exist_ok=True)
os.makedirs('raw_images/augmented_images_cutout_custom', exist_ok=True)

# Initialize the AugFlow pipeline
pipe = Pipeline()

# Configure the pipeline task for YOLO format
pipe.task(
    format='yolo',
    dataset_path='/home/omar/Downloads/coco128-seg/train/',
)

# Custom Cutout Configuration
cutout_configs = {
    'modes': ['targeted'],
    'focus_categories': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light'],
    'cutout_probability': 1.0,
    'num_augmented_images': 5,
    'margin_percent': 0.05,
    'max_shift_percent': 1.0,
    'shift_steps': 200,
    'max_clipped_area_per_category': None,
    'random_seed': 42,
    'enable_cutout': True,
    'visualize_overlays': True,
    'output_visualizations_dir': 'visualize/visualize_cutout_custom',
    'output_images_dir': 'raw_images/augmented_images_cutout_custom',
    'allowed_shifts': ['up', 'down', 'left', 'right'],
    'area_reduction_threshold': 0.1,  # 10%
}

# Apply the cutout augmentation with specified parameters
pipe.fuse(
    source_id='root',
    min_relative_area=0.025,
    min_width=200,
    min_height=200,
    type='cutout',
    id='aug_cutout',
    config=cutout_configs,
    merge=True,
)

# Define the output configuration for the pipeline
pipe.out(
    format='yolo',
    output_path=f'/home/omar/Desktop/aug100/res/{experiment_id}',
    ignore_masks=False,
    visualize_annotations=True,
)
