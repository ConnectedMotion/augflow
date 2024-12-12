import os
import logging
from augflow.pipeline import Pipeline
import augflow.utils.configs as config
from augflow.augmentations.cutout import CutoutAugmentation  # Ensure correct import path

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Define the experiment identifier
experiment_id = 'exp_6_source-cocseg-out-yoloseg'

# Ensure output directories exist
os.makedirs('visualize/visualize_cutout_custom', exist_ok=True)
os.makedirs('raw_images/augmented_images_cutout_custom', exist_ok=True)

# Initialize the AugFlow pipeline
pipe = Pipeline()

# Configure the pipeline task for YOLO format, specifying the dataset path containing images and labels
pipe.task(
    format='yolo',
    dataset_path='/home/omar/Videos/test_datasets/seg.yolov8/train/'
)

# Custom Cutout Configuration
custom_cutout_config = {
    'modes': ['targeted'],  # Applying targeted mode only
    'focus_categories': ['3_dent', '6_scratch'],  # Custom focus categories
    'cutout_probability': 1,
    'num_augmented_images': 3,
    'num_cutouts_per_image': 1,
    'cutout_size_percent': ((0.1, 0.3), (0.1, 0.3)),  # Adjust as needed
    'margin_percent': 0.05,
    'max_shift_percent': 1,
    'shift_steps': 200,
    'max_clipped_area_per_category': {},  # Defaults will be used
    'random_seed': 42,
    'enable_cutout': True,
    'visualize_overlays': True,
    'output_visualizations_dir': 'visualize/visualize_cutout_custom',
    'output_images_dir': 'raw_images/augmented_images_cutout_custom',
}

pipe.fuse(
    source_id='root', 
    type='cutout', 
    id='aug_cutout', 
    config=custom_cutout_config,  
    merge=True
)

# Define the output configuration for the pipeline, specifying YOLO format and the destination path
pipe.out(
    format='yolo',
    output_path=f'/home/omar/Desktop/aug100/res/{experiment_id}',
    ignore_masks=False,
    visualize_annotations=True
)
