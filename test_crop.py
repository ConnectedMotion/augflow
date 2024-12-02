import os
import logging
from augflow.pipeline import Pipeline
import augflow.utils.configs as config

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Define the experiment identifier
experiment_id = 'exp_crop_example'

# Ensure output directories exist
os.makedirs('visualize/visualize_cropping_custom', exist_ok=True)
os.makedirs('raw_images/augmented_images_cropping_custom', exist_ok=True)

# Initialize the AugFlow pipeline
pipe = Pipeline()

# Configure the pipeline task for YOLO format, specifying the dataset path containing images and labels
pipe.task(
    format='yolo',
    dataset_path='/home/omar/Videos/seg.yolov8/train'
)

# Custom Crop Configuration
custom_crop_config = {
    'modes': ['targeted'],  # Applying targeted mode
    'focus_categories': ['3_dent', '6_scratch'],  # Custom focus categories
    'num_crops_per_image': 20,
    'margin_percent': 0.05,  # Adjusted from 'margin' to 'margin_percent'
    'crop_size_percent': ((0.4, 1.0), (0.4, 1.0)),  # Adjusted to ensure minimum crop size
    'max_clipped_area_per_category': None,  # Defaults will be used
    'random_seed': 42,
    'enable_cropping': True,
    'visualize_overlays': True,
    'output_visualizations_dir': 'visualize/visualize_cropping_custom',
    'output_images_dir': 'raw_images/augmented_images_cropping_custom',
}

pipe.fuse(
    source_id='root',
    min_relative_area=0.05,
    type='crop',
    id='aug_crop',
    config=custom_crop_config,
    merge=True
)

# Define the output configuration for the pipeline, specifying YOLO format and the destination path
pipe.out(
    format='yolo',
    output_path=f'/home/omar/Desktop/aug100/res/{experiment_id}',
    ignore_masks=False,
    visualize_annotations=True
)


