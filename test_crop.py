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

# Configure the pipeline task for YOLO format
pipe.task(
    format='yolo',
    dataset_path='/home/omar/Downloads/coco128-seg/train/',
)

# Custom Crop Configuration
custom_crop_config = {
    'modes': ['targeted'],  # Applying targeted mode
    'focus_categories': ['person', 'bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light'],
    'num_crops_per_image': 4,
    'margin_percent': 0.05,
    'shift_attempts': 200,  # Number of random shifts to attempt
    'crop_size_percent': ((0.4, 1.0), (0.4, 1.0)),
    'max_clipped_area_per_category': None,
    'random_seed': 42,
    'enable_cropping': True,
    'visualize_overlays': True,
    'output_visualizations_dir': 'visualize/visualize_cropping_custom',
    'output_images_dir': 'raw_images/augmented_images_cropping_custom',
    'crop_iou_threshold': 0.1,
}

# Apply the crop augmentation with specified parameters
pipe.fuse(
    source_id='root',
    min_relative_area=0.025,
    min_width=200,
    min_height=200,
    type='crop',
    id='aug_crop',
    config=custom_crop_config,
    merge=True,
)

# Define the output configuration for the pipeline
pipe.out(
    format='yolo',
    output_path=f'/home/omar/Desktop/aug100/res/{experiment_id}',
    ignore_masks=False,
    visualize_annotations=True,
)
