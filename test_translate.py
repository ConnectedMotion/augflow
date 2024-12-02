# pipeline_code_translate.py

import os
import logging
from augflow.pipeline import Pipeline
import augflow.utils.configs as config

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Define the experiment identifier
experiment_id = 'exp_translate_example'

# Ensure output directories exist
os.makedirs('visualize/visualize_translation_custom', exist_ok=True)
os.makedirs('raw_images/augmented_images_translation_custom', exist_ok=True)

# Initialize the AugFlow pipeline
pipe = Pipeline()

# Configure the pipeline task for YOLO format, specifying the dataset path containing images and labels
pipe.task(
    format='yolo',
    dataset_path='/home/omar/Videos/test_datasets/seg.yolov8/train/'
)

# Custom Translate Configuration
custom_translate_config = {
    'modes': ['targeted'],  # Applying targeted mode only
    'focus_categories': ['3_dent', '6_scratch'],  # Custom focus categories
    'translate_probability': 1.0,
    'min_translate_x': -0.3,
    'max_translate_x': 0.3,
    'min_translate_y': -0.3,
    'max_translate_y': 0.3,
    'num_translations_per_image': 3,
    'max_clipped_area_per_category': None,  # Defaults will be used
    'random_seed': 42,
    'enable_translation': True,
    'visualize_overlays': True,
    'output_visualizations_dir': 'visualize/visualize_translation_custom',
    'output_images_dir': 'raw_images/augmented_images_translation_custom',
}

pipe.fuse(
    source_id='root',
    type='translate',
    id='aug_translate',
    config=custom_translate_config,
    merge=True
)

# Define the output configuration for the pipeline, specifying YOLO format and the destination path
pipe.out(
    format='yolo',
    output_path=f'/home/omar/Desktop/aug100/res/{experiment_id}',
    ignore_masks=False,
    visualize_annotations=True
)
