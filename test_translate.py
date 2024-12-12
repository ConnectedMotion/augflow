# pipeline_code_translate.py

import os
import logging
from augflow.pipeline import Pipeline
from augflow.utils.configs import example_custom_translate_config


logging.basicConfig(level=logging.INFO)

experiment_id = 'exp1'

pipe = Pipeline()

pipe.task(
    format='yolo',
    dataset_path='dataset_path_contains_data_dot_yaml_images_labels'
)



pipe.fuse(
    source_id='root',
    type='translate',
    id='aug_translate',
    config=example_custom_translate_config,
    merge=True
)

pipe.out(
    format='yolo',
    output_path=f'output_path/{experiment_id}',
    ignore_masks=False,
    visualize_annotations=True
)
