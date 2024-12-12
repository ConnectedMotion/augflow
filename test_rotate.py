

import os
import logging
from augflow.pipeline import Pipeline
import augflow.utils.configs as config
from augflow.utils.configs import example_custom_rotate_config

logging.basicConfig(level=logging.INFO)

experiment_id = 'exp1'


pipe = Pipeline()


pipe.task(
    format='yolo',
    dataset_path='dataset_path_contains_data_dot_yaml_images_labels'
)

pipe.fuse(
    source_id='root',
    type='rotate',
    id='aug_rotate',
    config=example_custom_rotate_config,
    merge=True
)


pipe.out(
    format='yolo',
    output_path=f'output_path/{experiment_id}',
    ignore_masks=False,
    visualize_annotations=True
)
