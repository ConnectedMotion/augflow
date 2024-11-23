import cv2
import random
import numpy as np
from .base import Augmentation
from augflow.utils.images import load_image, save_image
from augflow.utils.unified_format import UnifiedDataset, UnifiedImage, UnifiedAnnotation
import os
import uuid
import logging
import copy
from typing import Optional, List, Dict


class BlurAugmentation(Augmentation):
    def __init__(self, config=None, task='detection'):
        super().__init__()
        self.config = config or {}
        self.task = task.lower()
        self.config_defaults()
        random.seed(self.config.get('random_seed', 42))

        # Ensure output directories exist
        os.makedirs(self.config['output_images_dir'], exist_ok=True)

    def config_defaults(self):
        default_config = {
            'blur_probability': 0.5,
            'blur_kernel_sizes': [3, 5, 7],
            'median_blur_probability': 0.5,
            'median_blur_kernel_sizes': [3, 5, 7],
            'num_augmented_images': 1,
            'enable_blur': True,
            'output_images_dir': 'augmented_images_blur',
        }
        for key, value in default_config.items():
            self.config.setdefault(key, value)

    def apply(self, dataset: UnifiedDataset, output_dim: Optional[tuple] = None) -> UnifiedDataset:
        if not self.config.get('enable_blur', True):
            logging.info("Blur augmentation is disabled.")
            return UnifiedDataset(images=[], annotations=[], categories=dataset.categories)

        augmented_dataset = UnifiedDataset(
            images=[],
            annotations=[],
            categories=copy.deepcopy(dataset.categories)
        )

        # Get the maximum existing image and annotation IDs
        existing_image_ids = [img.id for img in dataset.images]
        existing_annotation_ids = [ann.id for ann in dataset.annotations]
        image_id_offset = max(existing_image_ids) + 1 if existing_image_ids else 1
        annotation_id_offset = max(existing_annotation_ids) + 1 if existing_annotation_ids else 1

        # Create a mapping from image_id to annotations
        image_id_to_annotations = {}
        for ann in dataset.annotations:
            image_id_to_annotations.setdefault(ann.image_id, []).append(ann)

        output_images_dir = self.config['output_images_dir']

        for img in dataset.images:
            image_path = img.file_name
            image = load_image(image_path)
            if image is None:
                logging.error(f"Failed to load image '{image_path}'. Skipping.")
                continue

            anns = image_id_to_annotations.get(img.id, [])

            for _ in range(self.config['num_augmented_images']):
                transformed_image = image.copy()

                # Apply Blur
                if random.random() < self.config['blur_probability']:
                    ksize = random.choice(self.config['blur_kernel_sizes'])
                    transformed_image = cv2.blur(transformed_image, (ksize, ksize))
                    logging.debug(f"Applied blur with kernel size {ksize}.")

                # Apply Median Blur
                if random.random() < self.config['median_blur_probability']:
                    ksize = random.choice(self.config['median_blur_kernel_sizes'])
                    # Kernel size must be odd and greater than 1
                    if ksize % 2 == 0:
                        ksize += 1
                    transformed_image = cv2.medianBlur(transformed_image, ksize)
                    logging.debug(f"Applied median blur with kernel size {ksize}.")

                # Generate new filename
                new_filename = f"{os.path.splitext(os.path.basename(img.file_name))[0]}_blur_{uuid.uuid4().hex}.jpg"
                output_image_path = os.path.join(output_images_dir, new_filename)

                # Save transformed image
                save_success = save_image(transformed_image, output_image_path)
                if not save_success:
                    logging.error(f"Failed to save blurred image '{output_image_path}'. Skipping.")
                    continue
                logging.info(f"Saved blurred image '{new_filename}' with ID {image_id_offset}.")

                # Create new image entry
                new_img = UnifiedImage(
                    id=image_id_offset,
                    file_name=output_image_path,
                    width=transformed_image.shape[1],
                    height=transformed_image.shape[0]
                )
                augmented_dataset.images.append(new_img)

                # Copy annotations
                for ann in anns:
                    new_ann = copy.deepcopy(ann)
                    new_ann.id = annotation_id_offset
                    new_ann.image_id = image_id_offset
                    augmented_dataset.annotations.append(new_ann)
                    annotation_id_offset += 1

                image_id_offset += 1

        logging.info(f"Blur augmentation completed. Total augmented images: {len(augmented_dataset.images)}.")
        return augmented_dataset
