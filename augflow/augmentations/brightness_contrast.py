
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



class BrightnessContrastAugmentation(Augmentation):
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
            'brightness_probability': 0.5,
            'brightness_limit': (-0.2, 0.2),  # Fractional change
            'contrast_probability': 0.5,
            'contrast_limit': (-0.2, 0.2),    # Fractional change
            'exposure_probability': 0.5,
            'exposure_limit': (-0.2, 0.2),    # Fractional change
            'num_augmented_images': 1,
            'enable_brightness_contrast': True,
            'output_images_dir': 'augmented_images_brightness_contrast',
        }
        for key, value in default_config.items():
            self.config.setdefault(key, value)

    def apply(self, dataset: UnifiedDataset, output_dim: Optional[tuple] = None) -> UnifiedDataset:
        if not self.config.get('enable_brightness_contrast', True):
            logging.info("Brightness/Contrast augmentation is disabled.")
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

                # Apply Brightness
                if random.random() < self.config['brightness_probability']:
                    brightness_factor = 1.0 + random.uniform(*self.config['brightness_limit'])
                    transformed_image = self.adjust_brightness(transformed_image, brightness_factor)
                    logging.debug(f"Applied brightness adjustment with factor {brightness_factor}.")

                # Apply Contrast
                if random.random() < self.config['contrast_probability']:
                    contrast_factor = 1.0 + random.uniform(*self.config['contrast_limit'])
                    transformed_image = self.adjust_contrast(transformed_image, contrast_factor)
                    logging.debug(f"Applied contrast adjustment with factor {contrast_factor}.")

                # Apply Exposure (Gamma Correction)
                if random.random() < self.config['exposure_probability']:
                    gamma = 1.0 + random.uniform(*self.config['exposure_limit'])
                    transformed_image = self.adjust_gamma(transformed_image, gamma)
                    logging.debug(f"Applied exposure adjustment with gamma {gamma}.")

                # Generate new filename
                new_filename = f"{os.path.splitext(os.path.basename(img.file_name))[0]}_brightness_contrast_{uuid.uuid4().hex}.jpg"
                output_image_path = os.path.join(output_images_dir, new_filename)

                # Save transformed image
                save_success = save_image(transformed_image, output_image_path)
                if not save_success:
                    logging.error(f"Failed to save brightness/contrast adjusted image '{output_image_path}'. Skipping.")
                    continue
                logging.info(f"Saved brightness/contrast adjusted image '{new_filename}' with ID {image_id_offset}.")

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

        logging.info(f"Brightness/Contrast augmentation completed. Total augmented images: {len(augmented_dataset.images)}.")
        return augmented_dataset

    @staticmethod
    def adjust_brightness(image, factor):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 2] *= factor
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        hsv = np.array(hsv, dtype=np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def adjust_contrast(image, factor):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab = np.array(lab, dtype=np.float64)
        lab[:, :, 0] *= factor
        lab[:, :, 0][lab[:, :, 0] > 255] = 255
        lab = np.array(lab, dtype=np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def adjust_gamma(image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(image, table)
