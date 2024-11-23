
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


class ColorShiftAugmentation(Augmentation):
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
            'rgb_shift_probability': 0.5,
            'r_shift_limit': (-50, 50),
            'g_shift_limit': (-50, 50),
            'b_shift_limit': (-50, 50),
            'hue_saturation_probability': 0.5,
            'hue_shift_limit': (-10, 10),
            'saturation_shift_limit': (-15, 15),
            'grayscale_probability': 0.1,
            'num_augmented_images': 1,
            'enable_color_shift': True,
            'output_images_dir': 'augmented_images_color_shift',
        }
        for key, value in default_config.items():
            self.config.setdefault(key, value)

    def apply(self, dataset: UnifiedDataset, output_dim: Optional[tuple] = None) -> UnifiedDataset:
        if not self.config.get('enable_color_shift', True):
            logging.info("Color shift augmentation is disabled.")
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

                # Apply RGB Shift
                if random.random() < self.config['rgb_shift_probability']:
                    r_shift = random.randint(*self.config['r_shift_limit'])
                    g_shift = random.randint(*self.config['g_shift_limit'])
                    b_shift = random.randint(*self.config['b_shift_limit'])
                    transformed_image = self.rgb_shift(transformed_image, r_shift, g_shift, b_shift)
                    logging.debug(f"Applied RGB shift with R:{r_shift}, G:{g_shift}, B:{b_shift}.")

                # Apply Hue and Saturation Shift
                if random.random() < self.config['hue_saturation_probability']:
                    hue_shift = random.randint(*self.config['hue_shift_limit'])
                    sat_shift = random.randint(*self.config['saturation_shift_limit'])
                    transformed_image = self.hue_saturation_shift(transformed_image, hue_shift, sat_shift)
                    logging.debug(f"Applied hue shift {hue_shift} and saturation shift {sat_shift}.")

                # Apply Grayscale
                if random.random() < self.config['grayscale_probability']:
                    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
                    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_GRAY2BGR)
                    logging.debug("Converted image to grayscale.")

                # Generate new filename
                new_filename = f"{os.path.splitext(os.path.basename(img.file_name))[0]}_color_shift_{uuid.uuid4().hex}.jpg"
                output_image_path = os.path.join(output_images_dir, new_filename)

                # Save transformed image
                save_success = save_image(transformed_image, output_image_path)
                if not save_success:
                    logging.error(f"Failed to save color-shifted image '{output_image_path}'. Skipping.")
                    continue
                logging.info(f"Saved color-shifted image '{new_filename}' with ID {image_id_offset}.")

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

        logging.info(f"Color shift augmentation completed. Total augmented images: {len(augmented_dataset.images)}.")
        return augmented_dataset

    @staticmethod
    def rgb_shift(image, r_shift, g_shift, b_shift):
        B, G, R = cv2.split(image)
        R = cv2.add(R, r_shift)
        G = cv2.add(G, g_shift)
        B = cv2.add(B, b_shift)
        shifted_image = cv2.merge([B, G, R])
        return shifted_image

    @staticmethod
    def hue_saturation_shift(image, hue_shift, sat_shift):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 0] += hue_shift
        hsv[:, :, 1] += sat_shift
        hsv[:, :, 0][hsv[:, :, 0] > 179] -= 179
        hsv[:, :, 0][hsv[:, :, 0] < 0] += 179
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        hsv[:, :, 1][hsv[:, :, 1] < 0] = 0
        hsv = np.array(hsv, dtype=np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
