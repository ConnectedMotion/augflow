
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


class WeatherAugmentation(Augmentation):
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
            'shadow_probability': 0.5,
            'sunflare_probability': 0.3,
            'rain_probability': 0.5,
            'num_augmented_images': 1,
            'enable_weather': True,
            'output_images_dir': 'augmented_images_weather',
        }
        for key, value in default_config.items():
            self.config.setdefault(key, value)

    def apply(self, dataset: UnifiedDataset, output_dim: Optional[tuple] = None) -> UnifiedDataset:
        if not self.config.get('enable_weather', True):
            logging.info("Weather augmentation is disabled.")
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

                # Apply Shadow
                if random.random() < self.config['shadow_probability']:
                    transformed_image = self.add_shadow(transformed_image)
                    logging.debug("Applied shadow effect.")

                # Apply Sunflare
                if random.random() < self.config['sunflare_probability']:
                    transformed_image = self.add_sunflare(transformed_image)
                    logging.debug("Applied sunflare effect.")

                # Apply Rain
                if random.random() < self.config['rain_probability']:
                    transformed_image = self.add_rain(transformed_image)
                    logging.debug("Applied rain effect.")

                # Generate new filename
                new_filename = f"{os.path.splitext(os.path.basename(img.file_name))[0]}_weather_{uuid.uuid4().hex}.jpg"
                output_image_path = os.path.join(output_images_dir, new_filename)

                # Save transformed image
                save_success = save_image(transformed_image, output_image_path)
                if not save_success:
                    logging.error(f"Failed to save weather augmented image '{output_image_path}'. Skipping.")
                    continue
                logging.info(f"Saved weather augmented image '{new_filename}' with ID {image_id_offset}.")

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

        logging.info(f"Weather augmentation completed. Total augmented images: {len(augmented_dataset.images)}.")
        return augmented_dataset

    @staticmethod
    def add_shadow(image):
        top_x, top_y = image.shape[1] * np.random.uniform(), 0
        bot_x, bot_y = image.shape[1] * np.random.uniform(), image.shape[0]
        image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        image_hls = image_hls.astype(np.float64)  # Convert to float64 for calculations
        shadow_mask = np.zeros_like(image_hls[:, :, 1], dtype=np.uint8)
        X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
        shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (X_m - top_y) >= 0)] = 1
        random_brightness = np.random.uniform(0.2, 0.6)  # Random brightness factor between 0.2 and 0.6

        if np.random.randint(2) == 1:
            # Apply the shadow to the masked area
            image_hls[:, :, 1][shadow_mask == 1] *= random_brightness
        else:
            # Apply the shadow to the unmasked area
            image_hls[:, :, 1][shadow_mask == 0] *= random_brightness

        # Ensure the values are within the valid range [0, 255]
        image_hls[:, :, 1] = np.clip(image_hls[:, :, 1], 0, 255)
        image_hls = image_hls.astype(np.uint8)  # Convert back to uint8
        return cv2.cvtColor(image_hls, cv2.COLOR_HLS2BGR)


    @staticmethod
    def add_sunflare(image):
        overlay = image.copy()
        output = image.copy()
        flare_center_x = random.randint(0, image.shape[1])
        flare_center_y = random.randint(0, image.shape[0] // 2)
        radius = random.randint(100, 300)
        cv2.circle(overlay, (flare_center_x, flare_center_y), radius, (255, 255, 255), -1)
        alpha = random.uniform(0.1, 0.3)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        return output

    @staticmethod
    def add_rain(image):
        rain_drops = np.random.randint(1500, 2000)
        image_rain = image.copy()
        for _ in range(rain_drops):
            x = random.randint(0, image.shape[1] - 1)
            y = random.randint(0, image.shape[0] - 1)
            length = random.randint(1, 15)
            cv2.line(image_rain, (x, y), (x, y + length), (200, 200, 200), 1)
        image_blurred = cv2.blur(image_rain, (3, 3))
        return image_blurred
