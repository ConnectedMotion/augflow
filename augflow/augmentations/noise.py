
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

class NoiseAugmentation(Augmentation):
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
            'noise_probability': 0.5,
            'noise_types': ['gaussian', 's&p'],
            'num_augmented_images': 1,
            'enable_noise': True,
            'output_images_dir': 'augmented_images_noise',
        }
        for key, value in default_config.items():
            self.config.setdefault(key, value)

    def apply(self, dataset: UnifiedDataset, output_dim: Optional[tuple] = None) -> UnifiedDataset:
        if not self.config.get('enable_noise', True):
            logging.info("Noise augmentation is disabled.")
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

                if random.random() < self.config['noise_probability']:
                    noise_type = random.choice(self.config['noise_types'])
                    transformed_image = self.add_noise(transformed_image, noise_type)
                    logging.debug(f"Applied {noise_type} noise.")

                # Generate new filename
                new_filename = f"{os.path.splitext(os.path.basename(img.file_name))[0]}_noise_{uuid.uuid4().hex}.jpg"
                output_image_path = os.path.join(output_images_dir, new_filename)

                # Save transformed image
                save_success = save_image(transformed_image, output_image_path)
                if not save_success:
                    logging.error(f"Failed to save noise augmented image '{output_image_path}'. Skipping.")
                    continue
                logging.info(f"Saved noise augmented image '{new_filename}' with ID {image_id_offset}.")

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

        logging.info(f"Noise augmentation completed. Total augmented images: {len(augmented_dataset.images)}.")
        return augmented_dataset

    @staticmethod
    def add_noise(image, noise_type):
        if noise_type == 'gaussian':
            mean = 0
            var = 10
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, image.shape).astype('uint8')
            noisy_image = cv2.add(image, gauss)
            return noisy_image
        elif noise_type == 's&p':
            amount = 0.004
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * 0.5)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
            out[coords[0], coords[1], :] = 255

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * 0.5)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
            out[coords[0], coords[1], :] = 0
            return out
        else:
            return image
