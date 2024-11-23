# augflow/augmentations/flip.py

import os
import copy
import random
import logging
import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box
from .base import Augmentation
from augflow.utils.images import (
    load_image,
    save_image,
    mosaic_visualize_transformed_overlays
)
from augflow.utils.annotations import (
    calculate_area_reduction,
    ensure_axis_aligned_rectangle,
    flip_annotations
)
from augflow.utils.unified_format import UnifiedDataset, UnifiedImage, UnifiedAnnotation
import uuid
from typing import Optional, List, Dict
from augflow.utils.configs import flip_default_config

class FlipAugmentation(Augmentation):
    def __init__(self, config=None, task: str = 'detection'):
        super().__init__()
        self.config = flip_default_config.copy()
        if config:
            self.config.update(config)
        self.task = task.lower()

        random.seed(self.config.get('random_seed', 42))
        np.random.seed(self.config.get('random_seed', 42))

        # Ensure output directories exist
        os.makedirs(self.config['output_images_dir'], exist_ok=True)
        if self.config.get('visualize_overlays') and self.config.get('output_visualizations_dir'):
            os.makedirs(self.config['output_visualizations_dir'], exist_ok=True)

        # Set max_clipped_area_per_category to default if not provided
        if not self.config.get('max_clipped_area_per_category'):
            # Will be set in apply() based on dataset categories
            self.config['max_clipped_area_per_category'] = {}

    def apply(self, dataset: UnifiedDataset, output_dim: Optional[tuple] = None) -> UnifiedDataset:
        if not self.config.get('enable_flipping', True):
            logging.info("Flipping augmentation is disabled.")
            return UnifiedDataset()

        augmented_dataset = UnifiedDataset(
            images=[],
            annotations=[],
            categories=copy.deepcopy(dataset.categories)
        )

        # Initialize IDs
        existing_image_ids = [img.id for img in dataset.images]
        existing_annotation_ids = [ann.id for ann in dataset.annotations]
        image_id_offset = max(existing_image_ids) + 1 if existing_image_ids else 1
        annotation_id_offset = max(existing_annotation_ids) + 1 if existing_annotation_ids else 1

        # Mapping from image_id to annotations
        image_id_to_annotations = {}
        for ann in dataset.annotations:
            image_id_to_annotations.setdefault(ann.image_id, []).append(ann)

        flip_modes = self.config.get('flip_modes', ['horizontal'])
        num_flips_per_image = self.config.get('num_flips_per_image', 1)
        flip_probability = self.config.get('flip_probability', 1.0)
        max_clipped_area_per_category = self.config.get('max_clipped_area_per_category', {})

        output_images_dir = self.config['output_images_dir']

        for img in dataset.images:
            image_path = img.file_name
            image = load_image(image_path)
            if image is None:
                logging.error(f"Failed to load image '{image_path}'. Skipping.")
                continue
            img_h, img_w = image.shape[:2]

            anns = image_id_to_annotations.get(img.id, [])

            for flip_num in range(num_flips_per_image):
                try:
                    # Decide whether to apply flip based on probability
                    prob = flip_probability
                    if random.random() > prob:
                        logging.info(f"Skipping flip augmentation {flip_num+1} for image ID {img.id} based on probability ({prob}).")
                        continue  # Skip this augmentation

                    # Select flip mode
                    flip_mode = random.choice(flip_modes)
                    flip_code = self.get_flip_code(flip_mode)

                    # Flip the image
                    flipped_image = cv2.flip(image, flip_code)

                    # Flip annotations
                    flipped_anns = flip_annotations(anns, img_w, img_h, flip_code)

                    # Clean annotations with clipping logic
                    cleaned_anns = []
                    discard_image = False
                    for ann in flipped_anns:
                        # Original coordinates
                        coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
                        if not coords:
                            continue  # Skip if coordinates are invalid

                        original_polygon = Polygon(coords)
                        if not original_polygon.is_valid:
                            original_polygon = original_polygon.buffer(0)
                        original_area = original_polygon.area

                        # Define the image boundary
                        image_boundary = box(0, 0, img_w, img_h)

                        # Clip the polygon to the image boundary
                        clipped_polygon = original_polygon.intersection(image_boundary)

                        if clipped_polygon.is_empty:
                            continue  # Polygon is completely outside; exclude it

                        if not clipped_polygon.is_valid:
                            clipped_polygon = clipped_polygon.buffer(0)
                        clipped_area = clipped_polygon.area

                        # Compute area reduction due to clipping
                        area_reduction_due_to_clipping = calculate_area_reduction(original_area, clipped_area)

                        # Determine if polygon was clipped
                        is_polygon_clipped = area_reduction_due_to_clipping > 0.01

                        # Check if area reduction exceeds the threshold
                        category_id = ann.category_id
                        max_allowed_reduction = max_clipped_area_per_category.get(category_id, 0.2)  # Default to 20%

                        if area_reduction_due_to_clipping > max_allowed_reduction:
                            logging.info(f"Annotation ID {ann.id} discarded due to area reduction {area_reduction_due_to_clipping:.2f} exceeding threshold {max_allowed_reduction}.")
                            discard_image = True
                            break  # Discard this annotation

                        # Handle MultiPolygon cases
                        polygons_to_process = []
                        if isinstance(clipped_polygon, Polygon):
                            polygons_to_process.append(clipped_polygon)
                        elif isinstance(clipped_polygon, MultiPolygon):
                            polygons_to_process.extend(clipped_polygon.geoms)
                        else:
                            logging.warning(f"Unknown geometry type for clipped polygon: {type(clipped_polygon)}")
                            continue

                        # Collect cleaned polygon coordinates
                        cleaned_polygon_coords = []
                        for poly in polygons_to_process:
                            if self.task == 'detection':
                                coords = ensure_axis_aligned_rectangle(list(poly.exterior.coords))
                                if coords:
                                    cleaned_polygon_coords.extend(coords)
                            else:
                                coords = list(poly.exterior.coords)
                                if coords:
                                    cleaned_polygon_coords.extend(coords)

                        if not cleaned_polygon_coords:
                            logging.debug(f"No valid coordinates found after processing clipped polygons. Skipping annotation.")
                            continue

                        # Update the annotation
                        new_ann = UnifiedAnnotation(
                            id=annotation_id_offset,
                            image_id=image_id_offset,
                            category_id=ann.category_id,
                            polygon=[coord for point in cleaned_polygon_coords for coord in point],
                            iscrowd=ann.iscrowd,
                            area=clipped_area,
                            area_reduction_due_to_clipping=area_reduction_due_to_clipping
                        )

                        cleaned_anns.append(new_ann)
                        annotation_id_offset += 1

                    if discard_image:
                        logging.info(f"Discarding image ID {img.id} due to exceeding area reduction threshold in one or more annotations.")
                        break  # Discard the image and move to the next image

                    if not cleaned_anns:
                        logging.info(f"No valid annotations for image ID {img.id} after flipping. Skipping.")
                        continue

                    # Generate new filename
                    new_filename = f"{os.path.splitext(os.path.basename(img.file_name))[0]}_flip_{uuid.uuid4().hex}.jpg"
                    output_image_path = os.path.join(output_images_dir, new_filename)

                    # Save flipped image
                    save_success = save_image(flipped_image, output_image_path)
                    if not save_success:
                        logging.error(f"Failed to save flipped image '{output_image_path}'. Skipping.")
                        continue
                    logging.info(f"Saved flipped image '{new_filename}' with ID {image_id_offset}.")

                    # Create new image entry
                    new_img = UnifiedImage(
                        id=image_id_offset,
                        file_name=output_image_path,
                        width=img_w,
                        height=img_h
                    )
                    augmented_dataset.images.append(new_img)

                    # Process and save flipped annotations
                    for new_ann in cleaned_anns:
                        augmented_dataset.annotations.append(new_ann)
                        logging.info(f"Added annotation ID {new_ann.id} for image ID {image_id_offset}.")

                    # Visualization
                    if self.config.get('visualize_overlays', False) and self.config.get('output_visualizations_dir'):
                        visualization_filename = f"{os.path.splitext(new_filename)[0]}_viz.jpg"
                        mosaic_visualize_transformed_overlays(
                            transformed_image=flipped_image.copy(),
                            cleaned_annotations=cleaned_anns,
                            output_visualizations_dir=self.config['output_visualizations_dir'],
                            new_filename=visualization_filename,
                            task=self.task
                        )

                    image_id_offset += 1

                except Exception as e:
                    logging.error(f"Exception during flip augmentation of image ID {img.id}: {e}", exc_info=True)
                    continue

        logging.info(f"Flipping augmentation completed. Total flipped images: {image_id_offset - max(existing_image_ids, default=0) -1}.")
        return augmented_dataset

    def get_flip_code(self, flip_mode: str) -> int:
        """
        Map flip mode string to OpenCV flip code.

        Parameters:
            flip_mode (str): 'horizontal', 'vertical', or 'both'.

        Returns:
            int: Corresponding OpenCV flip code.
        """
        flip_mode = flip_mode.lower()
        if flip_mode == 'horizontal':
            return 1
        elif flip_mode == 'vertical':
            return 0
        elif flip_mode == 'both':
            return -1
        else:
            logging.warning(f"Unsupported flip mode '{flip_mode}'. Defaulting to horizontal.")
            return 1