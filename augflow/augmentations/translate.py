# augflow/augmentations/translate.py

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
    generate_affine_transform_matrix,
    apply_affine_transform,
    mosaic_visualize_transformed_overlays
)
from augflow.utils.annotations import (
    transform_annotations,
    calculate_area_reduction,
    ensure_axis_aligned_rectangle
)
from augflow.utils.unified_format import UnifiedDataset, UnifiedImage, UnifiedAnnotation
import uuid
from typing import Optional, List, Dict
from augflow.utils.configs import translate_default_config

class TranslateAugmentation(Augmentation):
    def __init__(self, config=None, task: str = 'detection'):
        super().__init__()
        self.config = translate_default_config.copy() 
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
        if not self.config.get('enable_translation', True):
            logging.info("Translation augmentation is disabled.")
            return UnifiedDataset()

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

        # Define max_clipped_area_per_category if not provided
        max_clipped_area_per_category = self.config['max_clipped_area_per_category']
        if not max_clipped_area_per_category:
            # Assign a default value if not specified, e.g., 0.2 (20%) for all categories
            max_clipped_area_per_category = {cat['id']: 0.2 for cat in dataset.categories}

        output_images_dir = self.config['output_images_dir']

        for img in dataset.images:
            image_path = img.file_name
            image = load_image(image_path)
            if image is None:
                logging.error(f"Failed to load image '{image_path}'. Skipping.")
                continue
            img_h, img_w = image.shape[:2]

            anns = image_id_to_annotations.get(img.id, [])

            for translate_num in range(self.config['num_translations_per_image']):
                try:
                    # Decide whether to apply translation based on probability
                    prob = self.config['translate_probability']
                    if random.random() > prob:
                        logging.info(f"Skipping translation augmentation {translate_num+1} for image ID {img.id} based on probability ({prob}).")
                        continue  # Skip this augmentation

                    # Random translation parameters (percentages)
                    min_tx_percent = self.config['min_translate_x']
                    max_tx_percent = self.config['max_translate_x']
                    min_ty_percent = self.config['min_translate_y']
                    max_ty_percent = self.config['max_translate_y']

                    # Compute translation amounts in pixels
                    translate_x_percent = random.uniform(min_tx_percent, max_tx_percent)
                    translate_y_percent = random.uniform(min_ty_percent, max_ty_percent)

                    translate_x = translate_x_percent * img_w
                    translate_y = translate_y_percent * img_h

                    # Generate affine transformation matrix
                    M = generate_affine_transform_matrix(
                        image_size=(img_w, img_h),
                        rotation_deg=0,
                        scale=(1.0, 1.0),
                        shear_deg=(0, 0),
                        translation=(translate_x, translate_y)
                    )

                    if output_dim:
                        output_width, output_height = output_dim
                    else:
                        output_width, output_height = img_w, img_h

                    transformed_image = apply_affine_transform(image, M, (output_width, output_height))

                    transformed_anns = transform_annotations(anns, M)

                    # Clean annotations with clipping logic
                    cleaned_anns = []
                    discard_image = False
                    for ann in transformed_anns:
                        # Original coordinates
                        coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
                        if not coords:
                            continue  # Skip if coordinates are invalid

                        original_polygon = Polygon(coords)
                        if not original_polygon.is_valid:
                            original_polygon = original_polygon.buffer(0)
                        original_area = original_polygon.area

                        # Define the image boundary
                        image_boundary = box(0, 0, output_width, output_height)

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
                            is_polygon_clipped=is_polygon_clipped,
                        )

                        cleaned_anns.append(new_ann)
                        annotation_id_offset += 1

                    if discard_image:
                        logging.info(f"Discarding image ID {img.id} due to exceeding area reduction threshold in one or more annotations.")
                        break  # Discard the image and move to the next image

                    if not cleaned_anns:
                        logging.info(f"No valid annotations for image ID {img.id} after translation. Skipping.")
                        continue

                    # Generate new filename
                    new_filename = f"{os.path.splitext(os.path.basename(img.file_name))[0]}_translate_{uuid.uuid4().hex}.jpg"
                    output_image_path = os.path.join(output_images_dir, new_filename)

                    # Save transformed image
                    save_success = save_image(transformed_image, output_image_path)
                    if not save_success:
                        logging.error(f"Failed to save translated image '{output_image_path}'. Skipping.")
                        continue
                    logging.info(f"Saved translated image '{new_filename}' with ID {image_id_offset}.")

                    # Create new image entry
                    new_img = UnifiedImage(
                        id=image_id_offset,
                        file_name=output_image_path,
                        width=output_width,
                        height=output_height
                    )
                    augmented_dataset.images.append(new_img)

                    # Add cleaned annotations to the dataset
                    for new_ann in cleaned_anns:
                        augmented_dataset.annotations.append(new_ann)
                        logging.info(f"Added annotation ID {new_ann.id} for image ID {image_id_offset}.")

                    # Visualization
                    if self.config.get('visualize_overlays', False) and self.config.get('output_visualizations_dir'):
                        visualization_filename = f"{os.path.splitext(new_filename)[0]}_viz.jpg"
                        mosaic_visualize_transformed_overlays(
                            transformed_image=transformed_image.copy(),
                            cleaned_annotations=cleaned_anns,
                            output_visualizations_dir=self.config['output_visualizations_dir'],
                            new_filename=visualization_filename,
                            task=self.task
                        )

                    image_id_offset += 1

                except Exception as e:
                    logging.error(f"Exception during translation augmentation of image ID {img.id}: {e}", exc_info=True)
                    continue

        logging.info(f"Translation augmentation completed. Total augmented images: {len(augmented_dataset.images)}.")
        return augmented_dataset
