import os
import copy
import random
import logging
import cv2
import uuid
from typing import Optional, List, Tuple, Dict
import numpy as np

from .base import Augmentation
from augflow.utils.images import load_image, save_image, mosaic_visualize_transformed_overlays, scale_image
from augflow.utils.annotations import (
    reshape_segmentation,
    transform_annotations,
    calculate_area_reduction,
    compute_polygon_area,
    get_new_bbox,
    ensure_axis_aligned_rectangle
)
from augflow.utils.unified_format import UnifiedDataset, UnifiedImage, UnifiedAnnotation

from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection
from augflow.utils.configs import scale_default_config

class ScaleAugmentation(Augmentation):
    def __init__(self, config=None, task: str = 'detection'):
        super().__init__()
        self.task = task.lower()
        self.config = scale_default_config.copy() 
        if config:
            self.config.update(config)
       
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
        if not self.config.get('enable_scaling', True):
            logging.info("Scaling augmentation is disabled.")
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

        # Scaling Configuration
        scale_mode = self.config.get('scale_mode', 'uniform')
        scale_factors = self.config.get('scale_factors', [1.0])
        scale_factor_range = self.config.get('scale_factor_range', (0.8, 1.2))
        scale_step = self.config.get('scale_step', 0.1)
        interpolation_methods = self.config.get('interpolation_methods', ['linear'])
        preserve_aspect_ratio = self.config.get('preserve_aspect_ratio', True)
        num_scales_per_image = self.config.get('num_scales_per_image', 1)
        max_clipped_area_per_category = self.config['max_clipped_area_per_category']
        if not max_clipped_area_per_category:
            # Assign default max clipped area reduction per category (e.g., 0.2 or 20%)
            max_clipped_area_per_category = {cat['id']: 0.2 for cat in dataset.categories}

        output_images_dir = self.config['output_images_dir']
        output_visualizations_dir = self.config.get('output_visualizations_dir')

        for img in dataset.images:
            image_path = img.file_name
            image = load_image(image_path)
            if image is None:
                logging.error(f"Failed to load image '{image_path}'. Skipping.")
                continue
            img_h, img_w = image.shape[:2]

            anns = image_id_to_annotations.get(img.id, [])
            if not anns:
                logging.info(f"No annotations found for image ID {img.id}. Skipping scaling.")
                continue

            for _ in range(num_scales_per_image):
                try:
                    # Select scale factors based on mode
                    scale_x, scale_y, interpolation, non_uniform = self.select_scale_factor(
                        scale_mode, scale_factors, scale_factor_range, scale_step)

                    # Validate scale factors
                    if scale_x <= 0 or scale_y <= 0:
                        logging.warning(f"Invalid scale factors: scale_x={scale_x}, scale_y={scale_y}. Skipping this scale.")
                        continue

                    # Scale the image
                    scaled_image = scale_image(image, scale_x, scale_y, interpolation)
                    scaled_h, scaled_w = scaled_image.shape[:2]

                    # Generate scaling matrix
                    M = np.array([
                        [scale_x, 0, 0],
                        [0, scale_y, 0]
                    ], dtype=np.float32)

                    # Scale annotations
                    scaled_anns = transform_annotations(anns, M)

                    # Clean annotations with clipping logic
                    cleaned_anns = []
                    discard_image=False
                    for ann in scaled_anns:
                        # Original coordinates
                        coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
                        if not coords:
                            continue  # Skip if coordinates are invalid

                        original_polygon = Polygon(coords)
                        if not original_polygon.is_valid:
                            original_polygon = original_polygon.buffer(0)
                        original_area = original_polygon.area

                        # Define the image boundary
                        image_boundary = box(0, 0, scaled_w, scaled_h)

                        # Clip the polygon to the image boundary
                        clipped_polygon = original_polygon.intersection(image_boundary)

                        if clipped_polygon.is_empty:
                            continue  # Polygon is completely outside; exclude it

                        if not clipped_polygon.is_valid:
                            clipped_polygon = clipped_polygon.buffer(0)
                        clipped_area = clipped_polygon.area

                        # Compute area reduction due to clipping and scaling
                        area_reduction = calculate_area_reduction(original_area, clipped_area)

                        # Check if area reduction exceeds the threshold
                        category_id = ann.category_id
                        max_allowed_reduction = max_clipped_area_per_category.get(category_id, 0.2)  # Default to 20%

                        if area_reduction > max_allowed_reduction:
                            logging.info(f"Annotation ID {ann.id} discarded due to area reduction {area_reduction:.2f} exceeding threshold {max_allowed_reduction}.")
                            discard_image=True
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
                            area=clipped_area
                        )

                        cleaned_anns.append(new_ann)
                        annotation_id_offset += 1

                    if discard_image:
                        logging.info(f"Discarding image ID {img.id} due to exceeding area reduction threshold in one or more annotations.")
                        break  # Discard the image and move to the next image

                    if not cleaned_anns:
                        logging.info(f"No valid annotations for image ID {img.id} after scaling. Skipping.")
                        continue

                    # Generate unique filename
                    scale_desc = f"{scale_x:.2f}" if scale_x == scale_y else f"{scale_x:.2f}_{scale_y:.2f}"
                    new_filename = f"{os.path.splitext(os.path.basename(img.file_name))[0]}_scale_{scale_desc}_{uuid.uuid4().hex}{os.path.splitext(img.file_name)[1]}"
                    output_image_path = os.path.join(output_images_dir, new_filename)

                    # Save scaled image
                    save_success = save_image(scaled_image, output_image_path)
                    if not save_success:
                        logging.error(f"Failed to save scaled image '{output_image_path}'. Skipping.")
                        continue
                    logging.info(f"Saved scaled image '{new_filename}' with ID {image_id_offset}.")

                    # Create new image entry
                    new_img = UnifiedImage(
                        id=image_id_offset,
                        file_name=output_image_path,
                        width=scaled_w,
                        height=scaled_h
                    )
                    augmented_dataset.images.append(new_img)

                    # Process and save scaled annotations
                    for new_ann in cleaned_anns:
                        augmented_dataset.annotations.append(new_ann)
                        logging.info(f"Added annotation ID {new_ann.id} for image ID {image_id_offset}.")

                    # Visualization
                    if self.config.get('visualize_overlays', False) and output_visualizations_dir:
                        visualization_filename = f"{os.path.splitext(new_filename)[0]}_viz.jpg"
                        mosaic_visualize_transformed_overlays(
                            transformed_image=scaled_image.copy(),
                            cleaned_annotations=cleaned_anns,
                            output_visualizations_dir=output_visualizations_dir,
                            new_filename=visualization_filename,
                            task=self.task
                        )

                    image_id_offset += 1

                except Exception as e:
                    logging.error(f"Exception during scaling augmentation of image ID {img.id}: {e}", exc_info=True)
                    continue

        logging.info(f"Scaling augmentation completed. Total scaled images: {len(augmented_dataset.images)}.")
        return augmented_dataset

    def select_scale_factor(self, scale_mode, scale_factors, scale_factor_range, scale_step) -> Tuple[float, float, int, bool]:
        """
        Select scale factors based on the configuration.

        Args:
            scale_mode (str): Scaling mode ('uniform', 'non_uniform', 'range_random', 'range_step', 'list').
            scale_factors (List[float]): List of scale factors for 'list' or 'uniform' modes.
            scale_factor_range (Tuple[float, float]): Range for random scaling in 'range_random' mode.
            scale_step (float): Step size for 'range_step' mode.

        Returns:
            Tuple[float, float, int, bool]: scale_x, scale_y, interpolation method, non_uniform flag.
        """
        non_uniform = False
        scale_x = scale_y = 1.0  # Default scale factors
        interpolation = cv2.INTER_LINEAR  # Default interpolation

        if scale_mode == 'uniform':
            scale = random.choice(scale_factors)
            scale_x = scale_y = scale
            logging.debug(f"Uniform scaling selected with factor: {scale}")
        elif scale_mode == 'non_uniform':
            # For non_uniform, use separate scale factors for x and y
            scale_x = random.uniform(*scale_factor_range)
            scale_y = random.uniform(*scale_factor_range)
            non_uniform = True
            logging.debug(f"Non-uniform scaling selected with factors: scale_x={scale_x}, scale_y={scale_y}")
        elif scale_mode == 'range_random':
            scale = random.uniform(*scale_factor_range)
            scale_x = scale_y = scale
            logging.debug(f"Range random scaling selected with factor: {scale}")
        elif scale_mode == 'range_step':
            # Generate a list of possible scale factors based on step
            start, end = scale_factor_range
            possible_scales = np.arange(start, end + scale_step, scale_step).tolist()
            scale = random.choice(possible_scales)
            scale_x = scale_y = scale
            logging.debug(f"Range step scaling selected with factor: {scale}")
        elif scale_mode == 'list':
            scale = random.choice(scale_factors)
            scale_x = scale_y = scale
            logging.debug(f"List scaling selected with factor: {scale}")
        else:
            logging.warning(f"Unsupported scale_mode '{scale_mode}'. Defaulting to uniform scaling with factor 1.0.")
            scale_x = scale_y = 1.0

        # Choose interpolation method
        interpolation_str = random.choice(self.config.get('interpolation_methods', ['linear']))
        interpolation_map = {
            'nearest': cv2.INTER_NEAREST,
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
            'area': cv2.INTER_AREA,
            'lanczos4': cv2.INTER_LANCZOS4
        }
        interpolation = interpolation_map.get(interpolation_str.lower(), cv2.INTER_LINEAR)
        logging.debug(f"Selected interpolation method: {interpolation_str} ({interpolation})")

        return scale_x, scale_y, interpolation, non_uniform

    
    
