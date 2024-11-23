# augflow/augmentations/mosaic.py

import os
import copy
import random
import logging
import cv2
import numpy as np
from shapely.geometry import Polygon, box, MultiPolygon
from shapely.ops import unary_union
from .base import Augmentation
from augflow.utils.images import load_image, save_image, mosaic_visualize_transformed_overlays
from augflow.utils.unified_format import UnifiedDataset, UnifiedImage, UnifiedAnnotation
from augflow.utils.annotations import ensure_axis_aligned_rectangle, passes_filter_scale
import uuid
from typing import Optional, List, Dict
from augflow.utils.configs import mosaic_default_config

class MosaicAugmentation(Augmentation):
    def __init__(self, config=None, task: str = 'detection'):
        super().__init__()
        self.config = mosaic_default_config.copy() 
        if config:
            self.config.update(config)
        self.task = task.lower()
        random.seed(self.config.get('random_seed', 42))
        np.random.seed(self.config.get('random_seed', 42))
        
        # Ensure output directories exist
        os.makedirs(self.config['output_images_dir'], exist_ok=True)
        if self.config.get('visualize_overlays') and self.config.get('output_visualizations_dir'):
            os.makedirs(self.config['output_visualizations_dir'], exist_ok=True)
        
        # Set max_allowed_area_reduction_per_category to default if not provided
        if not self.config.get('max_allowed_area_reduction_per_category'):
            # This will be handled in the apply method if not set
            self.config['max_allowed_area_reduction_per_category'] = {}

        

    def apply(self, dataset: UnifiedDataset, output_dim: Optional[tuple] = None) -> UnifiedDataset:
        if not self.config.get('enable_mosaic', True):
            logging.info("Mosaic augmentation is disabled.")
            return dataset  # Return the original dataset

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

        # Define max_allowed_area_reduction_per_category if not provided
        max_allowed_area_reduction_per_category = self.config['max_allowed_area_reduction_per_category']
        if not max_allowed_area_reduction_per_category:
            # Assign a default value if not specified, e.g., 0.25 (25%) for all categories
            max_allowed_area_reduction_per_category = {cat['id']: 0.25 for cat in dataset.categories}

        # Extract valid category IDs from the dataset.categories list
        valid_category_ids = set(cat['id'] for cat in dataset.categories)

        # Initialize usage count for each image
        usage_count: Dict[int, int] = {img.id: 0 for img in dataset.images}
        max_usage = self.config.get('max_usage_per_image', 2)

        output_images_dir = self.config['output_images_dir']
        output_visualizations_dir = self.config.get('output_visualizations_dir')

        # Shuffle images to ensure randomness
        shuffled_images = copy.deepcopy(dataset.images)
        random.shuffle(shuffled_images)

        # Filter images based on max_usage_per_image
        available_images = [img for img in shuffled_images if usage_count[img.id] < max_usage]

        # Check if enough images are available to create the desired number of mosaics
        images_per_mosaic = self.config['grid_size'][0] * self.config['grid_size'][1]
        max_possible_mosaics = len(available_images) // images_per_mosaic
        if self.config['num_mosaics'] > max_possible_mosaics:
            logging.warning(
                f"Not enough unique images to create {self.config['num_mosaics']} mosaics. "
                f"Available images can only create up to {max_possible_mosaics} mosaics."
            )
            if max_possible_mosaics == 0:
                logging.error("Insufficient unique images to create any mosaics.")
                return augmented_dataset
            self.config['num_mosaics'] = max_possible_mosaics

        # Get the max offset values
        max_offset_x_fraction = self.config.get('max_offset_x', 0.2)
        max_offset_y_fraction = self.config.get('max_offset_y', 0.2)

        for mosaic_num in range(self.config['num_mosaics']):
            logging.info(f"Creating mosaic {mosaic_num + 1}/{self.config['num_mosaics']}.")
            grid_rows, grid_cols = self.config['grid_size']

            # Determine output_size if not provided
            if output_dim is not None:
                output_width, output_height = output_dim
            elif self.config['output_size'] is not None:
                output_width, output_height = self.config['output_size']
            else:
                # Calculate output size based on the size of the images used
                sample_images = random.sample(available_images, min(len(available_images), images_per_mosaic))
                avg_width = int(np.mean([img.width for img in sample_images]))
                avg_height = int(np.mean([img.height for img in sample_images]))
                output_width = avg_width * grid_cols
                output_height = avg_height * grid_rows
                logging.info(
                    f"Calculated output_size as ({output_width}, {output_height}) based on average image size."
                )

            # Initialize empty canvas and annotations for the current mosaic
            mosaic_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)

            cell_width = output_width // grid_cols
            cell_height = output_height // grid_rows

            logging.debug(
                f"Mosaic dimensions: output_width={output_width}, output_height={output_height}, "
                f"cell_width={cell_width}, cell_height={cell_height}"
            )

            # Define grid positions and boundaries
            grid_positions = []
            cell_boundaries = []
            for i in range(grid_rows):
                for j in range(grid_cols):
                    y = i * cell_height
                    x = j * cell_width
                    grid_positions.append((y, x))
                    # Define the boundary of the current cell
                    cell_boundary = box(x, y, x + cell_width, y + cell_height)
                    cell_boundaries.append(cell_boundary)

            # Shuffle grid positions if randomize_positions is True
            if self.config.get('randomize_positions', True):
                combined = list(zip(grid_positions, cell_boundaries))
                random.shuffle(combined)
                grid_positions, cell_boundaries = zip(*combined)

            # Initialize list for new annotations
            mosaic_annotations = []

            # Flag to determine if mosaic should be discarded
            discard_mosaic = False

            for cell_idx, ((y, x), cell_boundary) in enumerate(zip(grid_positions, cell_boundaries)):
                attempt = 0
                success = False
                while attempt < self.config['max_attempts_per_cell'] and not success:
                    attempt += 1
                    # Select a random image from available_images
                    if not available_images:
                        logging.warning("No available images left to select.")
                        discard_mosaic = True
                        break
                    img = random.choice(available_images)

                    logging.debug(f"Selected image ID {img.id} for cell {cell_idx + 1}")

                    # Determine scaling factors based on resizing to fit the mosaic cell
                    scale_x = cell_width / img.width
                    scale_y = cell_height / img.height
                    logging.debug(
                        f"Image ID {img.id}: img.width={img.width}, img.height={img.height}, "
                        f"cell_width={cell_width}, cell_height={cell_height}, scale_x={scale_x}, scale_y={scale_y}"
                    )

                    # Load and resize image
                    loaded_image = load_image(img.file_name)
                    if loaded_image is None:
                        logging.error(f"Failed to load image '{img.file_name}'. Skipping.")
                        continue
                    resized_image = cv2.resize(loaded_image, (cell_width, cell_height))
                    logging.debug(
                        f"Original image size: {loaded_image.shape[1]}x{loaded_image.shape[0]}, "
                        f"Resized image size: {resized_image.shape[1]}x{resized_image.shape[0]}"
                    )

                    # Apply random offsets
                    max_offset_x = max_offset_x_fraction * cell_width
                    max_offset_y = max_offset_y_fraction * cell_height
                    offset_x = random.uniform(-max_offset_x, max_offset_x)
                    offset_y = random.uniform(-max_offset_y, max_offset_y)
                    logging.debug(
                        f"Offsets for cell {cell_idx + 1}: offset_x={offset_x}, offset_y={offset_y}"
                    )

                    # Adjusted positions
                    adjusted_x = x + offset_x
                    adjusted_y = y + offset_y

                    # The clipping boundary remains the original cell boundary
                    clipping_boundary = cell_boundary  # Do not adjust the boundary with offsets

                    # Process annotations for this image
                    anns = image_id_to_annotations.get(img.id, [])
                    valid_annotations = []
                    exceeds_threshold = False

                    for ann in anns:
                        # Validate category_id
                        if ann.category_id not in valid_category_ids:
                            logging.error(
                                f"Annotation ID {ann.id} has invalid category_id {ann.category_id}. Skipping this annotation."
                            )
                            continue

                        # Check if annotation passes filter_scale
                        if not passes_filter_scale(ann, self.config.get('filter_scale', 10)):
                            continue

                        # Original coordinates
                        coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
                        if not coords:
                            continue  # Skip if coordinates are invalid

                        original_polygon = Polygon(coords)
                        if not original_polygon.is_valid:
                            original_polygon = original_polygon.buffer(0)

                        original_area = original_polygon.area

                        # Apply scaling to the original coordinates
                        scaled_coords = [(px * scale_x, py * scale_y) for px, py in coords]
                        scaled_polygon = Polygon(scaled_coords)
                        if not scaled_polygon.is_valid:
                            scaled_polygon = scaled_polygon.buffer(0)
                        scaled_area = scaled_polygon.area

                        # Calculate area reduction due to scaling

                        if self.config['allow_significant_area_reduction_due_to_scaling']:
                            area_reduction_due_to_scaling = 0.0

                        else:
                            
                            if original_area > 0:
                                area_reduction_due_to_scaling = max(0.0, (original_area - scaled_area) / original_area)
                            else:
                                area_reduction_due_to_scaling = 0.0

                        # Apply translation to place the polygon in the mosaic cell with offset
                        adjusted_coords = [(px + adjusted_x, py + adjusted_y) for px, py in scaled_coords]
                        adjusted_polygon = Polygon(adjusted_coords)
                        if not adjusted_polygon.is_valid:
                            adjusted_polygon = adjusted_polygon.buffer(0)

                        # Clipping boundary is the cell boundary
                        clipping_boundary = cell_boundary

                        # Clipping the adjusted polygon to the cell boundary
                        clipped_polygon = adjusted_polygon.intersection(clipping_boundary)

                        if clipped_polygon.is_empty:
                            continue  # Polygon is completely outside the cell; exclude it

                        if not clipped_polygon.is_valid:
                            clipped_polygon = clipped_polygon.buffer(0)

                        clipped_area = clipped_polygon.area

                        # Calculate area reduction due to clipping
                        if scaled_area > 0:
                            area_reduction_due_to_clipping = max(0.0, (scaled_area - clipped_area) / scaled_area)
                        else:
                            area_reduction_due_to_clipping = 0.0

                        if self.config['allow_significant_area_reduction_due_to_scaling']:
                            total_reduction=area_reduction_due_to_clipping
                        else:
                            if original_area > 0:
                                total_reduction = max(0.0, (original_area - clipped_area) / original_area)
                            else:
                                total_reduction = 0.0

                        # Determine if polygon was clipped
                        is_polygon_clipped = area_reduction_due_to_clipping > 0.01

                        # Determine if polygon was scaled
                        is_polygon_scaled = area_reduction_due_to_scaling > 0.01

                        # Logging for debugging purposes
                        logging.info(
                            f"Annotation ID {ann.id}: Original Area = {original_area}, Scaled Area = {scaled_area}, "
                            f"Clipped Area = {clipped_area}, area_reduction_due_to_scaling = {area_reduction_due_to_scaling}, "
                            f"area_reduction_due_to_clipping = {area_reduction_due_to_clipping}, "
                            f"total_reduction = {total_reduction}, "
                            f"is_polygon_scaled = {is_polygon_scaled}, is_polygon_clipped = {is_polygon_clipped}"
                        )

                        category_id = ann.category_id
                        max_allowed_reduction = max_allowed_area_reduction_per_category.get(category_id, 1.0)

                        # Check if total area reduction exceeds the threshold
                        if total_reduction > max_allowed_reduction:
                            logging.warning(
                                f"Mosaic {mosaic_num + 1}: Annotation ID {ann.id} in image ID {img.id} has total area reduction "
                                f"({total_reduction:.6f}) exceeding threshold ({max_allowed_reduction}) for category {category_id}."
                            )
                            exceeds_threshold = True
                            break  # Exit the annotations loop for this image

                        # Handle MultiPolygon and Polygon
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

                        # Assign area reductions and flags
                        cleaned_ann = UnifiedAnnotation(
                            id=ann.id,
                            image_id=ann.image_id,
                            category_id=ann.category_id,
                            polygon=[coord for point in cleaned_polygon_coords for coord in point],
                            iscrowd=ann.iscrowd,
                            area=clipped_area,
                            is_polygon_scaled=is_polygon_scaled,
                            is_polygon_clipped=is_polygon_clipped,
                            area_reduction_due_to_scaling=area_reduction_due_to_scaling,
                            area_reduction_due_to_clipping=area_reduction_due_to_clipping
                        )

                        valid_annotations.append(cleaned_ann)

                    if not exceeds_threshold and valid_annotations:
                        # Image passes all annotations area criteria

                        # Place the image in the mosaic canvas with potential offset
                        x1 = int(adjusted_x)
                        y1 = int(adjusted_y)
                        x2 = x1 + resized_image.shape[1]
                        y2 = y1 + resized_image.shape[0]

                        # Define the cell boundaries in terms of pixel coordinates
                        cell_x1 = x
                        cell_y1 = y
                        cell_x2 = x + cell_width
                        cell_y2 = y + cell_height

                        # Determine the valid region in the mosaic canvas, clipped to cell boundaries
                        canvas_x1 = max(cell_x1, x1)
                        canvas_y1 = max(cell_y1, y1)
                        canvas_x2 = min(cell_x2, x2)
                        canvas_y2 = min(cell_y2, y2)

                        # Determine the corresponding region in the resized image
                        image_x1 = canvas_x1 - x1
                        image_y1 = canvas_y1 - y1
                        image_x2 = canvas_x2 - x1
                        image_y2 = canvas_y2 - y1

                        # If the valid region is non-empty, place the image
                        if canvas_x2 > canvas_x1 and canvas_y2 > canvas_y1:
                            mosaic_image[canvas_y1:canvas_y2, canvas_x1:canvas_x2] = \
                                resized_image[image_y1:image_y2, image_x1:image_x2]

                        # Add annotations
                        mosaic_annotations.extend(valid_annotations)

                        # Update usage count and remove from available_images if max usage reached
                        usage_count[img.id] += 1
                        if usage_count[img.id] >= max_usage:
                            available_images.remove(img)

                        success = True
                        logging.debug(
                            f"Placed image ID {img.id} in cell {cell_idx + 1} after {attempt} attempts."
                        )
                    else:
                        # Annotation exceeded area reduction threshold or no valid annotations
                        logging.debug(
                            f"Attempt {attempt}: Image ID {img.id} does not meet area criteria for cell {cell_idx + 1}. Retrying."
                        )
                        continue  # Try another image

                if not success:
                    # Failed to find a suitable image for this cell after max attempts
                    logging.warning(
                        f"Failed to find a suitable image for cell {cell_idx + 1} in mosaic {mosaic_num + 1} after "
                        f"{self.config['max_attempts_per_cell']} attempts. Discarding mosaic."
                    )
                    discard_mosaic = True
                    break  # Discard the entire mosaic and proceed to the next one

            if discard_mosaic:
                continue  # Skip saving this mosaic and move to the next one

            if not mosaic_annotations:
                logging.warning(f"Mosaic {mosaic_num + 1} has no valid annotations. Skipping.")
                continue

            # Create unique filename using UUID
            new_filename = f"mosaic_{mosaic_num + 1}_{uuid.uuid4().hex}.jpg"
            output_image_path = os.path.join(output_images_dir, new_filename)

            # Save mosaic image
            save_success = save_image(mosaic_image, output_image_path)
            if not save_success:
                logging.error(f"Failed to save mosaic image '{output_image_path}'. Skipping this mosaic.")
                continue

            # Create new image entry
            new_img = UnifiedImage(
                id=image_id_offset,
                file_name=output_image_path,
                width=mosaic_image.shape[1],
                height=mosaic_image.shape[0]
            )
            augmented_dataset.images.append(new_img)

            # Process and save annotations
            for ann in mosaic_annotations:
                new_ann = UnifiedAnnotation(
                    id=annotation_id_offset,
                    image_id=image_id_offset,
                    category_id=ann.category_id,
                    polygon=ann.polygon,
                    iscrowd=ann.iscrowd,
                    area=ann.area,
                    is_polygon_scaled=ann.is_polygon_scaled,
                    is_polygon_clipped=ann.is_polygon_clipped,
                    area_reduction_due_to_scaling=ann.area_reduction_due_to_scaling,
                    area_reduction_due_to_clipping=ann.area_reduction_due_to_clipping
                )
                augmented_dataset.annotations.append(new_ann)
                logging.info(f"Added annotation ID {new_ann.id} for mosaic image ID {image_id_offset}.")
                annotation_id_offset += 1

            # Visualization
            if self.config['visualize_overlays'] and output_visualizations_dir:
                os.makedirs(output_visualizations_dir, exist_ok=True)
                visualization_filename = f"{os.path.splitext(new_filename)[0]}_viz.jpg"
                mosaic_visualize_transformed_overlays(
                    transformed_image=mosaic_image.copy(),
                    cleaned_annotations=mosaic_annotations,
                    output_visualizations_dir=output_visualizations_dir,
                    new_filename=visualization_filename,
                    task=self.task  # Pass the task ('detection' or 'segmentation')
                )

            logging.info(
                f"Mosaic {mosaic_num + 1} created and saved as '{new_filename}' with {len(mosaic_annotations)} annotations."
            )
            image_id_offset += 1

        # Return the augmented dataset after all mosaics are created
        logging.info(
            f"Mosaic augmentation completed. Total mosaics created: {len(augmented_dataset.images)}."
        )
        return augmented_dataset

    
  
