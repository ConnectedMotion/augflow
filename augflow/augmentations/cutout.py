# cutout.py
import os
import copy
import random
import logging
import cv2
import numpy as np
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import unary_union
from .base import Augmentation  # Adjust the import based on your project structure
from augflow.utils.images import load_image, save_image, mosaic_visualize_transformed_overlays
from augflow.utils.annotations import (
    calculate_area_reduction,
    ensure_axis_aligned_rectangle,
    passes_filter_scale
)
from augflow.utils.unified_format import UnifiedDataset, UnifiedImage, UnifiedAnnotation
import uuid
from typing import Optional, List, Dict
from augflow.utils.configs import cutout_default_config

class CutoutAugmentation(Augmentation):
    def __init__(self, config=None, task: str = 'detection', modes: List[str] = None, focus_categories: Optional[List[str]] = None):
        super().__init__()
        self.task = task.lower()
        self.config = cutout_default_config.copy()
        if config:
            self.config.update(config)
        self.modes = [mode.lower() for mode in (modes or self.config.get('modes', []))]
        self.focus_categories = focus_categories or self.config.get('focus_categories', [])
        random.seed(self.config.get('random_seed', 42))
        np.random.seed(self.config.get('random_seed', 42))

        # Ensure output directories exist
        os.makedirs(self.config['output_images_dir'], exist_ok=True)
        if self.config.get('visualize_overlays') and self.config.get('output_visualizations_dir'):
            os.makedirs(self.config['output_visualizations_dir'], exist_ok=True)

        # Initialize max_clipped_area_per_category
        self.max_clipped_area_per_category = self.config.get('max_clipped_area_per_category')

    def apply(self, dataset: UnifiedDataset, output_dim: Optional[tuple] = None) -> UnifiedDataset:
        if not self.config.get('enable_cutout', True):
            logging.info("Cutout augmentation is disabled.")
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

        # Define max_clipped_area_per_category if not provided
        if not self.max_clipped_area_per_category:
            # Assign a default value if not specified, e.g., 0.3 (30%) for all categories
            self.max_clipped_area_per_category = {cat['id']: 0.3 for cat in dataset.categories}

        max_clipped_area_per_category = self.max_clipped_area_per_category

        # Create mapping from category names to IDs
        category_name_to_id = {cat['name']: cat['id'] for cat in dataset.categories}
        logging.debug(f"Category name to ID mapping: {category_name_to_id}")

        for img in dataset.images:
            image_path = img.file_name
            image = load_image(image_path)
            if image is None:
                logging.error(f"Failed to load image '{image_path}'. Skipping.")
                continue
            img_h, img_w = image.shape[:2]

            anns = image_id_to_annotations.get(img.id, [])

            # Initialize a set to track used shifts for this image
            used_shifts = set()

            if 'non_targeted' in self.modes:
                self.apply_non_targeted(
                    img, image, anns, img_w, img_h, image_id_offset, annotation_id_offset,
                    augmented_dataset, max_clipped_area_per_category, output_dim, used_shifts
                )
                # Update offsets
                image_id_offset = max([img.id for img in augmented_dataset.images], default=image_id_offset) + 1
                annotation_id_offset = max([ann.id for ann in augmented_dataset.annotations], default=annotation_id_offset) + 1

            if 'targeted' in self.modes:
                self.apply_targeted(
                    img, image, anns, img_w, img_h, image_id_offset, annotation_id_offset,
                    augmented_dataset, max_clipped_area_per_category, output_dim, used_shifts,
                    category_name_to_id
                )
                # Update offsets
                image_id_offset = max([img.id for img in augmented_dataset.images], default=image_id_offset) + 1
                annotation_id_offset = max([ann.id for ann in augmented_dataset.annotations], default=annotation_id_offset) + 1

        logging.info(f"Cutout augmentation completed.")
        return augmented_dataset

    def apply_non_targeted(self, img, image, anns, img_w, img_h, image_id_offset, annotation_id_offset,
                           augmented_dataset, max_clipped_area_per_category, output_dim, used_shifts):
        num_augmented_images = self.config['num_augmented_images']
        image_successful_aug = 0

        while image_successful_aug < num_augmented_images:
            cutout_applied = False
            # Decide whether to apply cutout based on probability
            prob = self.config['cutout_probability']
            if random.random() > prob:
                logging.info(f"Skipping cutout augmentation {image_successful_aug+1} for image ID {img.id} based on probability ({prob}).")
                continue  # Skip this augmentation

            # Apply Random Cutout
            augmented_image, masks = self.apply_random_cutout(
                image=image,
                img_w=img_w,
                img_h=img_h,
                num_cutouts=self.config['num_cutouts_per_image'],
                cutout_size_percent=self.config['cutout_size_percent'],
                used_shifts=used_shifts
            )

            if not masks:
                logging.info(f"No cutouts applied for augmentation {image_successful_aug+1} on image ID {img.id}. Skipping augmentation.")
                continue

            # Now, process annotations and decide whether to keep the augmented image
            success = self.process_cutout(
                img, augmented_image, anns, masks, img_w, img_h,
                image_id_offset, annotation_id_offset, augmented_dataset,
                max_clipped_area_per_category, output_dim, focus_category_ids=None
            )
            if success:
                image_id_offset += 1
                annotation_id_offset += len(anns)
                image_successful_aug += 1
                cutout_applied = True
            else:
                logging.info(f"Cutout augmentation for image ID {img.id} discarded during processing.")
                break  # No valid cutouts left to try
            if not cutout_applied:
                break  # No valid cutouts left to try

    def apply_targeted(self, img, image, anns, img_w, img_h, image_id_offset, annotation_id_offset,
                       augmented_dataset, max_clipped_area_per_category, output_dim, used_shifts,
                       category_name_to_id):
        # For targeted mode, focus on specific categories
        if not self.focus_categories:
            logging.warning("No focus categories provided for targeted mode.")
            return
        focus_category_ids = [category_name_to_id[cat_name] for cat_name in self.focus_categories if cat_name in category_name_to_id]
        if not focus_category_ids:
            logging.warning("Focus categories do not match any categories in the dataset.")
            return
        logging.debug(f"Focus category IDs: {focus_category_ids}")

        num_augmented_images = self.config['num_augmented_images']

        # Identify annotations of focus categories
        focus_anns = [ann for ann in anns if ann.category_id in focus_category_ids]
        if not focus_anns:
            logging.info(f"No focus category annotations in image ID {img.id}. Skipping.")
            return

        image_successful_aug = 0

        while image_successful_aug < num_augmented_images:
            # For each focus annotation, create a cutout that covers it completely with safety margin
            cutouts = []
            for ann in anns:
                coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
                if not coords:
                    continue  # Skip if coordinates are invalid
                ann_poly = Polygon(coords)
                if not ann_poly.is_valid:
                    ann_poly = ann_poly.buffer(0)
                minx, miny, maxx, maxy = ann_poly.bounds

                # Add safety margin
                margin = self.config.get('margin_percent', 0.05)  # 5% of image size
                x1 = max(int(minx - margin * img_w), 0)
                y1 = max(int(miny - margin * img_h), 0)
                x2 = min(int(maxx + margin * img_w), img_w)
                y2 = min(int(maxy + margin * img_h), img_h)

                cutout = {'ann': ann, 'bbox': (x1, y1, x2, y2)}
                cutouts.append(cutout)

            # Now, for focus annotations, shift the cutout to cause clipping just below max allowed
            # For non-focus annotations, keep the cutout in place

            # For each focus annotation, we need to find shifts that cause the desired clipping
            shifted_cutouts = []
            for cutout in cutouts:
                ann = cutout['ann']
                x1, y1, x2, y2 = cutout['bbox']
                category_id = ann.category_id

                if category_id in focus_category_ids:
                    # Shift the cutout in varying directions and distances
                    best_shift = self.find_best_shift_for_cutout(
                        ann, (x1, y1, x2, y2), img_w, img_h, max_clipped_area_per_category, used_shifts
                    )
                    if best_shift is None:
                        logging.info(f"Could not find suitable shift for annotation ID {ann.id} in image ID {img.id}.")
                        continue  # Skip this annotation
                    shift = best_shift
                    # Apply shift to cutout bbox
                    x1_shifted = x1 + shift[0]
                    y1_shifted = y1 + shift[1]
                    x2_shifted = x2 + shift[0]
                    y2_shifted = y2 + shift[1]
                    # Ensure the shifted bbox is within image boundaries
                    x1_shifted = max(0, x1_shifted)
                    y1_shifted = max(0, y1_shifted)
                    x2_shifted = min(img_w, x2_shifted)
                    y2_shifted = min(img_h, y2_shifted)
                    shifted_cutouts.append((x1_shifted, y1_shifted, x2_shifted, y2_shifted))
                    used_shifts.add(shift)
                else:
                    # Non-focus categories, keep the cutout in place
                    shifted_cutouts.append((x1, y1, x2, y2))

            if not shifted_cutouts:
                logging.info(f"No valid cutouts for image ID {img.id}. Skipping.")
                return

            # Apply the cutouts to the image
            augmented_image = image.copy()
            for x1, y1, x2, y2 in shifted_cutouts:
                augmented_image[int(y1):int(y2), int(x1):int(x2)] = 0

            masks = shifted_cutouts

            # Now, process annotations and decide whether to keep the augmented image
            success = self.process_cutout(
                img, augmented_image, anns, masks, img_w, img_h,
                image_id_offset, annotation_id_offset, augmented_dataset,
                max_clipped_area_per_category, output_dim, focus_category_ids=focus_category_ids
            )
            if success:
                image_id_offset += 1
                annotation_id_offset += len(anns)
                image_successful_aug += 1
            else:
                logging.info(f"Cutout augmentation for image ID {img.id} discarded during processing.")
                break  # No valid shifts left to try

        if image_successful_aug < num_augmented_images:
            logging.info(f"Could not generate {num_augmented_images} unique augmentations for image ID {img.id}. Generated {image_successful_aug} instead.")

    def find_best_shift_for_cutout(self, ann, bbox, img_w, img_h, max_clipped_area_per_category, used_shifts):
        x1, y1, x2, y2 = bbox
        category_id = ann.category_id
        max_allowed_reduction = max_clipped_area_per_category.get(category_id, 0.3)  # Default to 30%

        # Define possible directions and distances
        max_shift_distance_x = self.config.get('max_shift_percent', 1.0) * img_w
        max_shift_distance_y = self.config.get('max_shift_percent', 1.0) * img_h

        shift_steps = self.config.get('shift_steps', 20)

        # Possible directions: left, right, up, down
        directions = ['left', 'right', 'up', 'down']

        best_shift = None
        min_diff = float('inf')  # Difference between area reduction and max_allowed_reduction

        for direction in directions:
            if direction == 'left':
                shift_x_range = np.linspace(0, -max_shift_distance_x, shift_steps)
                shift_y_range = [0]
            elif direction == 'right':
                shift_x_range = np.linspace(0, max_shift_distance_x, shift_steps)
                shift_y_range = [0]
            elif direction == 'up':
                shift_x_range = [0]
                shift_y_range = np.linspace(0, -max_shift_distance_y, shift_steps)
            elif direction == 'down':
                shift_x_range = [0]
                shift_y_range = np.linspace(0, max_shift_distance_y, shift_steps)
            else:
                continue

            for shift_x in shift_x_range:
                for shift_y in shift_y_range:
                    shift_key = (shift_x, shift_y)
                    if shift_key in used_shifts:
                        continue

                    # Shift the cutout bbox
                    x1_shifted = x1 + shift_x
                    y1_shifted = y1 + shift_y
                    x2_shifted = x2 + shift_x
                    y2_shifted = y2 + shift_y

                    # Ensure the shifted bbox is within image boundaries
                    x1_shifted = max(0, x1_shifted)
                    y1_shifted = max(0, y1_shifted)
                    x2_shifted = min(img_w, x2_shifted)
                    y2_shifted = min(img_h, y2_shifted)

                    # If the shifted cutout does not overlap the polygon at all, skip
                    shifted_cutout_poly = box(x1_shifted, y1_shifted, x2_shifted, y2_shifted)
                    ann_poly_coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
                    if not ann_poly_coords:
                        continue
                    ann_poly = Polygon(ann_poly_coords)
                    if not ann_poly.is_valid:
                        ann_poly = ann_poly.buffer(0)
                    if not ann_poly.intersects(shifted_cutout_poly):
                        continue  # No overlap, move to next shift

                    # Create the shifted cutout polygon
                    cutout_poly = box(x1_shifted, y1_shifted, x2_shifted, y2_shifted)

                    # Clip the annotation polygon with the cutout polygon
                    clipped_poly = ann_poly.difference(cutout_poly)

                    if clipped_poly.is_empty:
                        new_area = 0
                    else:
                        new_area = clipped_poly.area

                    original_area = ann_poly.area
                    area_reduction_due_to_clipping = calculate_area_reduction(original_area, new_area)

                    logging.debug(f"Trying shift ({shift_x}, {shift_y}) for annotation ID {ann.id} with area reduction {area_reduction_due_to_clipping}")

                    # We want area_reduction_due_to_clipping just below max_allowed_reduction
                    if area_reduction_due_to_clipping > max_allowed_reduction:
                        continue  # Exceeds max allowed reduction

                    diff = max_allowed_reduction - area_reduction_due_to_clipping
                    if 0 <= diff < min_diff:
                        min_diff = diff
                        best_shift = (shift_x, shift_y)
                        if min_diff == 0:
                            return best_shift  # Found the perfect shift

        return best_shift

    def apply_random_cutout(self, image: np.ndarray, img_w: int, img_h: int, num_cutouts: int, cutout_size_percent: tuple, used_shifts=None):
        augmented_image = image.copy()
        masks = []

        for _ in range(num_cutouts):
            # Randomly choose the size of the cutout based on percentage
            height_percent = random.uniform(cutout_size_percent[0][0], cutout_size_percent[0][1])
            width_percent = random.uniform(cutout_size_percent[1][0], cutout_size_percent[1][1])
            height = int(height_percent * img_h)
            width = int(width_percent * img_w)

            # Randomly choose the top-left corner of the cutout
            x1 = random.randint(0, max(img_w - width, 0))
            y1 = random.randint(0, max(img_h - height, 0))
            x2 = x1 + width
            y2 = y1 + height

            mask_key = (x1, y1, x2, y2)
            if used_shifts is not None and mask_key in used_shifts:
                continue  # Skip if we've already used this mask
            if used_shifts is not None:
                used_shifts.add(mask_key)

            # Apply the cutout (mask with black color)
            augmented_image[y1:y2, x1:x2] = 0

            # Save the mask for annotation handling
            mask = (x1, y1, x2, y2)
            masks.append(mask)
            logging.debug(f"Applied random cutout: Top-left=({x1}, {y1}), Bottom-right=({x2}, {y2})")

        return augmented_image, masks

    def process_cutout(self, img, augmented_image, anns, masks, img_w, img_h, image_id_offset, annotation_id_offset, augmented_dataset, max_clipped_area_per_category, output_dim, focus_category_ids):
        # Create mask polygons for annotation clipping
        mask_polygons = [box(x1, y1, x2, y2) for (x1, y1, x2, y2) in masks]

        # Define the image boundary
        image_boundary = box(0, 0, img_w, img_h)

        # Subtract mask polygons from image boundary to get the valid region
        valid_region = image_boundary
        for mask in mask_polygons:
            valid_region = valid_region.difference(mask)

        # Ensure valid_region is valid geometry
        if not valid_region.is_valid:
            valid_region = valid_region.buffer(0)

        # Process annotations
        transformed_annotations = copy.deepcopy(anns)
        cleaned_anns = []

        discard_augmentation = False  # Flag to decide whether to discard the entire augmentation

        self.significant_clipping_occurred_on_focus_categories = False

        for ann in transformed_annotations:
            category_id = ann.category_id
            max_allowed_reduction = max_clipped_area_per_category.get(category_id, 0.3)  # Allow up to 30% reduction
            coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
            if not coords:
                logging.warning(f"Empty polygon for annotation ID {ann.id} in image ID {img.id}. Skipping annotation.")
                continue
            ann_poly = Polygon(coords)
            if not ann_poly.is_valid:
                ann_poly = ann_poly.buffer(0)
            # Clip the annotation polygon against the valid region
            clipped_poly = ann_poly.intersection(valid_region)
            original_area = ann_poly.area

            if clipped_poly.is_empty:
                logging.info(f"Annotation ID {ann.id} in image ID {img.id} is fully masked out by cutout. Skipping annotation.")
                continue  # Annotation is fully masked out

            # Handle MultiPolygon cases
            if isinstance(clipped_poly, Polygon):
                polygons_to_process = [clipped_poly]
            elif isinstance(clipped_poly, MultiPolygon):
                polygons_to_process = list(clipped_poly.geoms)
            else:
                logging.warning(f"Unsupported geometry type {type(clipped_poly)} for annotation ID {ann.id} in image ID {img.id}. Skipping annotation.")
                continue  # Unsupported geometry type

            for poly in polygons_to_process:
                if not poly.is_valid:
                    poly = poly.buffer(0)
                new_area = poly.area
                area_reduction_due_to_clipping = calculate_area_reduction(original_area, new_area)

                logging.debug(f"Annotation ID {ann.id}, category ID {category_id}, area reduction due to clipping: {area_reduction_due_to_clipping}, max allowed reduction: {max_allowed_reduction}")

                if area_reduction_due_to_clipping > max_allowed_reduction:
                    logging.warning(f"Cutout augmentation for image ID {img.id} discarded due to area reduction ({area_reduction_due_to_clipping:.6f}) exceeding threshold ({max_allowed_reduction}) for category {category_id}.")
                    discard_augmentation = True
                    break  # Discard the entire augmentation

                if focus_category_ids and category_id in focus_category_ids:
                    # Check if significant clipping occurred (e.g., more than 50% of max allowed)
                    if area_reduction_due_to_clipping >= max_allowed_reduction * 0.5:
                        self.significant_clipping_occurred_on_focus_categories = True

                is_polygon_clipped = area_reduction_due_to_clipping > 0.01

                if self.task == 'detection':
                    # For detection, use bounding boxes
                    coords = ensure_axis_aligned_rectangle(list(poly.exterior.coords))
                    if not coords:
                        logging.debug(f"No valid coordinates found after processing clipped polygons. Skipping polygon.")
                        continue
                else:
                    # For segmentation, collect exterior and interior coordinates
                    coords = []
                    # Exterior ring
                    exterior_coords = list(poly.exterior.coords)
                    if exterior_coords:
                        coords.extend([coord for point in exterior_coords for coord in point])
                    # Interior rings (holes)
                    for interior in poly.interiors:
                        interior_coords = list(interior.coords)
                        if interior_coords:
                            coords.extend([coord for point in interior_coords for coord in point])
                    if not coords:
                        logging.debug(f"No valid coordinates found after processing clipped polygons. Skipping polygon.")
                        continue

                # Update the annotation
                new_ann = UnifiedAnnotation(
                    id=annotation_id_offset,
                    image_id=image_id_offset,
                    category_id=ann.category_id,
                    polygon=coords,
                    iscrowd=ann.iscrowd,
                    area=new_area,
                    is_polygon_clipped=is_polygon_clipped,
                )
                cleaned_anns.append(new_ann)
                annotation_id_offset += 1

            if discard_augmentation:
                logging.info(f"Cutout augmentation for image ID {img.id} discarded due to high area reduction.")
                return False  # Discard the entire augmentation

        # If no polygons remain after masking, skip augmentation
        if not cleaned_anns:
            logging.info(f"Cutout augmentation for image ID {img.id} results in all polygons being fully masked. Skipping augmentation.")
            return False

        # In targeted mode, discard images that do not cause significant clipping on focus categories
        if focus_category_ids and not self.significant_clipping_occurred_on_focus_categories:
            logging.info(f"No significant clipping occurred on focus categories for image ID {img.id}. Skipping augmentation.")
            return False

        # Generate new filename
        filename, ext = os.path.splitext(os.path.basename(img.file_name))
        new_filename = f"{filename}_cutout_aug{uuid.uuid4().hex}{ext}"
        output_image_path = os.path.join(self.config['output_images_dir'], new_filename)

        # Save augmented image
        save_success = save_image(augmented_image, output_image_path)
        if not save_success:
            logging.error(f"Failed to save augmented image '{output_image_path}'. Skipping this augmentation.")
            return False

        # Create new image entry
        new_img = UnifiedImage(
            id=image_id_offset,
            file_name=output_image_path,
            width=augmented_image.shape[1],
            height=augmented_image.shape[0]
        )
        augmented_dataset.images.append(new_img)

        # Add cleaned annotations to the dataset
        for new_ann in cleaned_anns:
            augmented_dataset.annotations.append(new_ann)
            logging.info(f"Added annotation ID {new_ann.id} for image ID {image_id_offset}.")

        # Visualization
        if self.config['visualize_overlays'] and self.config['output_visualizations_dir']:
            visualization_filename = f"{os.path.splitext(new_filename)[0]}_viz{ext}"
            mosaic_visualize_transformed_overlays(
                transformed_image=augmented_image.copy(),
                cleaned_annotations=cleaned_anns,
                output_visualizations_dir=self.config['output_visualizations_dir'],
                new_filename=visualization_filename,
                task=self.task
            )

        logging.info(f"Cutout augmented image '{new_filename}' saved with {len(cleaned_anns)} annotations.")
        return True