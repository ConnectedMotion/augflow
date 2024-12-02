import os
import copy
import random
import numpy as np
import cv2
import logging
from shapely.geometry import Polygon, box, MultiPolygon
from shapely import affinity
from typing import Optional, List, Dict, Tuple

# Import base class
from .base import Augmentation

# Import helper functions from utils
from augflow.utils.images import load_image, save_image, mosaic_visualize_transformed_overlays, pad_image_to_size
from augflow.utils.unified_format import UnifiedDataset, UnifiedImage, UnifiedAnnotation
from augflow.utils.annotations import ensure_axis_aligned_rectangle, calculate_area_reduction
from augflow.utils.configs import crop_default_config

import uuid


class CropAugmentation(Augmentation):
    def __init__(self, config=None, task: str = 'detection', modes: List[str] = None, focus_categories: Optional[List[str]] = None):
        super().__init__()
        self.task = task.lower()
        self.config = crop_default_config.copy()
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
        if not self.config.get('enable_cropping', True):
            logging.info("Cropping augmentation is disabled.")
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
            self.max_clipped_area_per_category = {cat['id']: 0.6 for cat in dataset.categories}

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

            # Initialize a set to track used crops for this image
            used_crops = set()

            if 'non_targeted' in self.modes:
                self.apply_non_targeted(
                    img, image, anns, img_w, img_h, image_id_offset, annotation_id_offset,
                    augmented_dataset, max_clipped_area_per_category, output_dim, used_crops
                )
                # Update offsets
                image_id_offset = max([img.id for img in augmented_dataset.images], default=image_id_offset) + 1
                annotation_id_offset = max([ann.id for ann in augmented_dataset.annotations], default=annotation_id_offset) + 1

            if 'targeted' in self.modes:
                self.apply_targeted(
                    img, image, anns, img_w, img_h, image_id_offset, annotation_id_offset,
                    augmented_dataset, max_clipped_area_per_category, output_dim, used_crops,
                    category_name_to_id
                )
                # Update offsets
                image_id_offset = max([img.id for img in augmented_dataset.images], default=image_id_offset) + 1
                annotation_id_offset = max([ann.id for ann in augmented_dataset.annotations], default=annotation_id_offset) + 1

        logging.info(f"Cropping augmentation completed.")
        return augmented_dataset

    def apply_non_targeted(self, img, image, anns, img_w, img_h, image_id_offset, annotation_id_offset,
                           augmented_dataset, max_clipped_area_per_category, output_dim, used_crops):
        num_crops = self.config['num_crops_per_image']
        image_successful_aug = 0
        max_attempts = 50
        attempts = 0

        while image_successful_aug < num_crops and attempts < max_attempts:
            attempts += 1
            # Decide whether to apply crop based on probability
            prob = self.config['crop_probability']
            if random.random() > prob:
                logging.info(f"Skipping crop augmentation {image_successful_aug+1} for image ID {img.id} based on probability ({prob}).")
                continue  # Skip this augmentation

            # Apply Random Crop
            crop_applied, padded_image, crop_coords, pad_left, pad_top = self.apply_random_crop(
                image=image,
                img_w=img_w,
                img_h=img_h,
                crop_size_percent=self.config['crop_size_percent'],
                used_crops=used_crops
            )

            if not crop_applied:
                logging.info(f"No crops applied for augmentation {image_successful_aug+1} on image ID {img.id}. Skipping augmentation.")
                continue

            # Now, process annotations and decide whether to keep the augmented image
            success = self.process_crop(
                img, padded_image, anns, crop_coords, img_w, img_h,
                image_id_offset, annotation_id_offset, augmented_dataset,
                max_clipped_area_per_category, output_dim, focus_category_ids=None, pad_left=pad_left, pad_top=pad_top, allow_empty_annotations=True
            )
            if success:
                image_id_offset += 1
                annotation_id_offset += len(anns)
                image_successful_aug += 1
            else:
                logging.info(f"Crop augmentation for image ID {img.id} discarded during processing.")
                continue  # Try next crop

        if attempts == max_attempts:
            logging.info(f"Reached maximum attempts ({max_attempts}) for image ID {img.id} in non-targeted mode.")

    def apply_targeted(self, img, image, anns, img_w, img_h, image_id_offset, annotation_id_offset,
                       augmented_dataset, max_clipped_area_per_category, output_dim, used_crops,
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

        num_crops = self.config['num_crops_per_image']

        # Identify annotations of focus categories
        focus_anns = [ann for ann in anns if ann.category_id in focus_category_ids]
        if not focus_anns:
            logging.info(f"No focus category annotations in image ID {img.id}. Skipping.")
            return

        image_successful_aug = 0

        for ann in focus_anns:
            if image_successful_aug >= num_crops:
                break  # Reached the number of crops per image

            # Create a crop rectangle around the annotation with safety margin
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

            # Initial crop rectangle
            crop_rect = (x1, y1, x2 - x1, y2 - y1)

            # Now, shift the crop rectangle to cause clipping just below max allowed
            best_shift = self.find_best_shift_for_crop(
                ann, crop_rect, img_w, img_h, max_clipped_area_per_category, used_crops
            )
            if best_shift is None:
                logging.info(f"Could not find suitable shift for annotation ID {ann.id} in image ID {img.id}.")
                continue  # Skip this annotation
            shift = best_shift
            # Apply shift to crop rectangle
            x_shifted = x1 + shift[0]
            y_shifted = y1 + shift[1]
            w_shifted = crop_rect[2]
            h_shifted = crop_rect[3]
            # Ensure the shifted crop is within image boundaries
            x_shifted = max(0, x_shifted)
            y_shifted = max(0, y_shifted)
            if x_shifted + w_shifted > img_w:
                x_shifted = img_w - w_shifted
            if y_shifted + h_shifted > img_h:
                y_shifted = img_h - h_shifted

            # Apply the crop to the image
            cropped_image = image[int(y_shifted):int(y_shifted + h_shifted), int(x_shifted):int(x_shifted + w_shifted)]

            # Pad the cropped image to the original size
            padded_image, pad_left, pad_top = pad_image_to_size(
                cropped_image,
                desired_size=(img_w, img_h),
                pad_color=self.config['padding_color']
            )

            # Now, process annotations and decide whether to keep the augmented image
            success = self.process_crop(
                img, padded_image, anns, (x_shifted, y_shifted, w_shifted, h_shifted), img_w, img_h,
                image_id_offset, annotation_id_offset, augmented_dataset,
                max_clipped_area_per_category, output_dim, focus_category_ids=focus_category_ids, pad_left=pad_left, pad_top=pad_top
            )
            if success:
                image_id_offset += 1
                annotation_id_offset += len(anns)
                image_successful_aug += 1
            else:
                logging.info(f"Crop augmentation for image ID {img.id} discarded during processing.")
                continue  # Try next annotation

    def find_best_shift_for_crop(self, ann, crop_rect, img_w, img_h, max_clipped_area_per_category, used_crops):
        x1, y1, w, h = crop_rect
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
                    shift_key = (x1 + shift_x, y1 + shift_y, w, h)
                    if shift_key in used_crops:
                        continue

                    # Shift the crop rectangle
                    x_shifted = x1 + shift_x
                    y_shifted = y1 + shift_y
                    w_shifted = w
                    h_shifted = h

                    # Ensure the shifted crop is within image boundaries
                    x_shifted = max(0, x_shifted)
                    y_shifted = max(0, y_shifted)
                    if x_shifted + w_shifted > img_w:
                        x_shifted = img_w - w_shifted
                    if y_shifted + h_shifted > img_h:
                        y_shifted = img_h - h_shifted

                    # If the shifted crop does not overlap the polygon at all, skip
                    shifted_crop_poly = box(x_shifted, y_shifted, x_shifted + w_shifted, y_shifted + h_shifted)
                    ann_poly_coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
                    if not ann_poly_coords:
                        continue
                    ann_poly = Polygon(ann_poly_coords)
                    if not ann_poly.is_valid:
                        ann_poly = ann_poly.buffer(0)
                    if not ann_poly.intersects(shifted_crop_poly):
                        continue  # No overlap, move to next shift

                    # Clip the annotation polygon with the shifted crop polygon
                    clipped_poly = ann_poly.intersection(shifted_crop_poly)

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
                            shift_key = (x1 + best_shift[0], y1 + best_shift[1], w, h)
                            used_crops.add(shift_key)
                            return best_shift  # Found the perfect shift

        if best_shift is not None:
            shift_key = (x1 + best_shift[0], y1 + best_shift[1], w, h)
            used_crops.add(shift_key)
        return best_shift

    def apply_random_crop(self, image: np.ndarray, img_w: int, img_h: int, crop_size_percent: tuple, used_crops=None):
        max_attempts = 10
        attempts = 0

        while attempts < max_attempts:
            attempts += 1
            # Randomly choose the size of the crop based on percentage
            width_percent = random.uniform(crop_size_percent[0][0], crop_size_percent[0][1])
            height_percent = random.uniform(crop_size_percent[1][0], crop_size_percent[1][1])
            crop_w = int(width_percent * img_w)
            crop_h = int(height_percent * img_h)

            # Randomly choose the top-left corner of the crop
            x1 = random.randint(0, max(img_w - crop_w, 0))
            y1 = random.randint(0, max(img_h - crop_h, 0))

            crop_key = (x1, y1, crop_w, crop_h)
            if used_crops is not None and crop_key in used_crops:
                continue  # Try another crop
            if used_crops is not None:
                used_crops.add(crop_key)

            # Apply the crop
            cropped_image = image[y1:y1 + crop_h, x1:x1 + crop_w]

            # Pad the cropped image to reach original image size
            padded_image, pad_left, pad_top = pad_image_to_size(
                cropped_image,
                desired_size=(img_w, img_h),
                pad_color=self.config['padding_color']
            )

            logging.debug(f"Applied random crop: Top-left=({x1}, {y1}), Width={crop_w}, Height={crop_h}")

            return True, padded_image, (x1, y1, crop_w, crop_h), pad_left, pad_top

        logging.info(f"Could not find a unique crop after {max_attempts} attempts.")
        return False, None, None, None, None

    def process_crop(self, img, padded_image, anns, crop_coords, img_w, img_h, image_id_offset, annotation_id_offset, augmented_dataset, max_clipped_area_per_category, output_dim, focus_category_ids, pad_left, pad_top, allow_empty_annotations=False):
        x_crop, y_crop, w_crop, h_crop = crop_coords

        # Scale factors (after padding, the image size is same as original)
        scale_x = 1.0
        scale_y = 1.0

        # Define the crop boundary
        crop_boundary = box(x_crop, y_crop, x_crop + w_crop, y_crop + h_crop)

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
            # Clip the annotation polygon against the crop boundary
            clipped_poly = ann_poly.intersection(crop_boundary)
            original_area = ann_poly.area

            if clipped_poly.is_empty:
                logging.info(f"Annotation ID {ann.id} in image ID {img.id} is fully outside the crop area. Skipping annotation.")
                continue  # Annotation is fully outside the crop area

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
                    logging.warning(f"Crop augmentation for image ID {img.id} discarded due to area reduction ({area_reduction_due_to_clipping:.6f}) exceeding threshold ({max_allowed_reduction}) for category {category_id}.")
                    discard_augmentation = True
                    break  # Discard the entire augmentation

                if focus_category_ids and category_id in focus_category_ids:
                    # Check if significant clipping occurred (e.g., more than 50% of max allowed)
                    if area_reduction_due_to_clipping >= max_allowed_reduction * 0.5:
                        self.significant_clipping_occurred_on_focus_categories = True

                is_polygon_clipped = area_reduction_due_to_clipping > 0.01

                # Adjust coordinates to new image space (after padding)
                adjusted_coords = []
                for px, py in poly.exterior.coords:
                    new_px = px - x_crop + pad_left  # Adjust for padding
                    new_py = py - y_crop + pad_top   # Adjust for padding
                    # Apply scaling factors if any (currently 1.0)
                    new_px = new_px * scale_x
                    new_py = new_py * scale_y
                    adjusted_coords.append((new_px, new_py))

                if self.task == 'detection':
                    # For detection, use bounding boxes
                    coords = ensure_axis_aligned_rectangle(adjusted_coords)
                    if not coords:
                        logging.debug(f"No valid coordinates found after processing clipped polygons. Skipping polygon.")
                        continue
                else:
                    coords = adjusted_coords

                # Update the annotation
                new_ann = UnifiedAnnotation(
                    id=annotation_id_offset,
                    image_id=image_id_offset,
                    category_id=ann.category_id,
                    polygon=[coord for point in coords for coord in point],
                    iscrowd=ann.iscrowd,
                    area=new_area,
                    is_polygon_clipped=is_polygon_clipped,
                )
                cleaned_anns.append(new_ann)
                annotation_id_offset += 1

            if discard_augmentation:
                logging.info(f"Crop augmentation for image ID {img.id} discarded due to high area reduction.")
                return False  # Discard the entire augmentation

        # In non-targeted mode, allow empty annotations
        if not cleaned_anns and not allow_empty_annotations:
            logging.info(f"Crop augmentation for image ID {img.id} results in all annotations being fully outside the crop area. Skipping augmentation.")
            return False

        # In targeted mode, discard images that do not cause significant clipping on focus categories
        if focus_category_ids and not self.significant_clipping_occurred_on_focus_categories:
            logging.info(f"No significant clipping occurred on focus categories for image ID {img.id}. Skipping augmentation.")
            return False

        # Generate new filename
        filename, ext = os.path.splitext(os.path.basename(img.file_name))
        new_filename = f"{filename}_crop_aug{uuid.uuid4().hex}{ext}"
        output_image_path = os.path.join(self.config['output_images_dir'], new_filename)

        # Save augmented image
        save_success = save_image(padded_image, output_image_path)
        if not save_success:
            logging.error(f"Failed to save augmented image '{output_image_path}'. Skipping this augmentation.")
            return False

        # Create new image entry
        new_img = UnifiedImage(
            id=image_id_offset,
            file_name=output_image_path,
            width=padded_image.shape[1],
            height=padded_image.shape[0]
        )
        augmented_dataset.images.append(new_img)

        # Add cleaned annotations to the dataset
        for new_ann in cleaned_anns:
            augmented_dataset.annotations.append(new_ann)
            logging.info(f"Added annotation ID {new_ann.id} for image ID {image_id_offset}.")

        # Visualization
        if self.config['visualize_overlays'] and self.config['output_visualizations_dir']:
            visualization_filename = f"{os.path.splitext(new_filename)[0]}_viz{ext}"
            visualization_path = os.path.join(self.config['output_visualizations_dir'], visualization_filename)
            mosaic_visualize_transformed_overlays(
                transformed_image=padded_image.copy(),
                cleaned_annotations=cleaned_anns,
                output_visualizations_dir=self.config['output_visualizations_dir'],
                new_filename=visualization_filename,
                task=self.task
            )

        logging.info(f"Crop augmented image '{new_filename}' saved with {len(cleaned_anns)} annotations.")
        return True

