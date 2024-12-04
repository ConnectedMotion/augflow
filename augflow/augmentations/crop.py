import os
import copy
import random
import numpy as np
import cv2
import logging
from shapely.geometry import Polygon, box, MultiPolygon
from typing import Optional, List, Dict, Tuple

# Import base class
from .base import Augmentation

# Import helper functions from utils
from augflow.utils.images import (
    load_image,
    save_image,
    mosaic_visualize_transformed_overlays,
    pad_image_to_size,
)
from augflow.utils.unified_format import (
    UnifiedDataset,
    UnifiedImage,
    UnifiedAnnotation,
)
from augflow.utils.annotations import (
    ensure_axis_aligned_rectangle,
    calculate_area_reduction,
)
from augflow.utils.configs import crop_default_config

import uuid


class CropAugmentation(Augmentation):
    def __init__(self, config=None, task: str = 'detection'):
        super().__init__()
        self.task = task.lower()
        self.config = crop_default_config.copy()
        if config:
            self.config.update(config)
        self.modes = [
            mode.lower() for mode in self.config.get('modes', [])
        ]
        self.focus_categories = self.config.get('focus_categories', [])
        random.seed(self.config.get('random_seed', 42))
        np.random.seed(self.config.get('random_seed', 42))

        # Ensure output directories exist
        os.makedirs(self.config['output_images_dir'], exist_ok=True)
        if self.config.get('visualize_overlays') and self.config.get(
            'output_visualizations_dir'
        ):
            os.makedirs(
                self.config['output_visualizations_dir'], exist_ok=True
            )

        # Initialize max_clipped_area_per_category
        self.max_clipped_area_per_category = self.config.get(
            'max_clipped_area_per_category'
        )

    def apply(
        self, dataset: UnifiedDataset, output_dim: Optional[tuple] = None
    ) -> UnifiedDataset:
        if not self.config.get('enable_cropping', True):
            logging.info("Cropping augmentation is disabled.")
            return dataset  # Return the original dataset

        augmented_dataset = UnifiedDataset(
            images=[],
            annotations=[],
            categories=copy.deepcopy(dataset.categories),
        )

        # Get the maximum existing image and annotation IDs
        existing_image_ids = [img.id for img in dataset.images]
        existing_annotation_ids = [ann.id for ann in dataset.annotations]
        image_id_offset = (
            max(existing_image_ids) + 1 if existing_image_ids else 1
        )
        annotation_id_offset = (
            max(existing_annotation_ids) + 1 if existing_annotation_ids else 1
        )

        # Create a mapping from image_id to annotations
        image_id_to_annotations = {}
        for ann in dataset.annotations:
            image_id_to_annotations.setdefault(ann.image_id, []).append(ann)

        # Define max_clipped_area_per_category if not provided
        if not self.max_clipped_area_per_category:
            # Assign a default value if not specified, e.g., 0.6 (60%)
            self.max_clipped_area_per_category = {
                cat['id']: 0.6 for cat in dataset.categories
            }

        max_clipped_area_per_category = self.max_clipped_area_per_category

        # Create mapping from category names to IDs
        category_name_to_id = {
            cat['name']: cat['id'] for cat in dataset.categories
        }
        logging.debug(f"Category name to ID mapping: {category_name_to_id}")

        for img in dataset.images:
            image_path = img.file_name
            image = load_image(image_path)
            if image is None:
                logging.error(
                    f"Failed to load image '{image_path}'. Skipping."
                )
                continue
            img_h, img_w = image.shape[:2]

            anns = image_id_to_annotations.get(img.id, [])

            # Initialize a list to track used crops for this image
            used_crops = []

            if 'non_targeted' in self.modes:
                self.apply_non_targeted(
                    img,
                    image,
                    anns,
                    img_w,
                    img_h,
                    image_id_offset,
                    annotation_id_offset,
                    augmented_dataset,
                    max_clipped_area_per_category,
                    output_dim,
                    used_crops,
                )
                # Update offsets
                image_id_offset = (
                    max(
                        [img.id for img in augmented_dataset.images],
                        default=image_id_offset,
                    )
                    + 1
                )
                annotation_id_offset = (
                    max(
                        [
                            ann.id
                            for ann in augmented_dataset.annotations
                        ],
                        default=annotation_id_offset,
                    )
                    + 1
                )

            if 'targeted' in self.modes:
                self.apply_targeted(
                    img,
                    image,
                    anns,
                    img_w,
                    img_h,
                    image_id_offset,
                    annotation_id_offset,
                    augmented_dataset,
                    max_clipped_area_per_category,
                    output_dim,
                    used_crops,
                    category_name_to_id,
                )
                # Update offsets
                image_id_offset = (
                    max(
                        [img.id for img in augmented_dataset.images],
                        default=image_id_offset,
                    )
                    + 1
                )
                annotation_id_offset = (
                    max(
                        [
                            ann.id
                            for ann in augmented_dataset.annotations
                        ],
                        default=annotation_id_offset,
                    )
                    + 1
                )

        logging.info(f"Cropping augmentation completed.")
        return augmented_dataset

    def apply_non_targeted(
        self,
        img,
        image,
        anns,
        img_w,
        img_h,
        image_id_offset,
        annotation_id_offset,
        augmented_dataset,
        max_clipped_area_per_category,
        output_dim,
        used_crops,
    ):
        num_crops = self.config['num_crops_per_image']
        image_successful_aug = 0
        max_attempts = 10
        attempts = 0

        iou_threshold = self.config.get('crop_iou_threshold', 0.5)

        while (
            image_successful_aug < num_crops and attempts < max_attempts
        ):
            attempts += 1
            # Decide whether to apply crop based on probability
            prob = self.config.get('crop_probability')
            if random.random() > prob:
                logging.info(
                    f"Skipping crop augmentation {image_successful_aug+1} "
                    f"for image ID {img.id} based on probability ({prob})."
                )
                continue  # Skip this augmentation

            # Apply Random Crop
            (
                crop_applied,
                padded_image,
                crop_coords,
                pad_left,
                pad_top,
            ) = self.apply_random_crop(
                image=image,
                img_w=img_w,
                img_h=img_h,
                crop_size_percent=self.config['crop_size_percent'],
                used_crops=used_crops,
                iou_threshold=iou_threshold,
            )

            if not crop_applied:
                continue

            # Now, process annotations and decide whether to keep
            # the augmented image
            success = self.process_crop(
                img,
                padded_image,
                anns,
                crop_coords,
                img_w,
                img_h,
                image_id_offset,
                annotation_id_offset,
                augmented_dataset,
                max_clipped_area_per_category,
                output_dim,
                focus_category_ids=None,
                pad_left=pad_left,
                pad_top=pad_top,
                allow_empty_annotations=True,
            )
            if success:
                used_crops.append(crop_coords)
                image_id_offset += 1
                annotation_id_offset += len(anns)
                image_successful_aug += 1
            else:
                logging.info(
                    f"Crop augmentation for image ID {img.id} "
                    f"discarded during processing."
                )
                continue  # Try next crop

        if attempts == max_attempts and image_successful_aug < num_crops:
            logging.info(
                f"Reached maximum attempts ({max_attempts}) for "
                f"image ID {img.id} in non-targeted mode."
            )

    def apply_targeted(
        self,
        img,
        image,
        anns,
        img_w,
        img_h,
        image_id_offset,
        annotation_id_offset,
        augmented_dataset,
        max_clipped_area_per_category,
        output_dim,
        used_crops,
        category_name_to_id,
    ):
        # For targeted mode, focus on specific categories
        if not self.focus_categories:
            logging.warning(
                "No focus categories provided for targeted mode."
            )
            return
        focus_category_ids = [
            category_name_to_id[cat_name]
            for cat_name in self.focus_categories
            if cat_name in category_name_to_id
        ]
        if not focus_category_ids:
            logging.warning(
                "Focus categories do not match any categories "
                "in the dataset."
            )
            return
        logging.debug(f"Focus category IDs: {focus_category_ids}")

        # Group annotations by focus category
        focus_anns_by_category = {
            cat_id: [] for cat_id in focus_category_ids
        }
        for ann in anns:
            if ann.category_id in focus_category_ids:
                focus_anns_by_category[ann.category_id].append(ann)

        iou_threshold = self.config.get('crop_iou_threshold', 0.5)

        for cat_id, focus_anns in focus_anns_by_category.items():
            if not focus_anns:
                continue
            num_crops = self.config['num_crops_per_image']
            image_successful_aug = 0
            max_attempts = 100
            attempts = 0

            while (
                image_successful_aug < num_crops
                and attempts < max_attempts
            ):
                attempts += 1
                ann = random.choice(focus_anns)

                # Create a crop rectangle around the annotation
                coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
                if not coords:
                    continue  # Skip if coordinates are invalid
                ann_poly = Polygon(coords)
                if not ann_poly.is_valid:
                    ann_poly = ann_poly.buffer(0)
                minx, miny, maxx, maxy = ann_poly.bounds

                # Add safety margin
                margin = self.config.get('margin_percent', 0.05)
                x1 = max(int(minx - margin * img_w), 0)
                y1 = max(int(miny - margin * img_h), 0)
                x2 = min(int(maxx + margin * img_w), img_w)
                y2 = min(int(maxy + margin * img_h), img_h)

                # Initial crop rectangle
                crop_rect = (x1, y1, x2 - x1, y2 - y1)

                # Now, shift the crop rectangle
                best_shift = self.find_best_shift_for_crop(
                    ann,
                    crop_rect,
                    img_w,
                    img_h,
                    max_clipped_area_per_category,
                    used_crops,
                    iou_threshold,
                )
                if best_shift is None:
                    logging.info(
                        f"Could not find suitable shift for "
                        f"annotation ID {ann.id} in image ID {img.id} after {attempts} attempts."
                    )
                    break  # No suitable shift found, move to next annotation
                shift = best_shift
                # Apply shift to crop rectangle
                x_shifted = x1 + shift[0]
                y_shifted = y1 + shift[1]
                w_shifted = crop_rect[2]
                h_shifted = crop_rect[3]
                # Ensure the shifted crop is within image boundaries
                x_shifted = max(0, min(x_shifted, img_w - w_shifted))
                y_shifted = max(0, min(y_shifted, img_h - h_shifted))

                # Apply the crop to the image
                cropped_image = image[
                    int(y_shifted): int(y_shifted + h_shifted),
                    int(x_shifted): int(x_shifted + w_shifted),
                ]

                # Pad the cropped image to the original size
                (
                    padded_image,
                    pad_left,
                    pad_top,
                ) = pad_image_to_size(
                    cropped_image,
                    desired_size=(img_w, img_h),
                    pad_color=self.config['padding_color'],
                )

                crop_coords = (x_shifted, y_shifted, w_shifted, h_shifted)

                # Check if the crop is similar to previous crops
                if any(
                    self.compute_iou(crop_coords, prev_crop) > iou_threshold
                    for prev_crop in used_crops
                ):
                    continue

                # Now, process annotations
                success = self.process_crop(
                    img,
                    padded_image,
                    anns,
                    crop_coords,
                    img_w,
                    img_h,
                    image_id_offset,
                    annotation_id_offset,
                    augmented_dataset,
                    max_clipped_area_per_category,
                    output_dim,
                    focus_category_ids=[cat_id],
                    pad_left=pad_left,
                    pad_top=pad_top,
                )
                if success:
                    used_crops.append(crop_coords)
                    image_id_offset += 1
                    annotation_id_offset += len(anns)
                    image_successful_aug += 1
                else:
                    logging.info(
                        f"Crop augmentation for image ID {img.id} "
                        f"discarded during processing."
                    )
                    continue  # Try next attempt

            if image_successful_aug < num_crops:
                logging.info(
                    f"Could not generate enough crops for image ID {img.id} "
                    f"and category ID {cat_id} in targeted mode after "
                    f"{attempts} attempts."
                )

    def find_best_shift_for_crop(
        self,
        ann,
        crop_rect,
        img_w,
        img_h,
        max_clipped_area_per_category,
        used_crops,
        iou_threshold,
    ):
        x1, y1, w, h = crop_rect
        category_id = ann.category_id
        max_allowed_reduction = max_clipped_area_per_category.get(
            category_id, 0.6  # Default to 60%
        )

        # Initialize variables
        best_shift = (0, 0)
        min_diff = float('inf')

        # Directions to shift: (dx, dy)
        directions = [
            (0, -1),  # Up
            (0, 1),   # Down
            (-1, 0),  # Left
            (1, 0),   # Right
        ]

        # Step size in pixels
        shift_step = self.config.get('shift_step_size', 10)

        # Maximum number of steps to shift in each direction
        max_shift_steps = self.config.get('max_shift_steps', 50)

        for direction in directions:
            dx, dy = direction
            for step in range(1, max_shift_steps + 1):
                shift_x = dx * shift_step * step
                shift_y = dy * shift_step * step

                # Shift the crop rectangle
                x_shifted = x1 + shift_x
                y_shifted = y1 + shift_y
                w_shifted = w
                h_shifted = h

                # Ensure the shifted crop is within image boundaries
                x_shifted = max(0, min(x_shifted, img_w - w_shifted))
                y_shifted = max(0, min(y_shifted, img_h - h_shifted))

                # Check if the crop is similar to previous crops
                crop_coords = (x_shifted, y_shifted, w_shifted, h_shifted)
                if any(
                    self.compute_iou(crop_coords, prev_crop) > iou_threshold
                    for prev_crop in used_crops
                ):
                    continue  # Too similar to previous crops

                # Compute the area reduction due to clipping
                area_reduction_due_to_clipping = self.calculate_area_reduction_for_shift(
                    ann, crop_coords
                )

                if area_reduction_due_to_clipping is None:
                    continue  # Invalid clipping, skip

                if area_reduction_due_to_clipping > max_allowed_reduction:
                    # Exceeds max allowed reduction, no need to shift further in this direction
                    break

                if area_reduction_due_to_clipping < 0.01:
                    # Not significant clipping, continue shifting
                    continue

                diff = max_allowed_reduction - area_reduction_due_to_clipping
                if 0 <= diff < min_diff:
                    min_diff = diff
                    best_shift = (shift_x, shift_y)
                    if min_diff == 0:
                        return best_shift  # Found the best possible shift

        if min_diff < float('inf'):
            return best_shift
        else:
            return None  # No suitable shift found

    def calculate_area_reduction_for_shift(self, ann, crop_coords):
        x_shifted, y_shifted, w_shifted, h_shifted = crop_coords
        # Define the shifted crop polygon
        shifted_crop_poly = box(
            x_shifted,
            y_shifted,
            x_shifted + w_shifted,
            y_shifted + h_shifted,
        )
        ann_poly_coords = list(
            zip(ann.polygon[0::2], ann.polygon[1::2])
        )
        if not ann_poly_coords:
            return None
        ann_poly = Polygon(ann_poly_coords)
        if not ann_poly.is_valid:
            ann_poly = ann_poly.buffer(0)
        if not ann_poly.intersects(shifted_crop_poly):
            return None  # No overlap, invalid shift

        # Clip the annotation polygon
        clipped_poly = ann_poly.intersection(shifted_crop_poly)

        if clipped_poly.is_empty:
            new_area = 0
        else:
            new_area = clipped_poly.area

        original_area = ann_poly.area
        area_reduction_due_to_clipping = calculate_area_reduction(
            original_area, new_area
        )

        return area_reduction_due_to_clipping

    def apply_random_crop(
        self,
        image: np.ndarray,
        img_w: int,
        img_h: int,
        crop_size_percent: tuple,
        used_crops=None,
        iou_threshold=0.5,
    ):
        max_attempts = 50  # Increased attempts for better chance
        attempts = 0

        while attempts < max_attempts:
            attempts += 1
            # Randomly choose the size of the crop
            width_percent = random.uniform(
                crop_size_percent[0][0], crop_size_percent[0][1]
            )
            height_percent = random.uniform(
                crop_size_percent[1][0], crop_size_percent[1][1]
            )
            crop_w = int(width_percent * img_w)
            crop_h = int(height_percent * img_h)

            # Randomly choose the top-left corner
            x1 = random.randint(0, max(img_w - crop_w, 0))
            y1 = random.randint(0, max(img_h - crop_h, 0))

            crop_coords = (x1, y1, crop_w, crop_h)
            if used_crops is not None and any(
                self.compute_iou(crop_coords, prev_crop) > iou_threshold
                for prev_crop in used_crops
            ):
                continue  # Crop too similar to previous ones

            # Apply the crop
            cropped_image = image[
                y1: y1 + crop_h, x1: x1 + crop_w
            ]

            # Pad the cropped image
            (
                padded_image,
                pad_left,
                pad_top,
            ) = pad_image_to_size(
                cropped_image,
                desired_size=(img_w, img_h),
                pad_color=self.config['padding_color'],
            )

            return (
                True,
                padded_image,
                crop_coords,
                pad_left,
                pad_top,
            )

        logging.info(
            f"Could not find a unique crop after {max_attempts} attempts."
        )
        return False, None, None, None, None

    def process_crop(
        self,
        img,
        padded_image,
        anns,
        crop_coords,
        img_w,
        img_h,
        image_id_offset,
        annotation_id_offset,
        augmented_dataset,
        max_clipped_area_per_category,
        output_dim,
        focus_category_ids,
        pad_left,
        pad_top,
        allow_empty_annotations=False,
    ):
        x_crop, y_crop, w_crop, h_crop = crop_coords

        # Scale factors (after padding, the image size is same)
        scale_x = 1.0
        scale_y = 1.0

        # Define the crop boundary
        crop_boundary = box(
            x_crop, y_crop, x_crop + w_crop, y_crop + h_crop
        )

        # Process annotations
        transformed_annotations = copy.deepcopy(anns)
        cleaned_anns = []

        discard_augmentation = False

        self.significant_clipping_occurred_on_focus_categories = False

        for ann in transformed_annotations:
            category_id = ann.category_id
            max_allowed_reduction = max_clipped_area_per_category.get(
                category_id, 0.6  # Default to 60%
            )
            coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
            if not coords:
                continue
            ann_poly = Polygon(coords)
            if not ann_poly.is_valid:
                ann_poly = ann_poly.buffer(0)
            # Clip the annotation polygon
            clipped_poly = ann_poly.intersection(crop_boundary)
            original_area = ann_poly.area

            if clipped_poly.is_empty:
                continue  # Annotation is fully outside

            # Handle MultiPolygon cases
            if isinstance(clipped_poly, Polygon):
                polygons_to_process = [clipped_poly]
            elif isinstance(clipped_poly, MultiPolygon):
                polygons_to_process = list(clipped_poly.geoms)
            else:
                continue  # Unsupported geometry type

            for poly in polygons_to_process:
                if not poly.is_valid:
                    poly = poly.buffer(0)
                new_area = poly.area
                area_reduction_due_to_clipping = calculate_area_reduction(
                    original_area, new_area
                )

                if (
                    area_reduction_due_to_clipping
                    > max_allowed_reduction
                ):
                    discard_augmentation = True
                    break  # Discard the entire augmentation

                if focus_category_ids and category_id in focus_category_ids:
                    # Check if significant clipping occurred
                    if (
                        area_reduction_due_to_clipping
                        >= max_allowed_reduction * 0.5
                    ):
                        self.significant_clipping_occurred_on_focus_categories = True

                is_polygon_clipped = area_reduction_due_to_clipping > 0.01

                # Adjust coordinates
                adjusted_coords = []
                for px, py in poly.exterior.coords:
                    new_px = px - x_crop + pad_left
                    new_py = py - y_crop + pad_top
                    new_px = new_px * scale_x
                    new_py = new_py * scale_y
                    adjusted_coords.append((new_px, new_py))

                if self.task == 'detection':
                    # For detection, use bounding boxes
                    coords = ensure_axis_aligned_rectangle(adjusted_coords)
                    if not coords:
                        continue
                else:
                    coords = adjusted_coords

                # Update the annotation
                new_ann = UnifiedAnnotation(
                    id=annotation_id_offset,
                    image_id=image_id_offset,
                    category_id=ann.category_id,
                    polygon=[
                        coord for point in coords for coord in point
                    ],
                    iscrowd=ann.iscrowd,
                    area=new_area,
                    is_polygon_clipped=is_polygon_clipped,
                )
                cleaned_anns.append(new_ann)
                annotation_id_offset += 1

            if discard_augmentation:
                return False  # Discard the entire augmentation

        # In non-targeted mode, allow empty annotations
        if not cleaned_anns and not allow_empty_annotations:
            return False

        # In targeted mode, discard images that do not cause significant clipping
        if focus_category_ids and not self.significant_clipping_occurred_on_focus_categories:
            return False

        # Generate new filename
        filename, ext = os.path.splitext(os.path.basename(img.file_name))
        new_filename = f"{filename}_crop_aug{uuid.uuid4().hex}{ext}"
        output_image_path = os.path.join(
            self.config['output_images_dir'], new_filename
        )

        # Save augmented image
        save_success = save_image(padded_image, output_image_path)
        if not save_success:
            logging.error(
                f"Failed to save augmented image '{output_image_path}'."
            )
            return False

        # Create new image entry
        new_img = UnifiedImage(
            id=image_id_offset,
            file_name=output_image_path,
            width=padded_image.shape[1],
            height=padded_image.shape[0],
        )
        augmented_dataset.images.append(new_img)

        # Add cleaned annotations
        for new_ann in cleaned_anns:
            augmented_dataset.annotations.append(new_ann)

        # Visualization
        if (
            self.config['visualize_overlays']
            and self.config['output_visualizations_dir']
        ):
            visualization_filename = (
                f"{os.path.splitext(new_filename)[0]}_viz{ext}"
            )
            visualization_path = os.path.join(
                self.config['output_visualizations_dir'],
                visualization_filename,
            )
            mosaic_visualize_transformed_overlays(
                transformed_image=padded_image.copy(),
                cleaned_annotations=cleaned_anns,
                output_visualizations_dir=self.config[
                    'output_visualizations_dir'
                ],
                new_filename=visualization_filename,
                task=self.task,
            )

        logging.info(
            f"Crop augmented image '{new_filename}' saved with "
            f"{len(cleaned_anns)} annotations."
        )
        return True

    def compute_iou(self, rect1, rect2):
        """
        Compute Intersection over Union (IoU) between two rectangles.
        Each rectangle is defined as (x1, y1, w, h).
        """
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0  # No overlap

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        union_area = w1 * h1 + w2 * h2 - intersection_area

        return intersection_area / union_area
