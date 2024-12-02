# augflow/augmentations/crop.py

import os
import cv2
import numpy as np
import random
import copy
from shapely.geometry import Polygon, MultiPolygon, box
import logging

# Import base class
from .base import Augmentation

# Import helper functions from utils
from augflow.utils.images import load_image, save_image, mosaic_visualize_transformed_overlays
from augflow.utils.unified_format import UnifiedDataset, UnifiedImage, UnifiedAnnotation
import uuid
from typing import Optional, List, Dict
from augflow.utils.annotations import ensure_axis_aligned_rectangle, calculate_iou, calculate_area_reduction
from augflow.utils.configs import crop_default_config


class CropAugmentation(Augmentation):
    def __init__(self, config=None, task: str = 'detection', modes: List[str] = None, focus_categories: Optional[List[str]] = None):
        super().__init__()

        self.config = crop_default_config.copy()
        if config:
            self.config.update(config)
        self.task = task.lower()
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
            # Assign a default value if not specified, e.g., 0.5 (50%) for all categories
            self.max_clipped_area_per_category = {cat['id']: 0.5 for cat in dataset.categories}

        max_clipped_area_per_category = self.max_clipped_area_per_category

        # Create mapping from category names to IDs
        category_name_to_id = {cat['name']: cat['id'] for cat in dataset.categories}

        for img in dataset.images:
            image_path = img.file_name
            image = load_image(image_path)
            if image is None:
                logging.error(f"Failed to load image '{image_path}'. Skipping.")
                continue
            img_h, img_w = image.shape[:2]

            anns = image_id_to_annotations.get(img.id, [])

            if 'targeted' in self.modes:
                self.apply_targeted(
                    img, image, anns, img_w, img_h, image_id_offset, annotation_id_offset,
                    augmented_dataset, max_clipped_area_per_category, category_name_to_id
                )
                # Update offsets
                image_id_offset = max([img.id for img in augmented_dataset.images], default=image_id_offset) + 1
                annotation_id_offset = max([ann.id for ann in augmented_dataset.annotations], default=annotation_id_offset) + 1

            if 'non_targeted' in self.modes:
                # Implement non-targeted mode if needed
                pass  # Currently focusing on targeted mode as per your requirement

        logging.info(f"Cropping augmentation completed.")
        return augmented_dataset

    def apply_targeted(self, img, image, anns, img_w, img_h, image_id_offset, annotation_id_offset,
                       augmented_dataset, max_clipped_area_per_category, category_name_to_id):
        if not self.focus_categories:
            logging.warning("No focus categories provided for targeted mode.")
            return

        focus_category_ids = [category_name_to_id[cat_name] for cat_name in self.focus_categories if cat_name in category_name_to_id]
        if not focus_category_ids:
            logging.warning("Focus categories do not match any categories in the dataset.")
            return

        # Find annotations belonging to focus categories
        focus_anns = [ann for ann in anns if ann.category_id in focus_category_ids]
        if not focus_anns:
            logging.info(f"No focus category annotations in image ID {img.id}. Skipping targeted cropping.")
            return

        # Sort focus annotations based on the order of focus_categories
        focus_anns.sort(key=lambda ann: self.focus_categories.index(
            [cat_name for cat_name, cat_id in category_name_to_id.items() if cat_id == ann.category_id][0]
        ))

        num_crops = self.config['num_crops_per_image']
        image_successful_aug = 0

        for ann in focus_anns:
            if image_successful_aug >= num_crops:
                break

            # Calculate significant clipping just below max allowed
            max_allowed_reduction = max_clipped_area_per_category.get(ann.category_id, 0.5)  # Default to 50%
            target_area_reduction = max_allowed_reduction * 0.9  # 90% of max allowed

            # Compute the original polygon
            coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
            if not coords:
                continue
            polygon = Polygon(coords)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            original_area = polygon.area

            # Start with the bounding box of the polygon
            minx, miny, maxx, maxy = polygon.bounds

            # Initialize crop box as the bounding box of the polygon
            x_left = minx
            x_right = maxx
            y_top = miny
            y_bottom = maxy

            # Expand the crop box to the maximum possible size while maintaining the area reduction constraint
            expansion_step = max(img_w, img_h) // 100  # Define an expansion step

            success = False
            best_crop_box = None
            largest_area = 0

            for expand in range(0, max(img_w, img_h), expansion_step):
                # Expand in all directions
                x1 = max(0, x_left - expand)
                x2 = min(img_w, x_right + expand)
                y1 = max(0, y_top - expand)
                y2 = min(img_h, y_bottom + expand)

                # Adjust to make the crop box square if possible
                w = x2 - x1
                h = y2 - y1
                if self.config.get('prefer_square', False):
                    if w != h:
                        size = max(w, h)
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        x1 = max(0, x_center - size / 2)
                        x2 = min(img_w, x_center + size / 2)
                        y1 = max(0, y_center - size / 2)
                        y2 = min(img_h, y_center + size / 2)
                        w = x2 - x1
                        h = y2 - y1

                crop_box = box(x1, y1, x2, y2)
                cropped_polygon = polygon.intersection(crop_box)
                if cropped_polygon.is_empty or not cropped_polygon.is_valid:
                    continue
                clipped_area = cropped_polygon.area
                area_reduction = calculate_area_reduction(original_area, clipped_area)
                if area_reduction >= target_area_reduction and area_reduction < max_allowed_reduction:
                    crop_area = w * h
                    if crop_area > largest_area:
                        largest_area = crop_area
                        best_crop_box = (int(x1), int(y1), int(w), int(h))
                        best_area_reduction = area_reduction
                        success = True
                elif area_reduction >= max_allowed_reduction:
                    break  # Exceeds max allowed reduction, stop expanding

            if not success or best_crop_box is None:
                logging.info(f"Could not find suitable crop for annotation ID {ann.id} in image ID {img.id}.")
                continue

            # Crop the image using the best crop box
            x, y, w, h = best_crop_box
            cropped_image = image[y:y+h, x:x+w]

            # Adjust annotations
            new_annotations = []
            for ann_orig in anns:
                coords = list(zip(ann_orig.polygon[0::2], ann_orig.polygon[1::2]))
                if not coords:
                    continue
                polygon_orig = Polygon(coords)
                if not polygon_orig.is_valid:
                    polygon_orig = polygon_orig.buffer(0)
                adjusted_coords = []
                for px, py in coords:
                    new_px = px - x
                    new_py = py - y
                    adjusted_coords.append((new_px, new_py))
                adjusted_polygon = Polygon(adjusted_coords)
                if not adjusted_polygon.is_valid:
                    adjusted_polygon = adjusted_polygon.buffer(0)
                # Clip adjusted polygon to crop boundary
                crop_boundary = box(0, 0, w, h)
                clipped_polygon = adjusted_polygon.intersection(crop_boundary)
                if clipped_polygon.is_empty:
                    continue  # Polygon is completely outside; exclude it
                if not clipped_polygon.is_valid:
                    clipped_polygon = clipped_polygon.buffer(0)
                clipped_area = clipped_polygon.area
                original_area_ann = polygon_orig.area
                area_reduction_ann = calculate_area_reduction(original_area_ann, clipped_area)
                # Check if area reduction exceeds max allowed
                category_id = ann_orig.category_id
                max_allowed_reduction_ann = max_clipped_area_per_category.get(category_id, 0.5)
                if area_reduction_ann > max_allowed_reduction_ann:
                    continue  # Exclude annotations that are too clipped

                # Handle MultiPolygon cases
                polygons_to_process = []
                if isinstance(clipped_polygon, Polygon):
                    polygons_to_process.append(clipped_polygon)
                elif isinstance(clipped_polygon, MultiPolygon):
                    polygons_to_process.extend(clipped_polygon.geoms)
                else:
                    continue

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
                    continue

                new_ann = UnifiedAnnotation(
                    id=annotation_id_offset,
                    image_id=image_id_offset,
                    category_id=ann_orig.category_id,
                    polygon=[coord for point in cleaned_polygon_coords for coord in point],
                    iscrowd=ann_orig.iscrowd,
                    area=clipped_area,
                    is_polygon_clipped=area_reduction_ann > 0.0,
                    area_reduction_due_to_clipping=area_reduction_ann,
                )
                annotation_id_offset += 1
                new_annotations.append(new_ann)

            if not new_annotations:
                logging.info(f"No annotations left after cropping for image ID {img.id}. Skipping.")
                continue

            # Generate unique filename
            filename, ext = os.path.splitext(os.path.basename(img.file_name))
            new_filename = f"{filename}_crop_{uuid.uuid4().hex}{ext}"
            output_image_path = os.path.join(self.config['output_images_dir'], new_filename)

            # Save cropped image
            save_success = save_image(cropped_image, output_image_path)
            if not save_success:
                logging.error(f"Failed to save cropped image '{output_image_path}'. Skipping this augmentation.")
                continue

            # Create new image entry
            new_img = UnifiedImage(
                id=image_id_offset,
                file_name=output_image_path,
                width=cropped_image.shape[1],
                height=cropped_image.shape[0]
            )
            augmented_dataset.images.append(new_img)
            augmented_dataset.annotations.extend(new_annotations)

            # Visualization
            if self.config['visualize_overlays'] and self.config['output_visualizations_dir']:
                os.makedirs(self.config['output_visualizations_dir'], exist_ok=True)
                visualization_filename = f"{os.path.splitext(new_filename)[0]}_viz.jpg"
                mosaic_visualize_transformed_overlays(
                    transformed_image=cropped_image.copy(),
                    cleaned_annotations=new_annotations,
                    output_visualizations_dir=self.config['output_visualizations_dir'],
                    new_filename=visualization_filename,
                    task=self.task
                )

            logging.info(f"Cropped image '{new_filename}' saved with annotations.")
            image_id_offset += 1
            image_successful_aug += 1




    def apply_non_targeted(self, img, image, anns, img_w, img_h, image_id_offset, annotation_id_offset,
                           augmented_dataset, max_clipped_area_per_category, existing_crops):
        num_crops = self.config['num_crops_per_image']

        crops = []

        # Generate crops based on crop_modes
        for crop_mode in self.config['crop_modes']:
            params = self.config['crop_size_parameters'].get(crop_mode, {})
            if crop_mode == 'fixed_size':
                w = params.get('crop_width', 800)
                h = params.get('crop_height', 800)
                if w <= 0 or h <= 0:
                    logging.warning(f"Invalid crop size for 'fixed_size': width={w}, height={h}. Skipping this mode.")
                    continue
                # Randomly position the fixed-size crop
                if img_w > w:
                    x = random.randint(0, img_w - w)
                else:
                    x = 0
                if img_h > h:
                    y = random.randint(0, img_h - h)
                else:
                    y = 0
                crops.append((int(x), int(y), int(w), int(h)))
            elif crop_mode == 'random_area':
                min_area_ratio = params.get('min_area_ratio', 0.5)
                max_area_ratio = params.get('max_area_ratio', 0.9)
                if not (0 < min_area_ratio <= max_area_ratio < 1):
                    logging.warning(f"Invalid area ratios for 'random_area': min={min_area_ratio}, max={max_area_ratio}. Skipping this mode.")
                    continue
                area_ratio = random.uniform(min_area_ratio, max_area_ratio)
                crop_area = area_ratio * img_w * img_h
                target_ratio = random.uniform(0.5, 2.0)
                w = int(np.sqrt(crop_area * target_ratio))
                h = int(np.sqrt(crop_area / target_ratio))
                w = min(w, img_w)
                h = min(h, img_h)
                if img_w > w:
                    x = random.randint(0, img_w - w)
                else:
                    x = 0
                if img_h > h:
                    y = random.randint(0, img_h - h)
                else:
                    y = 0
                crops.append((int(x), int(y), int(w), int(h)))
            else:
                logging.warning(f"Unsupported crop mode '{crop_mode}' in non-targeted mode. Skipping.")
                continue

        if not crops:
            logging.warning(f"No valid crops generated for image ID {img.id} in non-targeted mode. Skipping this image.")
            return

        # Shuffle crops to introduce randomness
        random.shuffle(crops)

        # Limit the number of crops
        crops = crops[:num_crops]

        image_successful_aug = 0

        for crop_coords in crops:
            x, y, w, h = crop_coords

            # Check minimum crop size
            min_crop_size = self.config.get('min_crop_size', 256)
            if w < min_crop_size or h < min_crop_size:
                logging.info(f"Crop size ({w}, {h}) is smaller than min_crop_size ({min_crop_size}). Skipping this crop.")
                continue  # Skip this crop

            # Check for overlap with existing crops
            new_crop_bbox = [x, y, w, h]
            overlap = False
            for existing_crop in existing_crops:
                existing_bbox = existing_crop
                iou = calculate_iou(new_crop_bbox, existing_bbox)
                if iou > self.config['overlap_parameters']['max_overlap']:
                    overlap = True
                    logging.info(f"New crop {new_crop_bbox} overlaps with existing crop {existing_bbox} (IoU={iou:.2f}) exceeding max_overlap={self.config['overlap_parameters']['max_overlap']}. Skipping this crop.")
                    break
            if overlap:
                continue  # Skip this crop due to overlap

            # Perform the cropping
            success = self.perform_cropping(
                img, image, anns, img_w, img_h, x, y, w, h,
                image_id_offset, annotation_id_offset, augmented_dataset,
                max_clipped_area_per_category, existing_crops
            )
            if success:
                image_id_offset += 1
                annotation_id_offset += len(anns)
                image_successful_aug += 1

    def perform_cropping(self, img, image, anns, img_w, img_h, x, y, w, h,
                         image_id_offset, annotation_id_offset, augmented_dataset,
                         max_clipped_area_per_category, existing_crops):

        # Crop the image directly without padding
        cropped_image = image[y:y + h, x:x + w]

        # No scaling factors since we are not resizing
        scale_x, scale_y = 1.0, 1.0
        pad_left_total, pad_top_total = 0, 0  # No padding

        # Process annotations for this image
        discard_augmentation = False  # Flag to decide whether to discard augmentation

        # List to hold cropped annotations temporarily
        temp_cropped_annotations = []

        for ann in anns:
            # Original coordinates
            coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
            if not coords:
                continue  # Skip if coordinates are invalid

            original_polygon = Polygon(coords)
            if not original_polygon.is_valid:
                original_polygon = original_polygon.buffer(0)
            original_area = original_polygon.area

            # Adjust coordinates based on crop
            adjusted_coords = []
            for px, py in coords:
                new_px = px - x + pad_left_total
                new_py = py - y + pad_top_total
                adjusted_coords.append((new_px, new_py))

            adjusted_polygon = Polygon(adjusted_coords)
            if not adjusted_polygon.is_valid:
                adjusted_polygon = adjusted_polygon.buffer(0)
            adjusted_area = adjusted_polygon.area

            # Compute area reduction due to scaling (none in this case)
            area_reduction_due_to_scaling = 0.0

            # Define the crop boundary (image boundary)
            crop_boundary = box(0, 0, cropped_image.shape[1], cropped_image.shape[0])

            # Clip the adjusted polygon to the crop boundary
            clipped_polygon = adjusted_polygon.intersection(crop_boundary)

            if clipped_polygon.is_empty:
                continue  # Polygon is completely outside; exclude it

            if not clipped_polygon.is_valid:
                clipped_polygon = clipped_polygon.buffer(0)
            clipped_area = clipped_polygon.area

            # Compute area reduction due to clipping
            if adjusted_area > 0:
                area_reduction_due_to_clipping = max(0.0, (adjusted_area - clipped_area) / adjusted_area)
            else:
                area_reduction_due_to_clipping = 0.0

            # Total area reduction
            if original_area > 0:
                total_area_reduction = max(0.0, (original_area - clipped_area) / original_area)
            else:
                total_area_reduction = 0.0

            # Check if area reduction exceeds the threshold
            category_id = ann.category_id
            max_allowed_reduction = max_clipped_area_per_category.get(category_id, 0.5)  # Default to 50%

            if total_area_reduction > max_allowed_reduction:
                discard_augmentation = True
                logging.warning(f"Crop for image ID {img.id} discarded due to total area reduction ({total_area_reduction:.6f}) exceeding threshold ({max_allowed_reduction}) for category {category_id}.")
                break  # Discard the entire augmentation

            # Determine if polygon was clipped
            is_polygon_clipped = area_reduction_due_to_clipping > 0.01

            # Handle MultiPolygon cases
            polygons_to_process = []
            if isinstance(clipped_polygon, Polygon):
                polygons_to_process.append(clipped_polygon)
            elif isinstance(clipped_polygon, MultiPolygon):
                polygons_to_process.extend(clipped_polygon.geoms)
            else:
                logging.warning(f"Unknown geometry type for clipped polygon: {type(clipped_polygon)}")
                continue

            # For detection task, we only need the bounding box of the polygon(s)
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
                logging.debug(f"No valid coordinates found after processing clipped polygons. Skipping.")
                continue

            # Assign area reductions and flags
            cleaned_ann = UnifiedAnnotation(
                id=annotation_id_offset,
                image_id=image_id_offset,
                category_id=ann.category_id,
                polygon=[coord for point in cleaned_polygon_coords for coord in point],
                iscrowd=ann.iscrowd,
                area=clipped_area,
                is_polygon_scaled=False,  # No scaling occurred
                is_polygon_clipped=is_polygon_clipped,
                area_reduction_due_to_scaling=area_reduction_due_to_scaling,
                area_reduction_due_to_clipping=area_reduction_due_to_clipping
            )

            temp_cropped_annotations.append(cleaned_ann)
            annotation_id_offset += 1

        if discard_augmentation:
            logging.info(f"Cropping augmentation for image ID {img.id} discarded due to high area reduction.")
            return False  # Skip this augmentation

        # If no polygons remain after excluding those completely outside, skip augmentation
        if not temp_cropped_annotations:
            logging.info(f"Cropping augmentation results in all polygons being completely outside. Skipping augmentation.")
            return False

        # Generate unique filename using UUID to prevent collisions
        filename, ext = os.path.splitext(os.path.basename(img.file_name))
        new_filename = f"{filename}_crop_{uuid.uuid4().hex}{ext}"
        output_image_path = os.path.join(self.config['output_images_dir'], new_filename)

        # Save cropped image
        save_success = save_image(cropped_image, output_image_path)
        if not save_success:
            logging.error(f"Failed to save cropped image '{output_image_path}'. Skipping this augmentation.")
            return False

        # Add the new crop's bounding box to existing_crops for overlap control
        existing_crops.append([x, y, w, h])

        # Create new image entry
        new_img = UnifiedImage(
            id=image_id_offset,
            file_name=output_image_path,
            width=cropped_image.shape[1],
            height=cropped_image.shape[0]
        )
        augmented_dataset.images.append(new_img)

        # Process and save cropped annotations
        for cropped_ann in temp_cropped_annotations:
            augmented_dataset.annotations.append(cropped_ann)
            logging.info(f"Added annotation ID {cropped_ann.id} for image ID {image_id_offset}.")

        # Visualization
        if self.config['visualize_overlays'] and self.config['output_visualizations_dir']:
            os.makedirs(self.config['output_visualizations_dir'], exist_ok=True)
            visualization_filename = f"{os.path.splitext(new_filename)[0]}_viz.jpg"
            mosaic_visualize_transformed_overlays(
                transformed_image=cropped_image.copy(),
                cleaned_annotations=temp_cropped_annotations,
                output_visualizations_dir=self.config['output_visualizations_dir'],
                new_filename=visualization_filename,
                task=self.task  # Pass the task ('detection' or 'segmentation')
            )

        logging.info(f"Cropped image '{new_filename}' saved with {len(temp_cropped_annotations)} annotations.")
        return True
