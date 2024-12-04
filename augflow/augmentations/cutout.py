# cutout.py
import copy
import os
import random
import uuid
import logging
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box
from typing import List, Optional

# Assume necessary imports from your project structure
from augflow.pipeline import Pipeline
from augflow.utils.configs import cutout_default_config
from augflow.utils.dataset import UnifiedAnnotation, UnifiedImage, UnifiedDataset
from augflow.utils.image_utils import load_image, save_image, calculate_area_reduction
from augflow.utils.geometry_utils import ensure_axis_aligned_rectangle
from augflow.utils.visualization import mosaic_visualize_transformed_overlays

class CutoutAugmentor:
    def __init__(self, config, task):
        self.config = config
        self.task = task
        self.modes = self.config.get('modes', ['targeted'])
        self.focus_categories = self.config.get('focus_categories', [])
        self.cutout_probability = self.config.get('cutout_probability', 1.0)
        self.num_augmented_images = self.config.get('num_augmented_images', 1)
        self.num_cutouts_per_image = self.config.get('num_cutouts_per_image', 1)
        self.cutout_size_percent = self.config.get('cutout_size_percent', ((0.1, 0.2), (0.1, 0.2)))
        self.margin_percent = self.config.get('margin_percent', 0.05)
        self.max_shift_percent = self.config.get('max_shift_percent', 1.0)
        self.shift_steps = self.config.get('shift_steps', 20)
        self.max_clipped_area_per_category = self.config.get('max_clipped_area_per_category', {})
        self.random_seed = self.config.get('random_seed', None)
        self.enable_cutout = self.config.get('enable_cutout', True)
        self.visualize_overlays = self.config.get('visualize_overlays', False)
        self.output_visualizations_dir = self.config.get('output_visualizations_dir', None)
        self.output_images_dir = self.config.get('output_images_dir', None)
        self.allowed_shifts = self.config.get('allowed_shifts', ['up', 'down', 'left', 'right'])
        self.area_reduction_threshold = self.config.get('area_reduction_threshold', 0.1)

        if self.random_seed is not None:
            random.seed(self.random_seed)

    def augment(self, dataset: UnifiedDataset) -> UnifiedDataset:
        if not self.enable_cutout:
            logging.info("Cutout augmentation is disabled in the configuration.")
            return dataset

        augmented_dataset = UnifiedDataset()
        augmented_dataset.categories = dataset.categories.copy()

        category_name_to_id = {cat.name: cat.id for cat in dataset.categories}
        logging.debug(f"Category name to ID mapping: {category_name_to_id}")

        max_clipped_area_per_category = self.max_clipped_area_per_category
        if not max_clipped_area_per_category:
            # Set default maximum area reduction per category to 50%
            max_clipped_area_per_category = {cat.id: 0.5 for cat in dataset.categories}

        image_id_offset = max([img.id for img in dataset.images], default=0) + 1
        annotation_id_offset = max([ann.id for ann in dataset.annotations], default=0) + 1

        for img in dataset.images:
            anns = [ann for ann in dataset.annotations if ann.image_id == img.id]
            image = load_image(img.file_name)
            if image is None:
                logging.warning(f"Image '{img.file_name}' could not be loaded. Skipping.")
                continue
            img_h, img_w = image.shape[:2]
            output_dim = (img_w, img_h)
            used_shifts = set()

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

    def apply_targeted(self, img, image, anns, img_w, img_h, image_id_offset, annotation_id_offset,
                       augmented_dataset, max_clipped_area_per_category, output_dim, used_shifts,
                       category_name_to_id):
        if not self.focus_categories:
            logging.warning("No focus categories provided for targeted mode.")
            return
        focus_category_ids = [category_name_to_id[cat_name] for cat_name in self.focus_categories if cat_name in category_name_to_id]
        if not focus_category_ids:
            logging.warning("Focus categories do not match any categories in the dataset.")
            return
        logging.debug(f"Focus category IDs: {focus_category_ids}")

        image_successful_aug = 0
        max_attempts = self.num_augmented_images * 5  # To prevent infinite loops

        attempts = 0
        while image_successful_aug < self.num_augmented_images and attempts < max_attempts:
            attempts += 1
            # Create cutouts for each focus annotation
            cutouts = []
            for ann in anns:
                if ann.category_id not in focus_category_ids:
                    continue  # Only process focus categories
                coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
                if not coords:
                    continue  # Skip if coordinates are invalid
                ann_poly = Polygon(coords)
                if not ann_poly.is_valid:
                    ann_poly = ann_poly.buffer(0)
                minx, miny, maxx, maxy = ann_poly.bounds

                # Add safety margin
                margin = self.margin_percent
                x1 = max(int(minx - margin * img_w), 0)
                y1 = max(int(miny - margin * img_h), 0)
                x2 = min(int(maxx + margin * img_w), img_w)
                y2 = min(int(maxy + margin * img_h), img_h)

                cutout = {
                    'ann': ann,
                    'bbox': (x1, y1, x2, y2),
                    'category_id': ann.category_id,
                    'is_focus': True,
                    'shifted_bbox': None  # To store the final shifted bbox
                }
                cutouts.append(cutout)

            if not cutouts:
                logging.info(f"No focus annotations to process in image ID {img.id}. Skipping.")
                break

            # Randomly select one focus cutout to fully mask out
            focus_cutout = random.choice(cutouts)
            focus_cutout['action'] = 'full_mask'

            # Randomly select another focus cutout to partially mask (clip)
            other_focus_cutouts = [c for c in cutouts if c != focus_cutout]
            if other_focus_cutouts:
                partial_cutout = random.choice(other_focus_cutouts)
                partial_cutout['action'] = 'partial_mask'
            else:
                logging.info(f"Not enough focus annotations to perform both full and partial masking in image ID {img.id}.")
                continue

            # Prepare masks
            masks = []
            augmented_image = image.copy()

            # Apply full mask to focus_cutout
            x1, y1, x2, y2 = focus_cutout['bbox']
            augmented_image[y1:y2, x1:x2] = 0
            masks.append((x1, y1, x2, y2))

            # Apply partial mask to partial_cutout by shifting its bbox randomly
            shift_x = random.randint(-int(self.max_shift_percent * img_w), int(self.max_shift_percent * img_w))
            shift_y = random.randint(-int(self.max_shift_percent * img_h), int(self.max_shift_percent * img_h))
            x1_p, y1_p, x2_p, y2_p = partial_cutout['bbox']
            x1_p_shifted = max(0, min(x1_p + shift_x, img_w))
            y1_p_shifted = max(0, min(y1_p + shift_y, img_h))
            x2_p_shifted = max(0, min(x2_p + shift_x, img_w))
            y2_p_shifted = max(0, min(y2_p + shift_y, img_h))

            augmented_image[int(y1_p_shifted):int(y2_p_shifted), int(x1_p_shifted):int(x2_p_shifted)] = 0
            masks.append((x1_p_shifted, y1_p_shifted, x2_p_shifted, y2_p_shifted))

            # Now, process annotations and decide whether to keep the augmented image
            success = self.process_cutout(
                img, augmented_image, anns, masks, img_w, img_h,
                image_id_offset, annotation_id_offset, augmented_dataset,
                max_clipped_area_per_category, output_dim, focus_category_ids=focus_category_ids,
                focus_cutout_id=focus_cutout['ann'].id,
                partial_cutout_id=partial_cutout['ann'].id
            )
            if success:
                image_id_offset += 1
                annotation_id_offset += len(anns)
                image_successful_aug += 1
            else:
                logging.info(f"Cutout augmentation for image ID {img.id} discarded during processing.")
                continue  # Try another attempt

        if image_successful_aug < self.num_augmented_images:
            logging.info(f"Could not generate {self.num_augmented_images} unique augmentations for image ID {img.id}. Generated {image_successful_aug} instead.")

    def process_cutout(self, img, augmented_image, anns, masks, img_w, img_h, image_id_offset, annotation_id_offset, augmented_dataset, max_clipped_area_per_category, output_dim, focus_category_ids, focus_cutout_id, partial_cutout_id):
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

        focus_polygons_clipped = 0

        for ann in transformed_annotations:
            category_id = ann.category_id
            max_allowed_reduction = max_clipped_area_per_category.get(category_id, 0.5)  # Default to 50% reduction
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
                new_area = 0
                area_reduction_due_to_clipping = 1.0
            else:
                new_area = clipped_poly.area
                area_reduction_due_to_clipping = calculate_area_reduction(original_area, new_area)

            is_focus_category = category_id in focus_category_ids
            is_focus_cutout = ann.id == focus_cutout_id
            is_partial_cutout = ann.id == partial_cutout_id

            if is_focus_category:
                if is_focus_cutout:
                    # Focus polygon that is fully masked out (acceptable)
                    if area_reduction_due_to_clipping < 1.0:
                        # Should be fully masked out
                        logging.info(f"Focus annotation ID {ann.id} in image ID {img.id} was not fully masked out as expected.")
                        discard_augmentation = True
                        break
                elif is_partial_cutout:
                    # Focus polygon that should be partially masked
                    if area_reduction_due_to_clipping == 1.0:
                        # Should not be fully masked out
                        logging.info(f"Focus annotation ID {ann.id} in image ID {img.id} was fully masked out, expected partial masking.")
                        discard_augmentation = True
                        break
                    elif area_reduction_due_to_clipping == 0.0:
                        # Not clipped at all
                        logging.info(f"Focus annotation ID {ann.id} in image ID {img.id} was not clipped at all, expected partial masking.")
                        discard_augmentation = True
                        break
                    else:
                        focus_polygons_clipped += 1
                else:
                    # Other focus polygons should remain unaffected
                    if area_reduction_due_to_clipping > 0.0:
                        logging.info(f"Focus annotation ID {ann.id} in image ID {img.id} was unexpectedly clipped.")
                        discard_augmentation = True
                        break
            else:
                # Non-focus polygons can be fully masked out or clipped up to the maximum allowed area reduction
                if area_reduction_due_to_clipping > max_allowed_reduction:
                    logging.info(f"Non-focus annotation ID {ann.id} in image ID {img.id} exceeds max allowed area reduction.")
                    discard_augmentation = True
                    break

            is_polygon_clipped = area_reduction_due_to_clipping > 0.01

            if clipped_poly.is_empty:
                continue  # Skip empty geometries

            if isinstance(clipped_poly, Polygon):
                polygons_to_process = [clipped_poly]
            elif isinstance(clipped_poly, MultiPolygon):
                polygons_to_process = list(clipped_poly.geoms)
            else:
                logging.warning(f"Unsupported geometry type {type(clipped_poly)} for annotation ID {ann.id} in image ID {img.id}. Skipping annotation.")
                continue  # Unsupported geometry type

            for poly in polygons_to_process:
                if not poly.is_valid or poly.is_empty:
                    continue  # Skip invalid or empty geometries

                new_area = poly.area
                area_reduction_due_to_clipping = calculate_area_reduction(original_area, new_area)
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
                logging.info(f"Cutout augmentation for image ID {img.id} discarded due to improper clipping.")
                return False  # Discard the entire augmentation

        # Acceptance Criteria
        if focus_polygons_clipped == 0:
            logging.info(f"No focus polygons were clipped as expected in image ID {img.id}. Skipping augmentation.")
            return False

        # If no polygons remain after masking, skip augmentation
        if not cleaned_anns:
            logging.info(f"Cutout augmentation for image ID {img.id} results in all polygons being fully masked. Skipping augmentation.")
            return False

        # Generate new filename
        filename, ext = os.path.splitext(os.path.basename(img.file_name))
        new_filename = f"{filename}_cutout_aug{uuid.uuid4().hex}{ext}"
        output_image_path = os.path.join(self.output_images_dir, new_filename)

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
        if self.visualize_overlays and self.output_visualizations_dir:
            visualization_filename = f"{os.path.splitext(new_filename)[0]}_viz{ext}"
            mosaic_visualize_transformed_overlays(
                transformed_image=augmented_image.copy(),
                cleaned_annotations=cleaned_anns,
                output_visualizations_dir=self.output_visualizations_dir,
                new_filename=visualization_filename,
                task=self.task
            )

        logging.info(f"Cutout augmented image '{new_filename}' saved with {len(cleaned_anns)} annotations.")
        return True
