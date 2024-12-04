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
from typing import Optional, List, Dict, Tuple
from augflow.utils.configs import translate_default_config

class TranslateAugmentation(Augmentation):
    def __init__(self, config=None, task: str = 'detection', modes: List[str] = None, focus_categories: Optional[List[str]] = None):
        super().__init__()
        self.config = translate_default_config.copy() 
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
        if not self.config.get('enable_translation', True):
            logging.info("Translation augmentation is disabled.")
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
            # Assign a default value if not specified, e.g., 0.8 (80%) for all categories
            self.max_clipped_area_per_category = {cat['id']: 0.8 for cat in dataset.categories}

        max_clipped_area_per_category = self.max_clipped_area_per_category

        # Create mapping from category names to IDs
        category_name_to_id = {cat['name']: cat['id'] for cat in dataset.categories}

        output_images_dir = self.config['output_images_dir']

        for img in dataset.images:
            image_path = img.file_name
            image = load_image(image_path)
            if image is None:
                logging.error(f"Failed to load image '{image_path}'. Skipping.")
                continue
            img_h, img_w = image.shape[:2]

            anns = image_id_to_annotations.get(img.id, [])

            # Initialize a set to track used translations for this image
            used_translations = set()

            if 'non_targeted' in self.modes:
                self.apply_non_targeted(
                    img, image, anns, img_w, img_h, image_id_offset, annotation_id_offset,
                    augmented_dataset, max_clipped_area_per_category, output_dim, used_translations
                )
                # Update offsets
                image_id_offset = max([img.id for img in augmented_dataset.images], default=image_id_offset) + 1
                annotation_id_offset = max([ann.id for ann in augmented_dataset.annotations], default=annotation_id_offset) + 1

            if 'targeted' in self.modes:
                self.apply_targeted(
                    img, image, anns, img_w, img_h, image_id_offset, annotation_id_offset,
                    augmented_dataset, max_clipped_area_per_category, output_dim, used_translations,
                    category_name_to_id
                )
                # Update offsets
                image_id_offset = max([img.id for img in augmented_dataset.images], default=image_id_offset) + 1
                annotation_id_offset = max([ann.id for ann in augmented_dataset.annotations], default=annotation_id_offset) + 1

        logging.info(f"Translation augmentation completed.")
        return augmented_dataset

    def apply_non_targeted(self, img, image, anns, img_w, img_h, image_id_offset, annotation_id_offset,
                           augmented_dataset, max_clipped_area_per_category, output_dim, used_translations):
        num_translations = self.config['num_translations_per_image']
        image_successful_aug = 0

        while image_successful_aug < num_translations:
            translation_applied = False
            # Decide whether to apply translation based on probability
            prob = self.config['translate_probability']
            if random.random() > prob:
                logging.info(f"Skipping translation augmentation {image_successful_aug+1} for image ID {img.id} based on probability ({prob}).")
                continue  # Skip this augmentation

            # Random translation parameters (percentages)
            min_tx_percent = self.config['min_translate_x']
            max_tx_percent = self.config['max_translate_x']
            min_ty_percent = self.config['min_translate_y']
            max_ty_percent = self.config['max_translate_y']

            # Generate possible translations within the ranges
            translate_x_percent = random.uniform(min_tx_percent, max_tx_percent)
            translate_y_percent = random.uniform(min_ty_percent, max_ty_percent)

            translation_key = (translate_x_percent, translate_y_percent)
            if translation_key in used_translations:
                continue  # Skip if we've already tried this translation
            used_translations.add(translation_key)

            translate_x = translate_x_percent * img_w
            translate_y = translate_y_percent * img_h

            success = self.try_translation(
                img, image, anns, img_w, img_h, translate_x, translate_y,
                image_id_offset, annotation_id_offset, augmented_dataset,
                max_clipped_area_per_category, output_dim, focus_category_ids=None
            )
            if success:
                image_id_offset += 1
                annotation_id_offset += len(anns)
                image_successful_aug += 1
                translation_applied = True
            else:
                break  # No valid translations left to try
            if not translation_applied:
                break  # No valid translations left to try

    def apply_targeted(self, img, image, anns, img_w, img_h, image_id_offset, annotation_id_offset,
                       augmented_dataset, max_clipped_area_per_category, output_dim, used_translations,
                       category_name_to_id):
        # For targeted mode, focus on specific categories
        if not self.focus_categories:
            logging.warning("No focus categories provided for targeted mode.")
            return
        focus_category_ids = [category_name_to_id[cat_name] for cat_name in self.focus_categories if cat_name in category_name_to_id]
        if not focus_category_ids:
            logging.warning("Focus categories do not match any categories in the dataset.")
            return

        num_translations = self.config['num_translations_per_image']

        # Identify the bounding boxes of focus category annotations
        focus_anns = [ann for ann in anns if ann.category_id in focus_category_ids]
        if not focus_anns:
            logging.info(f"No focus category annotations in image ID {img.id}. Skipping.")
            return

        image_successful_aug = 0

        while image_successful_aug < num_translations:
            # Find the translation that causes maximum acceptable clipping on focus categories
            best_translation = self.find_best_translation_targeted(
                anns, image, img_w, img_h,
                max_clipped_area_per_category, output_dim, focus_category_ids, used_translations
            )
            if best_translation is None:
                logging.info(f"Could not find suitable translation for image ID {img.id}.")
                break
            translate_x_percent, translate_y_percent = best_translation
            used_translations.add((translate_x_percent, translate_y_percent))
            translate_x = translate_x_percent * img_w
            translate_y = translate_y_percent * img_h
            # Try to apply translation
            success = self.try_translation(
                img, image, anns, img_w, img_h, translate_x, translate_y,
                image_id_offset, annotation_id_offset, augmented_dataset,
                max_clipped_area_per_category, output_dim, focus_category_ids=focus_category_ids
            )
            if success:
                # Check if significant clipping occurred on focus categories
                if self.significant_clipping_occurred_on_focus_categories:
                    image_id_offset += 1
                    annotation_id_offset += len(anns)
                    image_successful_aug += 1
                else:
                    # Discard image if no significant clipping occurred on focus categories
                    if augmented_dataset.images and augmented_dataset.images[-1].id == image_id_offset:
                        augmented_dataset.images.pop()
                    augmented_dataset.annotations = [ann for ann in augmented_dataset.annotations if ann.image_id != image_id_offset]
            else:
                break  # No valid translations left to try

        if image_successful_aug < num_translations:
            logging.info(f"Could not generate {num_translations} unique augmentations for image ID {img.id}. Generated {image_successful_aug} instead.")

    def find_best_translation_targeted(self, anns, image, img_w, img_h,
                                       max_clipped_area_per_category, output_dim, focus_category_ids, used_translations):
        """
        Find the translation parameters that cause maximum acceptable clipping on focus categories.
        """
        min_tx_percent = self.config['min_translate_x']
        max_tx_percent = self.config['max_translate_x']
        min_ty_percent = self.config['min_translate_y']
        max_ty_percent = self.config['max_translate_y']

        # Generate a grid of possible translations
        tx_values = np.linspace(min_tx_percent, max_tx_percent, num=10)
        ty_values = np.linspace(min_ty_percent, max_ty_percent, num=10)
        translations = [(tx, ty) for tx in tx_values for ty in ty_values]

        # Shuffle to introduce randomness
        random.shuffle(translations)

        best_translation = None
        max_clipping = 0

        for translate_x_percent, translate_y_percent in translations:
            if (translate_x_percent, translate_y_percent) in used_translations:
                continue
            translate_x = translate_x_percent * img_w
            translate_y = translate_y_percent * img_h

            M = generate_affine_transform_matrix(
                image_size=(img_w, img_h),
                rotation_deg=0,
                scale=(1.0, 1.0),
                shear_deg=0,
                translation=(translate_x, translate_y)
            )

            if output_dim:
                output_width, output_height = output_dim
            else:
                output_width, output_height = img_w, img_h

            # Apply transformation
            transformed_anns = transform_annotations(anns, M)

            total_clipping = 0
            discard_translation = False
            for ann, original_ann in zip(transformed_anns, anns):
                coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
                if not coords:
                    continue  # Skip if coordinates are invalid

                transformed_polygon = Polygon(coords)
                if not transformed_polygon.is_valid:
                    transformed_polygon = transformed_polygon.buffer(0)
                original_area = transformed_polygon.area

                # Define the image boundary
                image_boundary = box(0, 0, output_width, output_height)

                # Clip the polygon to the image boundary
                clipped_polygon = transformed_polygon.intersection(image_boundary)

                if clipped_polygon.is_empty:
                    clipped_area = 0
                else:
                    if not clipped_polygon.is_valid:
                        clipped_polygon = clipped_polygon.buffer(0)
                    clipped_area = clipped_polygon.area

                # Compute area reduction due to clipping
                area_reduction_due_to_clipping = calculate_area_reduction(original_area, clipped_area)

                category_id = original_ann.category_id
                max_allowed_reduction = max_clipped_area_per_category.get(category_id, 0.8)  # Default to 80%

                if category_id in focus_category_ids:
                    if area_reduction_due_to_clipping >= 1.0:
                        discard_translation = True
                        break  # Discard this translation
                    elif area_reduction_due_to_clipping > max_allowed_reduction:
                        discard_translation = True
                        break  # Discard this translation
                    else:
                        total_clipping += area_reduction_due_to_clipping

                else:
                    if area_reduction_due_to_clipping > max_allowed_reduction and area_reduction_due_to_clipping < 1.0:
                        discard_translation = True
                        break  # Discard this translation

            if discard_translation:
                continue

            if total_clipping > max_clipping and total_clipping > 0:
                max_clipping = total_clipping
                best_translation = (translate_x_percent, translate_y_percent)

                # If we've found a translation causing clipping just below max allowed, break
                if max_clipping >= max_allowed_reduction * 0.9 * len(focus_category_ids):
                    break

        return best_translation

    def try_translation(self, img, image, anns, img_w, img_h, translate_x, translate_y, image_id_offset, annotation_id_offset, augmented_dataset, max_clipped_area_per_category, output_dim, focus_category_ids):
        # Generate affine transformation matrix
        M = generate_affine_transform_matrix(
            image_size=(img_w, img_h),
            rotation_deg=0,
            scale=(1.0, 1.0),
            shear_deg=0,
            translation=(translate_x, translate_y)
        )

        if output_dim:
            output_width, output_height = output_dim
        else:
            output_width, output_height = img_w, img_h

        transformed_image = apply_affine_transform(image, M, (output_width, output_height))

        transformed_anns = transform_annotations(anns, M)

        # Check for significant clipping
        discard_image = False
        self.significant_clipping_occurred_on_focus_categories = False  # Flag to check if significant clipping occurred on focus categories
        cleaned_anns = []
        for ann, original_ann in zip(transformed_anns, anns):
            coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
            if not coords:
                continue  # Skip if coordinates are invalid

            transformed_polygon = Polygon(coords)
            if not transformed_polygon.is_valid:
                transformed_polygon = transformed_polygon.buffer(0)
            original_area = transformed_polygon.area

            # Define the image boundary
            image_boundary = box(0, 0, output_width, output_height)

            # Clip the polygon to the image boundary
            clipped_polygon = transformed_polygon.intersection(image_boundary)

            if clipped_polygon.is_empty:
                clipped_area = 0
            else:
                if not clipped_polygon.is_valid:
                    clipped_polygon = clipped_polygon.buffer(0)
                clipped_area = clipped_polygon.area

            # Compute area reduction due to clipping
            area_reduction_due_to_clipping = calculate_area_reduction(original_area, clipped_area)

            # Determine if polygon was clipped
            is_polygon_clipped = area_reduction_due_to_clipping > 0.01

            category_id = original_ann.category_id
            max_allowed_reduction = max_clipped_area_per_category.get(category_id, 0.8)  # Default to 80%

            if focus_category_ids and category_id in focus_category_ids:
                # For focus categories, they should not be completely clipped
                if area_reduction_due_to_clipping >= 1.0:
                    discard_image = True
                    break  # Discard this image
                elif area_reduction_due_to_clipping > max_allowed_reduction:
                    discard_image = True
                    break  # Discard this image
                else:
                    # Check if significant clipping occurred (e.g., more than 50% of max allowed)
                    if area_reduction_due_to_clipping >= max_allowed_reduction * 0.5:
                        self.significant_clipping_occurred_on_focus_categories = True
            else:
                # For non-focus categories
                if area_reduction_due_to_clipping > max_allowed_reduction and area_reduction_due_to_clipping < 1.0:
                    discard_image = True
                    break  # Discard this image
                # Area reduction is acceptable if it's less than or equal to max_allowed_reduction or exactly 100%
                # If area_reduction_due_to_clipping == 1.0 (completely clipped), it's acceptable

            if clipped_area == 0:
                continue  # Skip annotations that are completely outside the image

            # Handle MultiPolygon cases
            polygons_to_process = []
            if isinstance(clipped_polygon, Polygon):
                polygons_to_process.append(clipped_polygon)
            elif isinstance(clipped_polygon, MultiPolygon):
                polygons_to_process.extend(clipped_polygon.geoms)
            else:
                continue

            # Collect cleaned polygon coordinates
            for poly in polygons_to_process:
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if self.task == 'detection':
                    coords = ensure_axis_aligned_rectangle(list(poly.exterior.coords))
                    if not coords:
                        continue
                else:
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
                        continue

                # Update the annotation
                new_ann = UnifiedAnnotation(
                    id=annotation_id_offset,
                    image_id=image_id_offset,
                    category_id=category_id,
                    polygon=coords,
                    iscrowd=original_ann.iscrowd,
                    area=clipped_area,
                    is_polygon_clipped=is_polygon_clipped,
                )

                cleaned_anns.append(new_ann)
                annotation_id_offset += 1

        if discard_image or not cleaned_anns:
            return False

        # In targeted mode, discard images that do not cause significant clipping on focus categories
        if focus_category_ids and not self.significant_clipping_occurred_on_focus_categories:
            return False

        # Generate new filename
        new_filename = f"{os.path.splitext(os.path.basename(img.file_name))[0]}_translate_{uuid.uuid4().hex}.jpg"
        output_image_path = os.path.join(self.config['output_images_dir'], new_filename)

        # Save transformed image
        save_success = save_image(transformed_image, output_image_path)
        if not save_success:
            return False

        # Create new image entry
        new_img = UnifiedImage(
            id=image_id_offset,
            file_name=output_image_path,
            width=output_width,
            height=output_height
        )
        augmented_dataset.images.append(new_img)

        # Process and save transformed annotations
        for ann in cleaned_anns:
            augmented_dataset.annotations.append(ann)
            logging.info(f"Added annotation ID {ann.id} for image ID {image_id_offset}.")

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

        logging.info(f"Translation augmented image '{new_filename}' saved with {len(cleaned_anns)} annotations.")

        return True
