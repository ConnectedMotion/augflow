# # augflow/augmentations/cutout.py

# import os
# import copy
# import random
# import logging
# import cv2
# import numpy as np
# from shapely.geometry import box, Polygon, MultiPolygon
# from .base import Augmentation
# from augflow.utils.images import load_image, save_image, mosaic_visualize_transformed_overlays
# from augflow.utils.annotations import (
#     calculate_area_reduction,
#     ensure_axis_aligned_rectangle
# )
# from augflow.utils.unified_format import UnifiedDataset, UnifiedImage, UnifiedAnnotation
# import uuid
# from typing import Optional, List, Dict
# from augflow.utils.configs import cutout_default_config

# class CutoutAugmentation(Augmentation):
#     def __init__(self, config=None, task: str = 'detection'):
#         super().__init__()
#         self.task = task.lower()
#         self.config = cutout_default_config.copy() 
#         if config:
#             self.config.update(config)
#         random.seed(self.config.get('random_seed', 42))
#         np.random.seed(self.config.get('random_seed', 42))

#         # Ensure output directories exist
#         os.makedirs(self.config['output_images_dir'], exist_ok=True)
#         if self.config.get('visualize_overlays') and self.config.get('output_visualizations_dir'):
#             os.makedirs(self.config['output_visualizations_dir'], exist_ok=True)

#         # Set max_clipped_area_per_category to default if not provided
#         if not self.config.get('max_clipped_area_per_category'):
#             # Will be set in apply() based on dataset categories
#             self.config['max_clipped_area_per_category'] = {}

    

#     def apply(self, dataset: UnifiedDataset, output_dim: Optional[tuple] = None) -> UnifiedDataset:
#         if not self.config.get('enable_cutout', True):
#             logging.info("Cutout augmentation is disabled.")
#             return UnifiedDataset()

#         augmented_dataset = UnifiedDataset(
#             images=[],
#             annotations=[],
#             categories=copy.deepcopy(dataset.categories)
#         )

#         # Get the maximum existing image and annotation IDs
#         existing_image_ids = [img.id for img in dataset.images]
#         existing_annotation_ids = [ann.id for ann in dataset.annotations]
#         image_id_offset = max(existing_image_ids) + 1 if existing_image_ids else 1
#         annotation_id_offset = max(existing_annotation_ids) + 1 if existing_annotation_ids else 1

#         # Create a mapping from image_id to annotations
#         image_id_to_annotations = {}
#         for ann in dataset.annotations:
#             image_id_to_annotations.setdefault(ann.image_id, []).append(ann)

#         # Define max_clipped_area_per_category if not provided
#         max_clipped_area_per_category = self.config['max_clipped_area_per_category']
#         if not max_clipped_area_per_category:
#             # Assign a default value if not specified, e.g., 0.05 (5%) for all categories
#             max_clipped_area_per_category = {cat['id']: 0.5 for cat in dataset.categories}

#         output_images_dir = self.config['output_images_dir']

#         for img in dataset.images:
#             image_path = img.file_name
#             image = load_image(image_path)
#             if image is None:
#                 logging.error(f"Failed to load image '{image_path}'. Skipping.")
#                 continue
#             img_h, img_w = image.shape[:2]

#             # Get annotations for the current image
#             anns = image_id_to_annotations.get(img.id, [])

#             # Initialize cutouts_applied for each image
#             cutouts_applied = 0

#             # Apply augmentations per image
#             for cutout_num in range(self.config['num_cutouts']):
#                 try:
#                     # Decide whether to apply cutout based on probability
#                     prob = self.config['cutout_probability']
#                     if random.random() > prob:
#                         logging.info(f"Skipping cutout augmentation {cutout_num+1} for image ID {img.id} based on probability ({prob}).")
#                         continue  # Skip this augmentation

#                     # Check if systematic cutout is enabled
#                     if self.config.get('systematic_cutout', False):
#                         augmented_image, masks = self.apply_systematic_cutout(image, anns)
#                     else:
#                         # Apply Random Cutout
#                         augmented_image, masks = self.apply_random_cutout(
#                             image=image,
#                             cutout_size=self.config['cutout_size'],
#                             num_cutouts=3,  # Apply one cutout at a time
#                             p=self.config['cutout_p']
#                         )

#                     if not masks:
#                         logging.info(f"No cutouts applied for augmentation {cutout_num+1} on image ID {img.id}. Skipping augmentation.")
#                         continue

#                     # Create mask polygons for annotation clipping
#                     mask_polygons = [box(x1, y1, x2, y2) for (x1, y1, x2, y2) in masks]

#                     # Define the image boundary
#                     image_boundary = box(0, 0, img_w, img_h)

#                     # Subtract mask polygons from image boundary to get the valid region
#                     valid_region = image_boundary
#                     for mask in mask_polygons:
#                         valid_region = valid_region.difference(mask)

#                     # Process annotations
#                     transformed_annotations = copy.deepcopy(anns)
#                     cleaned_anns = []

#                     discard_augmentation = False  # Flag to decide whether to discard the entire augmentation

#                     for ann in transformed_annotations:
#                         category_id = ann.category_id
#                         max_allowed_reduction = max_clipped_area_per_category.get(category_id, 0.05)
#                         coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
#                         if not coords:
#                             logging.warning(f"Empty polygon for annotation ID {ann.id} in image ID {img.id}. Skipping annotation.")
#                             continue
#                         ann_poly = Polygon(coords)
#                         if not ann_poly.is_valid:
#                             ann_poly = ann_poly.buffer(0)
#                         # Clip the annotation polygon against the valid region
#                         clipped_poly = ann_poly.intersection(valid_region)
#                         original_area = ann_poly.area
#                         new_area = clipped_poly.area

#                         if clipped_poly.is_empty:
#                             logging.info(f"Annotation ID {ann.id} in image ID {img.id} is fully masked out by cutout. Skipping annotation.")
#                             continue  # Annotation is fully masked out

#                         area_reduction_due_to_clipping = calculate_area_reduction(original_area, new_area)

#                         if area_reduction_due_to_clipping > max_allowed_reduction:
#                             logging.warning(f"Cutout augmentation {cutout_num+1} for image ID {img.id} discarded due to area reduction ({area_reduction_due_to_clipping:.6f}) exceeding threshold ({max_allowed_reduction}) for category {category_id}.")
#                             discard_augmentation = True
#                             break  # Discard the entire augmentation

#                         # Determine if polygon was clipped
#                         is_polygon_clipped = area_reduction_due_to_clipping > 0.01

#                         # Handle MultiPolygon cases
#                         polygons_to_process = []
#                         if isinstance(clipped_poly, Polygon):
#                             polygons_to_process.append(clipped_poly)
#                         elif isinstance(clipped_poly, MultiPolygon):
#                             polygons_to_process.extend(clipped_poly.geoms)
#                         else:
#                             logging.warning(f"Unsupported geometry type {type(clipped_poly)} for annotation ID {ann.id} in image ID {img.id}. Skipping annotation.")
#                             continue  # Unsupported geometry type

#                         # Collect cleaned polygon coordinates
#                         cleaned_polygon_coords = []
#                         for poly in polygons_to_process:
#                             if self.task == 'detection':
#                                 coords = ensure_axis_aligned_rectangle(list(poly.exterior.coords))
#                                 if coords:
#                                     cleaned_polygon_coords.extend(coords)
#                             else:
#                                 coords = list(poly.exterior.coords)
#                                 if coords:
#                                     cleaned_polygon_coords.extend(coords)

#                         if not cleaned_polygon_coords:
#                             logging.debug(f"No valid coordinates found after processing clipped polygons. Skipping annotation.")
#                             continue

#                         # Update the annotation
#                         new_ann = UnifiedAnnotation(
#                             id=annotation_id_offset,
#                             image_id=image_id_offset,
#                             category_id=ann.category_id,
#                             polygon=[coord for point in cleaned_polygon_coords for coord in point],
#                             iscrowd=ann.iscrowd,
#                             area=new_area,
#                             is_polygon_clipped=is_polygon_clipped,
#                             # Note: Exclude 'area_reduction' and 'is_polygon_clipped' as they may not be accepted by UnifiedAnnotation
#                         )

#                         cleaned_anns.append(new_ann)
#                         annotation_id_offset += 1

#                     if discard_augmentation:
#                         logging.info(f"Cutout augmentation {cutout_num+1} for image ID {img.id} discarded due to high area reduction.")
#                         continue  # Skip this augmentation

#                     # If no polygons remain after masking, skip augmentation
#                     if not cleaned_anns:
#                         logging.info(f"Cutout augmentation {cutout_num+1} with masks for image ID {img.id} results in all polygons being fully masked. Skipping augmentation.")
#                         continue

#                     # Generate new filename
#                     filename, ext = os.path.splitext(os.path.basename(img.file_name))
#                     new_filename = f"{filename}_cutout_aug{uuid.uuid4().hex}{ext}"
#                     output_image_path = os.path.join(output_images_dir, new_filename)

#                     # Save augmented image
#                     save_success = save_image(augmented_image, output_image_path)
#                     if not save_success:
#                         logging.error(f"Failed to save augmented image '{output_image_path}'. Skipping this augmentation.")
#                         continue

#                     # Create new image entry
#                     new_img = UnifiedImage(
#                         id=image_id_offset,
#                         file_name=output_image_path,
#                         width=augmented_image.shape[1],
#                         height=augmented_image.shape[0]
#                     )
#                     augmented_dataset.images.append(new_img)

#                     # Add cleaned annotations to the dataset
#                     for new_ann in cleaned_anns:
#                         augmented_dataset.annotations.append(new_ann)
#                         logging.info(f"Added annotation ID {new_ann.id} for image ID {image_id_offset}.")

#                     # Visualization
#                     if self.config['visualize_overlays'] and self.config['output_visualizations_dir']:
#                         visualization_filename = f"{os.path.splitext(new_filename)[0]}_viz{ext}"
#                         mosaic_visualize_transformed_overlays(
#                             transformed_image=augmented_image.copy(),
#                             cleaned_annotations=cleaned_anns,
#                             output_visualizations_dir=self.config['output_visualizations_dir'],
#                             new_filename=visualization_filename,
#                             task=self.task
#                         )

#                     logging.info(f"Cutout augmented image '{new_filename}' saved with {len(cleaned_anns)} annotations.")
#                     cutouts_applied += 1
#                     image_id_offset += 1

#                 except Exception as e:
#                     logging.error(f"Exception occurred during cutout augmentation {cutout_num+1} of image ID {img.id}: {e}", exc_info=True)
#                     continue  # Proceed to the next augmentation

#         logging.info(f"Cutout augmentation applied. Total augmented images: {len(augmented_dataset.images)}, annotations: {len(augmented_dataset.annotations)}.")
#         return augmented_dataset

#     def apply_random_cutout(self, image: np.ndarray, cutout_size: tuple, num_cutouts: int = 1, p: float = 0.5):
#         """
#         Apply Random Cutout augmentation to an image by masking out square or rectangular regions.

#         Args:
#             image (np.ndarray): The input image.
#             cutout_size (tuple): Size of the cutout (height, width).
#             num_cutouts (int): Number of cutouts to apply.
#             p (float): Probability to apply each cutout.

#         Returns:
#             tuple: (augmented_image, masks)
#                 - augmented_image (np.ndarray): The image after applying cutouts.
#                 - masks (list): List of mask tuples (x1, y1, x2, y2).
#         """
#         augmented_image = image.copy()
#         masks = []
#         img_h, img_w = image.shape[:2]

#         for _ in range(num_cutouts):
#             if random.random() > p:
#                 continue  # Skip this cutout with probability (1 - p)

#             # Randomly choose the size of the cutout
#             height = random.randint(int(cutout_size[0] * 0.8), int(cutout_size[0] * 1.2))
#             width = random.randint(int(cutout_size[1] * 0.8), int(cutout_size[1] * 1.2))

#             # Randomly choose the top-left corner of the cutout
#             x1 = random.randint(0, max(img_w - width, 0))
#             y1 = random.randint(0, max(img_h - height, 0))
#             x2 = x1 + width
#             y2 = y1 + height

#             # Apply the cutout (mask with black color)
#             augmented_image[y1:y2, x1:x2] = 0

#             # Save the mask for annotation handling
#             mask = (x1, y1, x2, y2)
#             masks.append(mask)
#             logging.debug(f"Applied cutout: Top-left=({x1}, {y1}), Bottom-right=({x2}, {y2})")

#         return augmented_image, masks

#     def apply_systematic_cutout(self, image: np.ndarray, annotations: List[UnifiedAnnotation]):
#         """
#         Apply systematic cutout by masking out the largest or smallest polygon.

#         Args:
#             image (np.ndarray): The input image.
#             annotations (List[UnifiedAnnotation]): List of annotations.

#         Returns:
#             tuple: (augmented_image, masks)
#                 - augmented_image (np.ndarray): The image after applying cutout.
#                 - masks (list): List of mask tuples (x1, y1, x2, y2).
#         """
#         augmented_image = image.copy()
#         masks = []
#         img_h, img_w = image.shape[:2]
#         margin = self.config.get('margin_pixels', 10)
#         mode = self.config.get('systematic_cutout_mode', 'largest').lower()

#         # Find the target annotation based on mode
#         if not annotations:
#             logging.warning("No annotations available for systematic cutout.")
#             return augmented_image, masks

#         areas = []
#         for ann in annotations:
#             coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
#             if not coords:
#                 continue
#             poly = Polygon(coords)
#             if not poly.is_valid:
#                 poly = poly.buffer(0)
#             areas.append((poly.area, ann, poly))

#         if not areas:
#             logging.warning("No valid polygons found for systematic cutout.")
#             return augmented_image, masks

#         if mode == 'largest':
#             target_ann = max(areas, key=lambda x: x[0])
#         elif mode == 'smallest':
#             target_ann = min(areas, key=lambda x: x[0])
#         else:
#             logging.warning(f"Unsupported systematic_cutout_mode '{mode}'. Defaulting to 'largest'.")
#             target_ann = max(areas, key=lambda x: x[0])

#         # Get the bounding box of the target polygon with margin
#         poly = target_ann[2]
#         minx, miny, maxx, maxy = poly.bounds
#         x1 = max(int(minx) - margin, 0)
#         y1 = max(int(miny) - margin, 0)
#         x2 = min(int(maxx) + margin, img_w)
#         y2 = min(int(maxy) + margin, img_h)

#         # Apply the cutout (mask with black color)
#         augmented_image[y1:y2, x1:x2] = 0

#         # Save the mask for annotation handling
#         mask = (x1, y1, x2, y2)
#         masks.append(mask)
#         logging.debug(f"Applied systematic cutout: Top-left=({x1}, {y1}), Bottom-right=({x2}, {y2})")

#         return augmented_image, masks

# augflow/augmentations/cutout.py

import os
import copy
import random
import logging
import cv2
import numpy as np
from shapely.geometry import box, Polygon, MultiPolygon
from .base import Augmentation
from augflow.utils.images import load_image, save_image, mosaic_visualize_transformed_overlays
from augflow.utils.annotations import (
    calculate_area_reduction,
    ensure_axis_aligned_rectangle
)
from augflow.utils.unified_format import UnifiedDataset, UnifiedImage, UnifiedAnnotation
import uuid
from typing import Optional, List, Dict
from augflow.utils.configs import cutout_default_config

class CutoutAugmentation(Augmentation):
    def __init__(self, config=None, task: str = 'detection'):
        super().__init__()
        self.task = task.lower()
        self.config = cutout_default_config.copy() 
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
        if not self.config.get('enable_cutout', True):
            logging.info("Cutout augmentation is disabled.")
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
            # Assign a default value if not specified, e.g., 0.05 (5%) for all categories
            max_clipped_area_per_category = {cat['id']: 0.8 for cat in dataset.categories}

        output_images_dir = self.config['output_images_dir']

        for img in dataset.images:
            image_path = img.file_name
            image = load_image(image_path)
            if image is None:
                logging.error(f"Failed to load image '{image_path}'. Skipping.")
                continue
            img_h, img_w = image.shape[:2]

            # Get annotations for the current image
            anns = image_id_to_annotations.get(img.id, [])

            # Initialize cutouts_applied for each image
            cutouts_applied = 0

            # Apply augmentations per image
            for cutout_num in range(self.config['num_cutouts']):
                try:
                    # Decide whether to apply cutout based on probability
                    prob = self.config['cutout_probability']
                    if random.random() > prob:
                        logging.info(f"Skipping cutout augmentation {cutout_num+1} for image ID {img.id} based on probability ({prob}).")
                        continue  # Skip this augmentation

                    # Check if systematic cutout is enabled
                    if self.config.get('systematic_cutout', False):
                        augmented_image, masks = self.apply_systematic_cutout(image, anns)
                    else:
                        # Apply Random Cutout
                        augmented_image, masks = self.apply_random_cutout(
                            image=image,
                            cutout_size=self.config['cutout_size'],
                            num_cutouts=1,  # Apply one cutout at a time
                            p=self.config['cutout_p']
                        )

                    if not masks:
                        logging.info(f"No cutouts applied for augmentation {cutout_num+1} on image ID {img.id}. Skipping augmentation.")
                        continue

                    # Create mask polygons for annotation clipping
                    mask_polygons = [box(x1, y1, x2, y2) for (x1, y1, x2, y2) in masks]

                    # Define the image boundary
                    image_boundary = box(0, 0, img_w, img_h)

                    # Subtract mask polygons from image boundary to get the valid region
                    valid_region = image_boundary
                    for mask in mask_polygons:
                        valid_region = valid_region.difference(mask)

                    # Process annotations
                    transformed_annotations = copy.deepcopy(anns)
                    cleaned_anns = []

                    discard_augmentation = False  # Flag to decide whether to discard the entire augmentation

                    for ann in transformed_annotations:
                        category_id = ann.category_id
                        max_allowed_reduction = max_clipped_area_per_category.get(category_id, 0.05)
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
                            new_area = poly.area
                            area_reduction_due_to_clipping = calculate_area_reduction(original_area, new_area)

                            if area_reduction_due_to_clipping > max_allowed_reduction:
                                logging.warning(f"Cutout augmentation for image ID {img.id} discarded due to area reduction ({area_reduction_due_to_clipping:.6f}) exceeding threshold ({max_allowed_reduction}) for category {category_id}.")
                                discard_augmentation = True
                                break  # Discard the entire augmentation

                            is_polygon_clipped = area_reduction_due_to_clipping > 0.01

                            if self.task == 'detection':
                                coords = ensure_axis_aligned_rectangle(list(poly.exterior.coords))
                            else:
                                coords = list(poly.exterior.coords)

                            if not coords:
                                logging.debug(f"No valid coordinates found after processing clipped polygons. Skipping polygon.")
                                continue

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
                            logging.info(f"Cutout augmentation for image ID {img.id} discarded due to high area reduction.")
                            break  # Discard the entire augmentation

                    if discard_augmentation:
                        continue  # Skip this augmentation if discarded

                    # If no polygons remain after masking, skip augmentation
                    if not cleaned_anns:
                        logging.info(f"Cutout augmentation {cutout_num+1} with masks for image ID {img.id} results in all polygons being fully masked. Skipping augmentation.")
                        continue

                    # Generate new filename
                    filename, ext = os.path.splitext(os.path.basename(img.file_name))
                    new_filename = f"{filename}_cutout_aug{uuid.uuid4().hex}{ext}"
                    output_image_path = os.path.join(output_images_dir, new_filename)

                    # Save augmented image
                    save_success = save_image(augmented_image, output_image_path)
                    if not save_success:
                        logging.error(f"Failed to save augmented image '{output_image_path}'. Skipping this augmentation.")
                        continue

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
                    cutouts_applied += 1
                    image_id_offset += 1

                except Exception as e:
                    logging.error(f"Exception occurred during cutout augmentation {cutout_num+1} of image ID {img.id}: {e}", exc_info=True)
                    continue  # Proceed to the next augmentation

        logging.info(f"Cutout augmentation applied. Total augmented images: {len(augmented_dataset.images)}, annotations: {len(augmented_dataset.annotations)}.")
        return augmented_dataset

    def apply_random_cutout(self, image: np.ndarray, cutout_size: tuple, num_cutouts: int = 1, p: float = 0.5):
        """
        Apply Random Cutout augmentation to an image by masking out square or rectangular regions.

        Args:
            image (np.ndarray): The input image.
            cutout_size (tuple): Size of the cutout (height, width).
            num_cutouts (int): Number of cutouts to apply.
            p (float): Probability to apply each cutout.

        Returns:
            tuple: (augmented_image, masks)
                - augmented_image (np.ndarray): The image after applying cutouts.
                - masks (list): List of mask tuples (x1, y1, x2, y2).
        """
        augmented_image = image.copy()
        masks = []
        img_h, img_w = image.shape[:2]

        for _ in range(num_cutouts):
            if random.random() > p:
                continue  # Skip this cutout with probability (1 - p)

            # Randomly choose the size of the cutout
            height = random.randint(int(cutout_size[0] * 0.8), int(cutout_size[0] * 1.2))
            width = random.randint(int(cutout_size[1] * 0.8), int(cutout_size[1] * 1.2))

            # Randomly choose the top-left corner of the cutout
            x1 = random.randint(0, max(img_w - width, 0))
            y1 = random.randint(0, max(img_h - height, 0))
            x2 = x1 + width
            y2 = y1 + height

            # Apply the cutout (mask with black color)
            augmented_image[y1:y2, x1:x2] = 0

            # Save the mask for annotation handling
            mask = (x1, y1, x2, y2)
            masks.append(mask)
            logging.debug(f"Applied cutout: Top-left=({x1}, {y1}), Bottom-right=({x2}, {y2})")

        return augmented_image, masks

    def apply_systematic_cutout(self, image: np.ndarray, annotations: List[UnifiedAnnotation]):
        """
        Apply systematic cutout by masking out the largest or smallest polygon.

        Args:
            image (np.ndarray): The input image.
            annotations (List[UnifiedAnnotation]): List of annotations.

        Returns:
            tuple: (augmented_image, masks)
                - augmented_image (np.ndarray): The image after applying cutout.
                - masks (list): List of mask tuples (x1, y1, x2, y2).
        """
        augmented_image = image.copy()
        masks = []
        img_h, img_w = image.shape[:2]
        margin = self.config.get('margin_pixels', 10)
        mode = self.config.get('systematic_cutout_mode', 'largest').lower()

        # Find the target annotation based on mode
        if not annotations:
            logging.warning("No annotations available for systematic cutout.")
            return augmented_image, masks

        areas = []
        for ann in annotations:
            coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
            if not coords:
                continue
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            areas.append((poly.area, ann, poly))

        if not areas:
            logging.warning("No valid polygons found for systematic cutout.")
            return augmented_image, masks

        if mode == 'largest':
            target_ann = max(areas, key=lambda x: x[0])
        elif mode == 'smallest':
            target_ann = min(areas, key=lambda x: x[0])
        else:
            logging.warning(f"Unsupported systematic_cutout_mode '{mode}'. Defaulting to 'largest'.")
            target_ann = max(areas, key=lambda x: x[0])

        # Get the bounding box of the target polygon with margin
        poly = target_ann[2]
        minx, miny, maxx, maxy = poly.bounds
        x1 = max(int(minx) - margin, 0)
        y1 = max(int(miny) - margin, 0)
        x2 = min(int(maxx) + margin, img_w)
        y2 = min(int(maxy) + margin, img_h)

        # Apply the cutout (mask with black color)
        augmented_image[y1:y2, x1:x2] = 0

        # Save the mask for annotation handling
        mask = (x1, y1, x2, y2)
        masks.append(mask)
        logging.debug(f"Applied systematic cutout: Top-left=({x1}, {y1}), Bottom-right=({x2}, {y2})")

        return augmented_image, masks
