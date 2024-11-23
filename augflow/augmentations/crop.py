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
from augflow.utils.images import load_image, save_image, mosaic_visualize_transformed_overlays, crop_image
from augflow.utils.unified_format import UnifiedDataset, UnifiedImage, UnifiedAnnotation
import uuid
from typing import Optional, List, Dict
from augflow.utils.annotations import ensure_axis_aligned_rectangle, calculate_iou
from augflow.utils.configs import crop_default_config


class CropAugmentation(Augmentation):
    def __init__(self, config=None, task: str = 'detection'):
        super().__init__()
        
        self.config = crop_default_config.copy() 
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
            # This will be handled in the apply method if not set
            self.config['max_clipped_area_per_category'] = {}
    
    
    
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
        max_clipped_area_per_category = self.config['max_clipped_area_per_category']
        if not max_clipped_area_per_category:
            # Assign a default value if not specified, e.g., 0.5 (99%) for all categories
            max_clipped_area_per_category = {cat['id']: 0.5 for cat in dataset.categories}

        # Handle predefined_crops being None
        predefined_crops = self.config.get('predefined_crops')
        if predefined_crops is None:
            predefined_crops = {}

        # Iterate through each image
        for img in dataset.images:
            image_path = img.file_name
            image = load_image(image_path)
            if image is None:
                logging.error(f"Failed to load image '{image_path}'. Skipping.")
                continue
            img_h, img_w = image.shape[:2]

            # Validate desired_output_size
            if self.config['desired_output_size'] is not None:
                if not (isinstance(self.config['desired_output_size'], tuple) and len(self.config['desired_output_size']) == 2):
                    logging.error("desired_output_size must be a tuple of (width, height) or None.")
                    continue
                desired_w, desired_h = self.config['desired_output_size']
                if desired_w <= 0 or desired_h <= 0:
                    logging.error("desired_output_size dimensions must be positive integers.")
                    continue
            else:
                desired_w, desired_h = img_w, img_h  # Use original size for reference

            # Create Shapely polygon for image boundaries
            image_boundary = box(0, 0, img_w, img_h)

            # Determine number of augmentations for this image
            num_crops = self.config['num_crops_per_image']

            # Initialize list to keep track of existing crop bounding boxes for overlap control
            existing_crops = []

            # Initialize list to hold crops
            crops = []

            # Check if systematic crops are enabled
            if self.config.get('systematic_crops', False):
                sys_params = self.config.get('systematic_crops_parameters', {})
                mode = sys_params.get('mode', 'largest')
                margin = sys_params.get('margin', 50)  # Default margin is 50 pixels
                if mode == 'largest':
                    # Find the largest polygon in the image
                    anns = image_id_to_annotations.get(img.id, [])
                    if not anns:
                        logging.info(f"No annotations found for image ID {img.id}. Skipping systematic cropping.")
                        continue

                    # Find the annotation with the largest area
                    max_area = 0
                    largest_ann = None
                    for ann in anns:
                        coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
                        if not coords:
                            continue
                        polygon = Polygon(coords)
                        if not polygon.is_valid:
                            polygon = polygon.buffer(0)
                        area = polygon.area
                        if area > max_area:
                            max_area = area
                            largest_ann = ann

                    if largest_ann is None:
                        logging.info(f"No valid polygons found for image ID {img.id}. Skipping systematic cropping.")
                        continue

                    # Create a crop around the largest polygon
                    coords = list(zip(largest_ann.polygon[0::2], largest_ann.polygon[1::2]))
                    polygon = Polygon(coords)
                    minx, miny, maxx, maxy = polygon.bounds

                    # Expand the bounds by margin, ensuring within image boundaries
                    x1 = int(minx) - margin
                    y1 = int(miny) - margin
                    x2 = int(maxx) + margin
                    y2 = int(maxy) + margin

                    # Clip to image boundaries
                    x1 = max(x1, 0)
                    y1 = max(y1, 0)
                    x2 = min(x2, img_w)
                    y2 = min(y2, img_h)

                    w = x2 - x1
                    h = y2 - y1

                    crops = [(x1, y1, w, h)]
                elif mode == 'categories':
                    categories_to_focus = sys_params.get('categories', [])
                    if not categories_to_focus:
                        logging.info(f"No categories specified for systematic cropping in 'categories' mode. Skipping systematic cropping.")
                        continue
                    margin = sys_params.get('margin', 50)  # Default margin is 50 pixels

                    # Find annotations belonging to the specified categories
                    anns = image_id_to_annotations.get(img.id, [])
                    crops = []
                    for ann in anns:
                        if ann.category_id in categories_to_focus:
                            coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
                            if not coords:
                                continue
                            polygon = Polygon(coords)
                            if not polygon.is_valid:
                                polygon = polygon.buffer(0)
                            minx, miny, maxx, maxy = polygon.bounds

                            # Expand the bounds by margin, ensuring within image boundaries
                            x1 = int(minx) - margin
                            y1 = int(miny) - margin
                            x2 = int(maxx) + margin
                            y2 = int(maxy) + margin

                            # Clip to image boundaries
                            x1 = max(x1, 0)
                            y1 = max(y1, 0)
                            x2 = min(x2, img_w)
                            y2 = min(y2, img_h)

                            w = x2 - x1
                            h = y2 - y1

                            crops.append((x1, y1, w, h))

                    if not crops:
                        logging.info(f"No annotations found for specified categories in image ID {img.id}. Skipping systematic cropping.")
                        continue
                else:
                    logging.warning(f"Unsupported systematic cropping mode '{mode}'. Skipping systematic cropping.")
                    continue
            else:
                # Generate crops based on crop_modes
                for crop_mode in self.config['crop_modes']:
                    params = self.config['crop_size_parameters'].get(crop_mode, {})
                    if crop_mode == 'fixed_size':
                        w = params.get('crop_width', 800)
                        h = params.get('crop_height', 800)
                        if w <= 0 or h <= 0:
                            logging.warning(f"Invalid crop size for 'fixed_size': width={w}, height={h}. Skipping this mode.")
                            continue
                        # Adjust aspect ratio if needed
                        if self.config['aspect_ratio_parameters']['preserve_aspect_ratio']:
                            target_ratio = self.config['aspect_ratio_parameters']['target_aspect_ratio'][0] / self.config['aspect_ratio_parameters']['target_aspect_ratio'][1]
                            current_ratio = w / h
                            if current_ratio > target_ratio:
                                w = int(h * target_ratio)
                            else:
                                h = int(w / target_ratio)
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
                        if self.config['aspect_ratio_parameters']['preserve_aspect_ratio']:
                            target_ratio = self.config['aspect_ratio_parameters']['target_aspect_ratio'][0] / self.config['aspect_ratio_parameters']['target_aspect_ratio'][1]
                        else:
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
                    elif crop_mode == 'random_aspect':
                        min_ar = params.get('min_aspect_ratio', 0.75)
                        max_ar = params.get('max_aspect_ratio', 1.33)
                        if not (0 < min_ar <= max_ar):
                            logging.warning(f"Invalid aspect ratios for 'random_aspect': min={min_ar}, max={max_ar}. Skipping this mode.")
                            continue
                        target_ratio = random.uniform(min_ar, max_ar)
                        crop_area = random.uniform(0.5, 0.9) * img_w * img_h
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
                    elif crop_mode == 'predefined':
                        predefined = predefined_crops.get(str(img.id), [])
                        if not isinstance(predefined, list):
                            logging.warning(f"Predefined crops for image ID {img.id} are not in a list format. Skipping these crops.")
                            continue
                        for predefined_crop in predefined:
                            if not (isinstance(predefined_crop, tuple) and len(predefined_crop) == 4):
                                logging.warning(f"Invalid predefined crop format: {predefined_crop}. Skipping this crop.")
                                continue
                            x_p, y_p, w_p, h_p = predefined_crop
                            if w_p <= 0 or h_p <= 0:
                                logging.warning(f"Invalid predefined crop size: width={w_p}, height={h_p}. Skipping this crop.")
                                continue
                            # Adjust aspect ratio if needed
                            if self.config['aspect_ratio_parameters']['preserve_aspect_ratio']:
                                target_ratio = self.config['aspect_ratio_parameters']['target_aspect_ratio'][0] / self.config['aspect_ratio_parameters']['target_aspect_ratio'][1]
                                current_ratio = w_p / h_p
                                if current_ratio > target_ratio:
                                    w_p = int(h_p * target_ratio)
                                else:
                                    h_p = int(w_p / target_ratio)
                            crops.append((int(x_p), int(y_p), int(w_p), int(h_p)))
                    elif crop_mode == 'grid':
                        grid_rows = params.get('grid_rows', 2)
                        grid_cols = params.get('grid_cols', 2)
                        if not (isinstance(grid_rows, int) and grid_rows > 0 and isinstance(grid_cols, int) and grid_cols > 0):
                            logging.warning(f"Invalid grid parameters: rows={grid_rows}, cols={grid_cols}. Skipping this mode.")
                            continue
                        grid_w = img_w // grid_cols
                        grid_h = img_h // grid_rows
                        for row in range(grid_rows):
                            for col in range(grid_cols):
                                x = col * grid_w
                                y = row * grid_h
                                w = grid_w
                                h = grid_h
                                # Adjust aspect ratio if needed
                                if self.config['aspect_ratio_parameters']['preserve_aspect_ratio']:
                                    target_ratio = self.config['aspect_ratio_parameters']['target_aspect_ratio'][0] / self.config['aspect_ratio_parameters']['target_aspect_ratio'][1]
                                    current_ratio = w / h
                                    if current_ratio > target_ratio:
                                        w = int(h * target_ratio)
                                    else:
                                        h = int(w / target_ratio)
                                crops.append((int(x), int(y), int(w), int(h)))
                    elif crop_mode == 'sliding_window':
                        window_size = params.get('window_size', (400, 400))
                        step_size = params.get('step_size', (200, 200))
                        if not (isinstance(window_size, tuple) and len(window_size) == 2 and
                                isinstance(step_size, tuple) and len(step_size) == 2):
                            logging.warning(f"Invalid sliding_window parameters: window_size={window_size}, step_size={step_size}. Skipping this mode.")
                            continue
                        win_w, win_h = window_size
                        step_w, step_h = step_size
                        if win_w <= 0 or win_h <= 0 or step_w <= 0 or step_h <= 0:
                            logging.warning(f"Invalid window or step sizes: window_size={window_size}, step_size={step_size}. Skipping this mode.")
                            continue
                        for y_start in range(0, img_h - win_h + 1, step_h):
                            for x_start in range(0, img_w - win_w + 1, step_w):
                                x = x_start
                                y = y_start
                                w = win_w
                                h = win_h
                                # Adjust aspect ratio if needed
                                if self.config['aspect_ratio_parameters']['preserve_aspect_ratio']:
                                    target_ratio = self.config['aspect_ratio_parameters']['target_aspect_ratio'][0] / self.config['aspect_ratio_parameters']['target_aspect_ratio'][1]
                                    current_ratio = w / h
                                    if current_ratio > target_ratio:
                                        w_adj = int(h * target_ratio)
                                        h_adj = h
                                    else:
                                        w_adj = w
                                        h_adj = int(w / target_ratio)
                                    crops.append((int(x), int(y), int(w_adj), int(h_adj)))
                                else:
                                    crops.append((int(x), int(y), int(w), int(h)))
                    else:
                        logging.warning(f"Unsupported crop mode '{crop_mode}'. Skipping.")
                        continue

                if not crops:
                    logging.warning(f"No valid crops generated for image ID {img.id}. Skipping this image.")
                    continue

                # Shuffle crops to introduce randomness
                random.shuffle(crops)

                # Limit the number of crops
                crops = crops[:num_crops]

            # Process each crop
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

                # Crop the image and pad to desired size if specified
                cropped_image, pad_left_total, pad_top_total, resize_factors = crop_image(
                    image=image,
                    crop_coords=(x, y, w, h),
                    desired_output_size=self.config['desired_output_size'],
                    clipping_mode=self.config['clipping_parameters']['clipping_mode'],
                    padding_color=self.config['clipping_parameters']['padding_color']
                )

                if cropped_image is None:
                    continue  # Skip this crop due to clipping_mode 'ignore' or other issues

                # Scale factors
                scale_x, scale_y = resize_factors

                # Process annotations for this image
                anns = image_id_to_annotations.get(img.id, [])
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

                    # Apply scaling factors if any
                    if scale_x != 1 or scale_y != 1:
                        scaled_coords = [(px * scale_x, py * scale_y) for px, py in adjusted_coords]
                    else:
                        scaled_coords = adjusted_coords

                    adjusted_polygon = Polygon(scaled_coords)
                    if not adjusted_polygon.is_valid:
                        adjusted_polygon = adjusted_polygon.buffer(0)
                    adjusted_area = adjusted_polygon.area

                    # Compute area reduction due to scaling
                    if original_area > 0:
                        area_reduction_due_to_scaling = max(0.0, (original_area - adjusted_area) / original_area)
                    else:
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
                    max_allowed_reduction = max_clipped_area_per_category.get(category_id, 0.99)  # 99% default

                    if total_area_reduction > max_allowed_reduction:
                        discard_augmentation = True
                        logging.warning(f"Crop {crop_coords} for image ID {img.id} discarded due to total area reduction ({total_area_reduction:.6f}) exceeding threshold ({max_allowed_reduction}) for category {category_id}.")
                        break  # Discard the entire augmentation

                    # Determine if polygon was scaled
                    is_polygon_scaled = area_reduction_due_to_scaling > 0.01

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
                        is_polygon_scaled=is_polygon_scaled,
                        is_polygon_clipped=is_polygon_clipped,
                        area_reduction_due_to_scaling=area_reduction_due_to_scaling,
                        area_reduction_due_to_clipping=area_reduction_due_to_clipping
                    )

                    temp_cropped_annotations.append(cleaned_ann)
                    annotation_id_offset += 1

                if discard_augmentation:
                    logging.info(f"Cropping augmentation for image ID {img.id} discarded due to high area reduction.")
                    continue  # Skip this augmentation

                # If no polygons remain after excluding those completely outside, skip augmentation
                if not temp_cropped_annotations:
                    logging.info(f"Cropping augmentation with crop {crop_coords} for image ID {img.id} results in all polygons being completely outside. Skipping augmentation.")
                    continue

                # Generate unique filename using UUID to prevent collisions
                filename, ext = os.path.splitext(os.path.basename(img.file_name))
                new_filename = f"{filename}_crop_{uuid.uuid4().hex}{ext}"
                output_image_path = os.path.join(self.config['output_images_dir'], new_filename)

                # Save cropped and padded image
                save_success = save_image(cropped_image, output_image_path)
                if not save_success:
                    logging.error(f"Failed to save cropped image '{output_image_path}'. Skipping this augmentation.")
                    continue

                # Add the new crop's bounding box to existing_crops for overlap control
                existing_crops.append(new_crop_bbox)

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
                image_id_offset += 1

        return augmented_dataset



    
    