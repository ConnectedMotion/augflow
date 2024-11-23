
import os
import copy
import logging
from typing import Optional, List, Dict, Union
from shapely.geometry import Polygon
import numpy as np

from augflow.augmentations.affine import AffineAugmentation
from augflow.augmentations.rotate import RotateAugmentation
from augflow.augmentations.shear import ShearAugmentation
from augflow.augmentations.translate import TranslateAugmentation
from augflow.augmentations.cutout import CutoutAugmentation
from augflow.augmentations.crop import CropAugmentation
from augflow.augmentations.mosaic import MosaicAugmentation
from augflow.augmentations.flip import FlipAugmentation
from augflow.augmentations.scale import ScaleAugmentation
from augflow.augmentations.blur import BlurAugmentation
from augflow.augmentations.brightness_contrast import BrightnessContrastAugmentation
from augflow.augmentations.color_shift import ColorShiftAugmentation
from augflow.augmentations.weather import WeatherAugmentation
from augflow.augmentations.noise import NoiseAugmentation
from augflow.utils.parsers import CocoParser, YoloParser
from augflow.utils.images import save_visualization, load_image
from augflow.utils.unified_format import UnifiedDataset

# Configure logging
logging.basicConfig(
    filename='augflow_pipeline.log',
    filemode='w',
    level=logging.DEBUG,  # Set to DEBUG level
    format='%(asctime)s:%(levelname)s:%(message)s'
)

class Pipeline:
    def __init__(self):
        self.datasets = {}
        self.current_id = 'root'
        self.augmentations = {}
        self.merged_dataset_ids = []
        logging.info("Pipeline initialized.")

    def task(self, format: str, dataset_path: str):
        """
        Initialize the pipeline with a dataset.

        Args:
            format (str): 'coco' or 'yolo'.
            dataset_path (str): Path to the dataset directory.
        """
        # Initialize the appropriate parser
        if format.lower() == 'coco':
            parser = CocoParser()
        elif format.lower() == 'yolo':
            parser = YoloParser()
        else:
            raise NotImplementedError(f"Format '{format}' is not supported.")

        # Load the dataset and discover the task
        dataset, task = parser.load(dataset_path)
        if dataset is None:
            raise ValueError("Failed to load dataset.")

        # Assign to 'root' dataset
        self.datasets['root'] = {
            'dataset': dataset,
            'format': format.lower(),
            'task': task.lower()
        }
        logging.info(f"Dataset loaded with task '{task}' and root dataset initialized.")

    def fuse(
        self,
        source_id: str,
        type: str,
        config: Optional[dict] = None,
        output_dim: Optional[tuple] = None,
        id: Optional[str] = None,
        merge: bool = False,
        focus: Optional[Dict[str, Union[List[Union[int, str]], str]]] = None,
        min_relative_area: Optional[float] = None
    ):
        """
        Apply an augmentation to the dataset.

        Args:
            source_id (str): ID of the source dataset to augment.
            type (str): Type of augmentation ('affine', etc.).
            config (dict, optional): Configuration parameters for the augmentation.
            output_dim (tuple, optional): Desired output dimensions (width, height).
            id (str, optional): ID for the augmented dataset.
            merge (bool, optional): Whether to include the augmented dataset in the final output.
            focus (dict, optional): Include or exclude specific categories.
            min_relative_area (float, optional): Minimum relative area threshold for annotations.
        """
        if id is None:
            id = f"aug_{len(self.augmentations)}"
        if source_id not in self.datasets:
            raise ValueError(f"Source dataset '{source_id}' not found.")

        source_dataset = self.datasets[source_id]['dataset']
        format = self.datasets[source_id]['format']
        task = self.datasets[source_id]['task']

        # Ensure areas are computed for all annotations
        self.compute_annotation_areas(source_dataset.annotations)

        # Build category name to ID mapping
        category_name_to_id = {cat['name']: cat['id'] for cat in source_dataset.categories}
        category_id_to_name = {cat['id']: cat['name'] for cat in source_dataset.categories}

        # Handle focus parameter
        if focus:
            include_categories = set()
            exclude_categories = set()

            if 'include' in focus:
                include_items = focus['include']
                if not isinstance(include_items, list):
                    raise ValueError("'include' must be a list of category names or IDs.")
                for item in include_items:
                    if isinstance(item, int):
                        include_categories.add(item)
                    elif isinstance(item, str):
                        if item in category_name_to_id:
                            include_categories.add(category_name_to_id[item])
                        else:
                            logging.warning(f"Included category name '{item}' not found in dataset categories.")
                    else:
                        raise ValueError("'include' items must be category IDs (int) or names (str).")
            else:
                include_categories = set(cat['id'] for cat in source_dataset.categories)

            if 'exclude' in focus:
                exclude_items = focus['exclude']
                if exclude_items == 'all_others':
                    # Exclude all categories not in include_categories
                    exclude_categories = set(cat['id'] for cat in source_dataset.categories) - include_categories
                else:
                    if not isinstance(exclude_items, list):
                        raise ValueError("'exclude' must be 'all_others' or a list of category names or IDs.")
                    for item in exclude_items:
                        if isinstance(item, int):
                            exclude_categories.add(item)
                        elif isinstance(item, str):
                            if item in category_name_to_id:
                                exclude_categories.add(category_name_to_id[item])
                            else:
                                logging.warning(f"Excluded category name '{item}' not found in dataset categories.")
                        else:
                            raise ValueError("'exclude' items must be category IDs (int) or names (str).")

            # Remove any None values (categories not found)
            include_categories.discard(None)
            exclude_categories.discard(None)

            # Ensure include_categories does not contain any excluded categories
            include_categories -= exclude_categories

            # If include is empty after exclusion, include all categories not in exclude
            if not include_categories:
                include_categories = set(cat['id'] for cat in source_dataset.categories) - exclude_categories
        else:
            include_categories = set(cat['id'] for cat in source_dataset.categories)
            exclude_categories = set()

        # Build a mapping from image_id to annotations
        image_id_to_annotations = {}
        for ann in source_dataset.annotations:
            image_id_to_annotations.setdefault(ann.image_id, []).append(ann)

        filtered_images = []
        filtered_annotations = []
        for img in source_dataset.images:
            anns = image_id_to_annotations.get(img.id, [])

            # Skip image if any annotation has category_id in exclude_categories
            if any(ann.category_id in exclude_categories for ann in anns):
                continue

            # Check if image has at least one annotation in include_categories
            if not any(ann.category_id in include_categories for ann in anns):
                continue

            # Check for small annotations relative to image area
            discard_image_due_to_small_annotation = False
            if min_relative_area is not None:
                img_area = img.width * img.height
                for ann in anns:
                    if ann.category_id in include_categories and ann.area / img_area < min_relative_area:
                        discard_image_due_to_small_annotation = True
                        logging.debug(
                            f"Discarding image ID {img.id} due to annotation ID {ann.id} with relative area "
                            f"{ann.area / img_area:.6f} below threshold {min_relative_area}."
                        )
                        break  # No need to check other annotations

            if discard_image_due_to_small_annotation:
                continue  # Discard this image and its annotations

            # Include the image
            filtered_images.append(img)

            # Include only annotations with category_id in include_categories
            for ann in anns:
                if ann.category_id in include_categories:
                    filtered_annotations.append(ann)

        # Filter categories
        filtered_categories = [cat for cat in source_dataset.categories if cat['id'] in include_categories]

        # Create filtered dataset
        filtered_dataset = UnifiedDataset(
            images=filtered_images,
            annotations=filtered_annotations,
            categories=filtered_categories
        )
        logging.info(
            f"Filtered dataset '{id}' with {len(filtered_images)} images and {len(filtered_annotations)} annotations."
        )

        # Initialize augmentation
        if type == 'affine':
            augmentation = AffineAugmentation(config, task)

        elif type == 'rotate':
            augmentation = RotateAugmentation(config, task)

        elif type == 'shear':
            augmentation = ShearAugmentation(config, task)

        elif type == 'translate':
            augmentation = TranslateAugmentation(config, task)

        elif type == 'cutout':
            augmentation = CutoutAugmentation(config, task)

        elif type == 'crop':
            augmentation = CropAugmentation(config, task)

        elif type == 'mosaic':
            augmentation = MosaicAugmentation(config, task)

        elif type == 'flip':
            augmentation = FlipAugmentation(config, task)

        elif type == 'scale':
            augmentation = ScaleAugmentation(config, task)

        elif type == 'blur':
            augmentation = BlurAugmentation(config, task)

        elif type == 'brightness_contrast':
            augmentation = BrightnessContrastAugmentation(config, task)

        elif type == 'color_shift':
            augmentation = ColorShiftAugmentation(config, task)

        elif type == 'weather':
            augmentation = WeatherAugmentation(config, task)

        elif type == 'noise':
            augmentation = NoiseAugmentation(config, task)

        else:
            raise NotImplementedError(f"Augmentation type '{type}' is not implemented.")

        # Apply augmentation
        augmented_dataset = augmentation.apply(filtered_dataset, output_dim)

        # Save augmented data
        self.datasets[id] = {
            'dataset': augmented_dataset,
            'format': format,
            'task': task  # Use the same task as the source dataset
        }
        logging.info(f"Augmentation '{id}' of type '{type}' applied.")

        if merge:
            self.merged_dataset_ids.append(id)
            logging.info(f"Dataset '{id}' marked for merging into final output.")

        self.current_id = id
        self.augmentations[id] = augmentation

    def compute_annotation_areas(self, annotations):
        """
        Ensure that all annotations have the 'area' attribute computed.
        """
        for ann in annotations:
            if ann.area is None or ann.area == 0:
                coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
                if coords:
                    polygon = Polygon(coords)
                    ann.area = polygon.area
                else:
                    ann.area = 0.0  # Default to zero area if no valid polygon

    def merge_datasets(self, datasets_list: List[str]) -> UnifiedDataset:
        """
        Merge multiple datasets.

        Args:
            datasets_list (List[str]): List of dataset IDs to merge.

        Returns:
            UnifiedDataset: Merged dataset.
        """
        merged_dataset = UnifiedDataset()
        image_id_offset = 0
        annotation_id_offset = 0

        # Collect all categories
        all_categories = {}
        next_category_id = 0
        category_id_mapping = {}  # Map original category IDs to new category IDs

        for ds_id in datasets_list:
            dataset_info = self.datasets[ds_id]
            dataset = dataset_info['dataset']

            # Remap category IDs
            for cat in dataset.categories:
                original_cat_id = cat['id']
                if original_cat_id not in category_id_mapping:
                    # Assign new category ID
                    new_cat_id = next_category_id
                    next_category_id += 1
                    all_categories[new_cat_id] = {
                        'id': new_cat_id,
                        'name': cat['name'],
                        'supercategory': cat.get('supercategory', 'none')
                    }
                    category_id_mapping[original_cat_id] = new_cat_id

            # Create a mapping from old image ids to new image ids
            image_id_mapping = {}
            for img in dataset.images:
                new_img = copy.deepcopy(img)
                new_img.id = img.id + image_id_offset
                merged_dataset.images.append(new_img)
                image_id_mapping[img.id] = new_img.id

            for ann in dataset.annotations:
                new_ann = copy.deepcopy(ann)
                new_ann.id = ann.id + annotation_id_offset
                new_ann.image_id = image_id_mapping.get(ann.image_id, ann.image_id)
                # Update category ID
                new_ann.category_id = category_id_mapping.get(ann.category_id, ann.category_id)
                merged_dataset.annotations.append(new_ann)

            # Update offsets
            if dataset.images:
                image_id_offset = max(img.id for img in merged_dataset.images) + 1
            if dataset.annotations:
                annotation_id_offset = max(ann.id for ann in merged_dataset.annotations) + 1

        # Assign merged categories
        merged_dataset.categories = list(all_categories.values())

        logging.info(f"Merged datasets into one dataset with {len(merged_dataset.images)} images and {len(merged_dataset.annotations)} annotations.")
        return merged_dataset

    def out(
        self,
        format: str,
        output_path: str,
        ignore_masks: bool = False,
        visualize_annotations: bool = False
    ):
        """
        Output the augmented dataset in the specified format.

        Args:
            format (str): 'coco' or 'yolo'.
            output_path (str): Path to save the augmented dataset.
            ignore_masks (bool, optional): Whether to ignore masks (convert segmentation to bounding boxes).
            visualize_annotations (bool, optional): Whether to visualize annotations.
        """
        if not self.merged_dataset_ids:
            raise ValueError("No datasets marked for merging. Please set 'merge=True' in 'fuse' method.")

        # Merge datasets
        merged_dataset = self.merge_datasets(self.merged_dataset_ids)

        # If ignore_masks is True, convert polygons to bounding boxes
        if ignore_masks:
            for ann in merged_dataset.annotations:
                coords = ann.polygon
                if coords and len(coords) >= 6:  # At least 3 points needed
                    x_coords = coords[0::2]
                    y_coords = coords[1::2]
                    x_min = min(x_coords)
                    y_min = min(y_coords)
                    x_max = max(x_coords)
                    y_max = max(y_coords)
                    # Update the polygon to be the bounding box coordinates
                    ann.polygon = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
            # Set the task to detection
            task = 'detection'
        else:
            # Determine task based on whether polygons represent bounding boxes
            is_detection = all(self.is_axis_aligned_rectangle(ann.polygon) for ann in merged_dataset.annotations)
            task = 'detection' if is_detection else 'segmentation'

        # Initialize the appropriate parser for output format
        if format.lower() == 'coco':
            parser = CocoParser()
        elif format.lower() == 'yolo':
            parser = YoloParser()
        else:
            raise NotImplementedError(f"Output format '{format}' is not supported.")

        # Save the dataset
        parser.save(merged_dataset, output_path, task=task)
        logging.info(f"Dataset saved in '{format}' format to '{output_path}' with task '{task}'.")

        # Handle visualization
        if visualize_annotations:
            visualizations_output_dir = os.path.join(output_path, 'visualizations')
            os.makedirs(visualizations_output_dir, exist_ok=True)
            category_id_to_name = {cat['id']: cat['name'] for cat in merged_dataset.categories}
            image_id_to_annotations = {}
            for ann in merged_dataset.annotations:
                image_id_to_annotations.setdefault(ann.image_id, []).append(ann)

            for img in merged_dataset.images:
                img_id = img.id
                img_filename = img.file_name
                anns = image_id_to_annotations.get(img_id, [])
                if not anns:
                    logging.warning(f"No annotations found for image ID {img_id}.")
                    continue
                visualization_filename = os.path.splitext(os.path.basename(img_filename))[0] + '_viz.jpg'
                visualization_path = os.path.join(visualizations_output_dir, visualization_filename)
                image = load_image(img_filename)
                if image is None:
                    logging.error(f"Cannot load image '{img_filename}' for visualization.")
                    continue
                # Pass format and bbox_format to visualization
                save_visualization(
                    transformed_image=image,
                    cleaned_annotations=anns,
                    category_id_to_name=category_id_to_name,
                    output_path=visualization_path,
                    format=format.lower(),
                    task=task  # Pass the task
                )
            logging.info(f"Saved visualization images to '{visualizations_output_dir}'.")

    def is_axis_aligned_rectangle(self, polygon: List[float]) -> bool:
        """
        Check if a polygon is an axis-aligned rectangle.

        Args:
            polygon (List[float]): The polygon coordinates [x1, y1, x2, y2, ..., xn, yn].

        Returns:
            bool: True if the polygon is an axis-aligned rectangle, False otherwise.
        """
        if len(polygon) != 8:
            return False  # Not a rectangle
        x_coords = polygon[0::2]
        y_coords = polygon[1::2]
        unique_x = set(x_coords)
        unique_y = set(y_coords)
        if len(unique_x) != 2 or len(unique_y) != 2:
            return False  # Not axis-aligned rectangle
        return True
