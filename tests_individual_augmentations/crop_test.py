# tests_individual_augmentations/test_crop_separated_objects.py

import os
import copy
import random
import numpy as np
import cv2
import logging
from shapely.geometry import Polygon
from typing import List, Tuple, Optional

from augflow.augmentations.crop import CropAugmentation
from augflow.utils.unified_format import UnifiedDataset, UnifiedImage, UnifiedAnnotation
from augflow.utils.images import load_image
from augflow.utils.configs import crop_default_config

# Set up logging
log_filename = 'crop_separated_objects_test.log'
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# Assign colors to categories, avoiding red, green, orange, yellow, and their shades
category_colors = {
    1: (75, 0, 130),     # Indigo
    2: (238, 130, 238),  # Violet
    3: (255, 105, 180),  # Hot Pink (focus category)
    4: (0, 191, 255),    # Deep Sky Blue (focus category)
}

# Map category IDs to shapes
category_shapes = {
    1: 'rectangle',
    2: 'pentagon',
    3: 'heptagon',     # Focus category
    4: 'hendecagon',   # Focus category
}

def create_polygon_at_position(position, width, height, shape, size=100):
    """
    Create a polygon at the specified position in the image.
    """
    x_min, y_min = position

    # Ensure the polygon fits within the image boundaries
    x_min = max(0, min(width - size, x_min))
    y_min = max(0, min(height - size, y_min))

    center_x = x_min + size // 2
    center_y = y_min + size // 2
    radius = size // 2

    if shape == 'rectangle':
        x_max = x_min + size
        y_max = y_min + size
        polygon_coords = [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
            (x_min, y_min)  # Close the polygon
        ]
    elif shape in ['pentagon', 'heptagon', 'hendecagon']:
        if shape == 'pentagon':
            num_points = 5
        elif shape == 'heptagon':
            num_points = 7
        elif shape == 'hendecagon':
            num_points = 11
        else:
            raise ValueError("Invalid shape type.")

        angle_offset = 0  # Can adjust if needed
        polygon_coords = []
        for i in range(num_points):
            angle = angle_offset + i * (2 * np.pi / num_points)
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            polygon_coords.append((x, y))
        polygon_coords.append(polygon_coords[0])  # Close the polygon
    else:
        raise ValueError("Invalid shape type.")

    return polygon_coords

def create_separated_objects(image, image_id, annotation_id, width, height, focus_category_ids, non_focus_category_ids, task):
    """
    Create annotations for separated objects (not overlapping).
    """
    annotations = []
    positions = [
        (0, 0), (width - 200, 0), (0, height - 200), (width - 200, height - 200),
        (width // 2 - 100, height // 2 - 100)
    ]
    random.shuffle(positions)
    categories = focus_category_ids + non_focus_category_ids
    random.shuffle(categories)

    for idx, category_id in enumerate(categories):
        if idx >= len(positions):
            break
        position = positions[idx]
        shape = category_shapes[category_id]
        # Increase size for focus categories
        size = 200 if category_id in focus_category_ids else 100
        polygon_coords = create_polygon_at_position(position, width, height, shape, size=size)
        polygon_shapely = Polygon(polygon_coords)

        # Draw the object
        color = category_colors[category_id]
        cv2.fillPoly(image, [np.array(polygon_coords, dtype=np.int32)], color)

        # Flatten the polygon coordinates
        polygon_flat = [coord for point in polygon_coords for coord in point]

        # Create annotation
        area = polygon_shapely.area
        annotation = UnifiedAnnotation(
            id=annotation_id,
            image_id=image_id,
            category_id=category_id,
            polygon=polygon_flat,
            area=area,
            is_polygon_clipped=False  # Initially False
        )
        annotations.append(annotation)
        annotation_id += 1

    return annotations

def create_synthetic_dataset_separated_objects(num_images: int, output_dir: str, task: str) -> UnifiedDataset:
    """
    Create a synthetic dataset with separated objects.
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset = UnifiedDataset()
    image_id = 1
    annotation_id = 1

    categories = [
        {'id': 1, 'name': 'category_rectangle', 'supercategory': 'synthetic'},
        {'id': 2, 'name': 'category_pentagon', 'supercategory': 'synthetic'},
        {'id': 3, 'name': 'category_heptagon', 'supercategory': 'synthetic'},     # Focus category
        {'id': 4, 'name': 'category_hendecagon', 'supercategory': 'synthetic'},   # Focus category
    ]
    dataset.categories = categories

    focus_categories = ['category_heptagon', 'category_hendecagon']
    focus_category_ids = [cat['id'] for cat in categories if cat['name'] in focus_categories]
    non_focus_category_ids = [cat['id'] for cat in categories if cat['name'] not in focus_categories]

    for i in range(num_images):
        width = 1024
        height = 1024
        image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

        annotations = []

        # Only create separated objects
        annotations += create_separated_objects(image, image_id, annotation_id, width, height, focus_category_ids, non_focus_category_ids, task)

        # Update annotation_id
        if annotations:
            annotation_id = max([ann.id for ann in annotations]) + 1

        # Save image
        image_filename = os.path.join(output_dir, f'synthetic_image_{image_id}.jpg')
        cv2.imwrite(image_filename, image)

        # Create image entry
        image_entry = UnifiedImage(
            id=image_id,
            file_name=image_filename,
            width=width,
            height=height
        )
        dataset.images.append(image_entry)
        dataset.annotations.extend(annotations)

        image_id += 1

    return dataset

def test_crop_augmentation_separated_objects():
    """
    Test the CropAugmentation class with separated objects for the segmentation task.
    """
    # Create output directories
    output_dir = "crop_separated_objects_test"
    os.makedirs(output_dir, exist_ok=True)
    augmented_images_dir = os.path.join(output_dir, "augmented_images")
    os.makedirs(augmented_images_dir, exist_ok=True)
    visualizations_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)

    # Define the task to test
    task = 'segmentation'

    # Create synthetic dataset with separated objects
    dataset = create_synthetic_dataset_separated_objects(num_images=10, output_dir=output_dir, task=task)

    crop_config = {
        'crop_probability': 1.0,
        'num_crops_per_image': 3,
        'margin_percent': 0.05,
        'max_shift_percent': 1.0,
        'shift_steps': 20,
        'max_clipped_area_per_category': None,
        'random_seed': 42,
        'enable_cropping': True,
        'visualize_overlays': True,
        'output_visualizations_dir': visualizations_dir,
        'output_images_dir': augmented_images_dir,
        'padding_color': (0, 0, 0),
        'modes': ['targeted'],
        'focus_categories': ['category_heptagon', 'category_hendecagon'],
    }

    augmenter = CropAugmentation(config=crop_config, task=task)
    augmented_dataset = augmenter.apply(dataset)

    # Assertions
    assert len(augmented_dataset.images) > 0, f"No images were augmented for crop in task {task}."
    assert len(augmented_dataset.annotations) > 0, f"No annotations were augmented for crop in task {task}."

    for img in augmented_dataset.images:
        # Load augmented image
        image = load_image(img.file_name)
        assert image is not None, f"Failed to load augmented image {img.file_name}."

        # Get corresponding annotations
        anns = [ann for ann in augmented_dataset.annotations if ann.image_id == img.id]

        if len(anns) == 0:
            logging.warning(f"No annotations found for image ID {img.id} after augmentation. Skipping this image.")
            continue  # Skip to the next image

        for ann in anns:
            polygon_coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
            assert len(polygon_coords) >= 3, f"Invalid polygon with coordinates: {ann.polygon}"

            # Draw the polygon on the image
            polygon_np = np.array(polygon_coords, dtype=np.int32)
            category_color = category_colors.get(ann.category_id, (0, 0, 0))

            # Determine outline color based on clipping
            if hasattr(ann, 'is_polygon_clipped') and ann.is_polygon_clipped:
                outline_color = (0, 0, 255)  # Red for clipped
            else:
                outline_color = (0, 255, 0)  # Green for not clipped

            # Fill the polygon with its color
            cv2.fillPoly(image, [polygon_np], category_color)
            # Draw the polygon outline
            cv2.polylines(image, [polygon_np], isClosed=True, color=outline_color, thickness=2)

            # Check if the polygon is within image boundaries
            for x, y in polygon_coords:
                assert 0 <= x <= img.width, f"Annotation ID {ann.id} has x-coordinate {x} outside image width."
                assert 0 <= y <= img.height, f"Annotation ID {ann.id} has y-coordinate {y} outside image height."

        # Save the image with drawn annotations
        annotated_image_filename = os.path.join(augmented_images_dir, f"annotated_{os.path.basename(img.file_name)}")
        cv2.imwrite(annotated_image_filename, image)

    logging.info(f"Crop augmentation tests with separated objects completed successfully for task '{task}'.")

    logging.info("All crop augmentation tests completed successfully.")

# Run the test function
if __name__ == "__main__":
    test_crop_augmentation_separated_objects()
