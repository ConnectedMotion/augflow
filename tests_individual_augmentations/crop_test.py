import os
import copy
import random
import numpy as np
import cv2
import logging
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import affinity
from typing import List, Tuple, Optional

from augflow.augmentations.crop import CropAugmentation
from augflow.utils.unified_format import UnifiedDataset, UnifiedImage, UnifiedAnnotation
from augflow.utils.images import load_image
from augflow.utils.configs import crop_default_config

# Set up logging
log_filename = 'crop_augmentation_test.log'
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def create_polygon_at_position(position, width, height, shape, size=100):
    """
    Create a polygon at the specified position in the image.
    """
    x_min, y_min = position

    # Generate different shapes
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
    elif shape == 'pentagon':
        center_x = x_min + size // 2
        center_y = y_min + size // 2
        radius = size // 2
        num_points = 5
        angle_offset = 0
        polygon_coords = []
        for i in range(num_points):
            angle = angle_offset + i * (2 * np.pi / num_points)
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            polygon_coords.append((x, y))
        polygon_coords.append(polygon_coords[0])
    elif shape == 'irregular':
        num_points = random.randint(6, 10)
        angles = sorted([random.uniform(0, 2 * np.pi) for _ in range(num_points)])
        center_x = x_min + size // 2
        center_y = y_min + size // 2
        radius = size // 2
        polygon_coords = []
        for angle in angles:
            r = radius * random.uniform(0.5, 1.0)
            x = int(center_x + r * np.cos(angle))
            y = int(center_y + r * np.sin(angle))
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            polygon_coords.append((x, y))
        polygon_coords.append(polygon_coords[0])
    else:
        raise ValueError("Invalid shape type. Choose 'rectangle', 'pentagon', or 'irregular'.")

    return polygon_coords


def create_polygon_at_random_position(width, height, shape, size=100):
    """
    Create a polygon at a random position within the image.
    """
    margin = 50
    x_min = random.randint(margin, width - size - margin)
    y_min = random.randint(margin, height - size - margin)
    return create_polygon_at_position((x_min, y_min), width, height, shape, size)


def create_separated_objects(image, image_id, annotation_id, width, height, focus_category_ids, non_focus_category_ids, task):
    """
    Create annotations for separated objects (not overlapping).
    """
    annotations = []
    positions = [
        (50, 50), (width - 150, 50), (50, height - 150), (width - 150, height - 150),
        (width // 2 - 50, height // 2 - 50)
    ]
    random.shuffle(positions)
    shapes = ['rectangle', 'pentagon', 'irregular']
    categories = focus_category_ids + non_focus_category_ids
    random.shuffle(categories)

    for idx, category_id in enumerate(categories):
        if idx >= len(positions):
            break
        position = positions[idx]
        shape = random.choice(shapes) if task == 'segmentation' else 'rectangle'
        polygon_coords = create_polygon_at_position(position, width, height, shape)
        polygon_shapely = Polygon(polygon_coords)

        # Draw the object
        cv2.fillPoly(image, [np.array(polygon_coords, dtype=np.int32)], (128, 0, 128))

        # Flatten the polygon coordinates
        polygon_flat = [coord for point in polygon_coords for coord in point]

        # Create annotation
        area = polygon_shapely.area
        annotation = UnifiedAnnotation(
            id=annotation_id,
            image_id=image_id,
            category_id=category_id,
            polygon=polygon_flat,
            area=area
        )
        annotations.append(annotation)
        annotation_id += 1

    return annotations


def create_overlapping_objects(image, image_id, annotation_id, width, height, focus_category_ids, non_focus_category_ids, task):
    """
    Create annotations for overlapping objects.
    """
    annotations = []
    shapes = ['rectangle', 'pentagon', 'irregular']

    # First half: focus-focus overlaps
    focus_categories = random.sample(focus_category_ids, 2)
    shape1 = random.choice(shapes) if task == 'segmentation' else 'rectangle'
    shape2 = random.choice(shapes) if task == 'segmentation' else 'rectangle'
    polygon_coords1 = create_polygon_at_random_position(width, height, shape1)
    polygon_shapely1 = Polygon(polygon_coords1)

    # Move the second polygon to overlap with the first
    polygon_shapely2 = affinity.translate(polygon_shapely1, xoff=20, yoff=20)
    polygon_coords2 = list(polygon_shapely2.exterior.coords)

    # Draw the objects
    cv2.fillPoly(image, [np.array(polygon_coords1, dtype=np.int32)], (128, 0, 128))
    cv2.fillPoly(image, [np.array(polygon_coords2, dtype=np.int32)], (128, 0, 128))

    # Create annotations
    area1 = polygon_shapely1.area
    area2 = polygon_shapely2.area

    annotation1 = UnifiedAnnotation(
        id=annotation_id,
        image_id=image_id,
        category_id=focus_categories[0],
        polygon=[coord for point in polygon_coords1 for coord in point],
        area=area1
    )
    annotations.append(annotation1)
    annotation_id += 1

    annotation2 = UnifiedAnnotation(
        id=annotation_id,
        image_id=image_id,
        category_id=focus_categories[1],
        polygon=[coord for point in polygon_coords2 for coord in point],
        area=area2
    )
    annotations.append(annotation2)
    annotation_id += 1

    # Second half: focus-nonfocus overlaps
    focus_category = random.choice(focus_category_ids)
    non_focus_category = random.choice(non_focus_category_ids)
    shape1 = random.choice(shapes) if task == 'segmentation' else 'rectangle'
    shape2 = random.choice(shapes) if task == 'segmentation' else 'rectangle'
    polygon_coords1 = create_polygon_at_random_position(width, height, shape1)
    polygon_shapely1 = Polygon(polygon_coords1)

    # Move the second polygon to overlap with the first
    polygon_shapely2 = affinity.translate(polygon_shapely1, xoff=-20, yoff=-20)
    polygon_coords2 = list(polygon_shapely2.exterior.coords)

    # Draw the objects
    cv2.fillPoly(image, [np.array(polygon_coords1, dtype=np.int32)], (128, 0, 128))
    cv2.fillPoly(image, [np.array(polygon_coords2, dtype=np.int32)], (128, 0, 128))

    # Create annotations
    area1 = polygon_shapely1.area
    area2 = polygon_shapely2.area

    annotation1 = UnifiedAnnotation(
        id=annotation_id,
        image_id=image_id,
        category_id=focus_category,
        polygon=[coord for point in polygon_coords1 for coord in point],
        area=area1
    )
    annotations.append(annotation1)
    annotation_id += 1

    annotation2 = UnifiedAnnotation(
        id=annotation_id,
        image_id=image_id,
        category_id=non_focus_category,
        polygon=[coord for point in polygon_coords2 for coord in point],
        area=area2
    )
    annotations.append(annotation2)
    annotation_id += 1

    return annotations


def create_containing_objects(image, image_id, annotation_id, width, height, focus_category_ids, non_focus_category_ids, task):
    """
    Create annotations where one polygon completely contains another.
    """
    annotations = []
    shapes = ['rectangle', 'pentagon', 'irregular']

    # First case: focus polygon contains non-focus polygon
    focus_category = random.choice(focus_category_ids)
    non_focus_category = random.choice(non_focus_category_ids)
    shape_outer = random.choice(shapes) if task == 'segmentation' else 'rectangle'
    shape_inner = random.choice(shapes) if task == 'segmentation' else 'rectangle'

    # Create outer polygon (focus)
    polygon_coords_outer = create_polygon_at_random_position(width, height, shape_outer, size=200)
    polygon_shapely_outer = Polygon(polygon_coords_outer)

    # Create inner polygon (non-focus) inside the outer polygon
    polygon_shapely_inner = polygon_shapely_outer.buffer(-40)
    if polygon_shapely_inner.is_empty or not polygon_shapely_inner.is_valid:
        # Cannot create inner polygon, skip
        return annotations

    # If it's a multipolygon, take the largest
    if isinstance(polygon_shapely_inner, MultiPolygon):
        polygon_shapely_inner = max(polygon_shapely_inner.geoms, key=lambda a: a.area)

    polygon_coords_inner = list(polygon_shapely_inner.exterior.coords)

    # Draw the objects
    cv2.fillPoly(image, [np.array(polygon_coords_outer, dtype=np.int32)], (128, 0, 128))
    cv2.fillPoly(image, [np.array(polygon_coords_inner, dtype=np.int32)], (0, 128, 128))

    # Create annotations
    area_outer = polygon_shapely_outer.area
    area_inner = polygon_shapely_inner.area

    annotation_outer = UnifiedAnnotation(
        id=annotation_id,
        image_id=image_id,
        category_id=focus_category,
        polygon=[coord for point in polygon_coords_outer for coord in point],
        area=area_outer
    )
    annotations.append(annotation_outer)
    annotation_id += 1

    annotation_inner = UnifiedAnnotation(
        id=annotation_id,
        image_id=image_id,
        category_id=non_focus_category,
        polygon=[coord for point in polygon_coords_inner for coord in point],
        area=area_inner
    )
    annotations.append(annotation_inner)
    annotation_id += 1

    # Second case: non-focus polygon contains focus polygon
    focus_category = random.choice(focus_category_ids)
    non_focus_category = random.choice(non_focus_category_ids)
    shape_outer = random.choice(shapes) if task == 'segmentation' else 'rectangle'
    shape_inner = random.choice(shapes) if task == 'segmentation' else 'rectangle'

    # Create outer polygon (non-focus)
    polygon_coords_outer = create_polygon_at_random_position(width, height, shape_outer, size=200)
    polygon_shapely_outer = Polygon(polygon_coords_outer)

    # Create inner polygon (focus) inside the outer polygon
    polygon_shapely_inner = polygon_shapely_outer.buffer(-40)
    if polygon_shapely_inner.is_empty or not polygon_shapely_inner.is_valid:
        # Cannot create inner polygon, skip
        return annotations

    # If it's a multipolygon, take the largest
    if isinstance(polygon_shapely_inner, MultiPolygon):
        polygon_shapely_inner = max(polygon_shapely_inner.geoms, key=lambda a: a.area)

    polygon_coords_inner = list(polygon_shapely_inner.exterior.coords)

    # Draw the objects
    cv2.fillPoly(image, [np.array(polygon_coords_outer, dtype=np.int32)], (128, 0, 128))
    cv2.fillPoly(image, [np.array(polygon_coords_inner, dtype=np.int32)], (0, 128, 128))

    # Create annotations
    area_outer = polygon_shapely_outer.area
    area_inner = polygon_shapely_inner.area

    annotation_outer = UnifiedAnnotation(
        id=annotation_id,
        image_id=image_id,
        category_id=non_focus_category,
        polygon=[coord for point in polygon_coords_outer for coord in point],
        area=area_outer
    )
    annotations.append(annotation_outer)
    annotation_id += 1

    annotation_inner = UnifiedAnnotation(
        id=annotation_id,
        image_id=image_id,
        category_id=focus_category,
        polygon=[coord for point in polygon_coords_inner for coord in point],
        area=area_inner
    )
    annotations.append(annotation_inner)
    annotation_id += 1

    return annotations


def create_synthetic_dataset_with_edge_cases(num_images: int, output_dir: str, task: str) -> UnifiedDataset:
    """
    Create a synthetic dataset with specified edge cases:
    - 30% separated objects
    - 30% overlapping objects (half focus-focus, half focus-nonfocus)
    - 30% containing objects (focus contains non-focus, and vice versa)
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset = UnifiedDataset()
    image_id = 1
    annotation_id = 1

    # Define categories and focus categories
    categories = [
        {'id': 1, 'name': 'category1', 'supercategory': 'synthetic'},  # focus category
        {'id': 2, 'name': 'category2', 'supercategory': 'synthetic'},  # focus category
        {'id': 3, 'name': 'category3', 'supercategory': 'synthetic'},  # non-focus category
        {'id': 4, 'name': 'category4', 'supercategory': 'synthetic'},  # non-focus category
        {'id': 5, 'name': 'category5', 'supercategory': 'synthetic'},  # non-focus category
    ]
    dataset.categories = categories
    focus_categories = ['category1', 'category2']
    focus_category_ids = [cat['id'] for cat in categories if cat['name'] in focus_categories]
    non_focus_category_ids = [cat['id'] for cat in categories if cat['name'] not in focus_categories]

    # Calculate number of images per case
    num_images_separated = int(0.3 * num_images)
    num_images_overlapping = int(0.3 * num_images)
    num_images_containing = int(0.3 * num_images)
    num_images_remaining = num_images - (num_images_separated + num_images_overlapping + num_images_containing)
    num_images_containing += num_images_remaining  # Adjust to match total num_images

    for i in range(num_images):
        width = 800
        height = 600
        image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

        annotations = []

        # Decide which case this image will be
        if i < num_images_separated:
            # Separated objects
            annotations += create_separated_objects(image, image_id, annotation_id, width, height, focus_category_ids, non_focus_category_ids, task)
        elif i < num_images_separated + num_images_overlapping:
            # Overlapping objects
            annotations += create_overlapping_objects(image, image_id, annotation_id, width, height, focus_category_ids, non_focus_category_ids, task)
        else:
            # Containing objects
            annotations += create_containing_objects(image, image_id, annotation_id, width, height, focus_category_ids, non_focus_category_ids, task)

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


def test_crop_augmentation_edge_cases():
    """
    Test the CropAugmentation class with all configurations and edge cases for both detection and segmentation tasks.
    """
    # Create output directories
    output_dir = "crop_augmentation_test"
    os.makedirs(output_dir, exist_ok=True)
    augmented_images_dir = os.path.join(output_dir, "augmented_images")
    os.makedirs(augmented_images_dir, exist_ok=True)
    visualizations_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)

    # Define tasks to test
    tasks = ['segmentation']

    for task in tasks:
        # Create synthetic dataset
        dataset = create_synthetic_dataset_with_edge_cases(
            num_images=10,
            output_dir=output_dir,
            task=task
        )

        crop_configs = [
            # # Non-targeted mode
            # {
            #     'modes': ['non_targeted'],
            #     'focus_categories': [],
            #     'config': {
            #         'crop_probability': 1.0,
            #         'num_crops_per_image': 5,
            #         'crop_size_percent': ((0.1, 0.5), (0.1, 0.5)),
            #         'max_clipped_area_per_category': None,
            #         'random_seed': 42,
            #         'enable_cropping': True,
            #         'visualize_overlays': True,
            #         'output_visualizations_dir': visualizations_dir,
            #         'output_images_dir': augmented_images_dir,
            #         'padding_color': (0, 0, 0),
            #     }
            # },
            # Targeted mode
            {
                'modes': ['targeted'],
                'focus_categories': ['category1', 'category2'],
                'config': {
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
                }
            }
        ]

        for config_info in crop_configs:
            modes = config_info['modes']
            focus_categories = config_info.get('focus_categories', [])
            crop_config = config_info['config']
            augmenter = CropAugmentation(config=crop_config, task=task, modes=modes, focus_categories=focus_categories)
            augmented_dataset = augmenter.apply(dataset)

            # Assertions
            assert len(augmented_dataset.images) > 0, f"No images were augmented for crop in modes {modes} and task {task}."
            # For non-targeted mode, images may have no annotations after cropping
            if 'targeted' in modes:
                assert len(augmented_dataset.annotations) > 0, f"No annotations were augmented for crop in modes {modes} and task {task}."

            for img in augmented_dataset.images:
                # Load augmented image
                image = load_image(img.file_name)
                assert image is not None, f"Failed to load augmented image {img.file_name}."

                # Get corresponding annotations
                anns = [ann for ann in augmented_dataset.annotations if ann.image_id == img.id]

                # For non-targeted mode, annotations may be empty
                if 'targeted' in modes:
                    assert len(anns) > 0, f"No annotations found for image ID {img.id}."

                for ann in anns:
                    polygon_coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
                    assert len(polygon_coords) >= 3, f"Invalid polygon with coordinates: {ann.polygon}"

                    # Check if the polygon is within image boundaries
                    for x, y in polygon_coords:
                        assert 0 <= x <= img.width, f"Annotation ID {ann.id} has x-coordinate {x} outside image width."
                        assert 0 <= y <= img.height, f"Annotation ID {ann.id} has y-coordinate {y} outside image height."

                    # Additional assertions can be added here

                    # Log the area reduction due to clipping if any
                    if ann.is_polygon_clipped:
                        logging.info(f"Annotation ID {ann.id} has area reduction due to clipping.")

                # Visualizations are handled within the augmentation class
                # No need to call visualization functions here

            logging.info(f"Crop augmentation tests completed successfully for modes '{modes}', task '{task}'.")

    logging.info("All crop augmentation tests completed successfully.")


# Run the test function
if __name__ == "__main__":
    test_crop_augmentation_edge_cases()
