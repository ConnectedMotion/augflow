import os
import copy
import random
import numpy as np
import cv2
import logging
from shapely.geometry import Polygon
from shapely import affinity
from typing import List, Tuple, Optional

from augflow.augmentations.cutout import CutoutAugmentation
from augflow.utils.unified_format import UnifiedDataset, UnifiedImage, UnifiedAnnotation
from augflow.utils.images import load_image
from augflow.utils.configs import cutout_default_config

# Set up logging
log_filename = 'cutout_augmentation_test.log'
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# Assign colors to categories, avoiding red, green, orange, yellow and their shades
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

# Synthetic data generation functions
def create_polygon_at_position(position: str, width: int, height: int, shape: str) -> List[Tuple[int, int]]:
    """
    Create a polygon (rectangle, pentagon, heptagon, hendecagon) at a specified position in the image.
    """
    size = 100  # Fixed size for test
    margin = 10  # Small margin from the edge

    # Determine the top-left corner of the polygon based on position
    if position == 'top_left':
        x_min = margin
        y_min = margin
    elif position == 'top_right':
        x_min = width - size - margin
        y_min = margin
    elif position == 'bottom_left':
        x_min = margin
        y_min = height - size - margin
    elif position == 'bottom_right':
        x_min = width - size - margin
        y_min = height - size - margin
    elif position == 'center':
        x_min = (width - size) // 2
        y_min = (height - size) // 2
    elif position == 'left_edge':
        x_min = margin
        y_min = (height - size) // 2
    elif position == 'right_edge':
        x_min = width - size - margin
        y_min = (height - size) // 2
    elif position == 'top_edge':
        x_min = (width - size) // 2
        y_min = margin
    elif position == 'bottom_edge':
        x_min = (width - size) // 2
        y_min = height - size - margin
    else:
        x_min = random.randint(margin, width - size - margin)
        y_min = random.randint(margin, height - size - margin)

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
        angle_offset = 0  # Can adjust if needed
        polygon_coords = []
        for i in range(num_points):
            angle = angle_offset + i * (2 * np.pi / num_points)
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            # Ensure the points are within the image boundaries
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            polygon_coords.append((x, y))
        polygon_coords.append(polygon_coords[0])  # Close the polygon
    else:
        raise ValueError("Invalid shape type. Choose 'rectangle', 'pentagon', 'heptagon', or 'hendecagon'.")

    return polygon_coords

def create_synthetic_dataset_with_edge_cases(num_images: int, output_dir: str, task: str, overlap_percentage: float = 0.3) -> UnifiedDataset:
    """
    Create a synthetic dataset with the specified number of images.
    Each image will have annotations for the specified categories and shapes.
    Overlap_percentage determines how much two polygons overlap (values between 0 and 1).
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset = UnifiedDataset()
    image_id = 1
    annotation_id = 1

    categories = [
        {'id': 1, 'name': 'category_rectangle', 'supercategory': 'synthetic'},
        {'id': 2, 'name': 'category_pentagon', 'supercategory': 'synthetic'},
        {'id': 3, 'name': 'category_heptagon', 'supercategory': 'synthetic'},
        {'id': 4, 'name': 'category_hendecagon', 'supercategory': 'synthetic'},
    ]
    dataset.categories = categories

    for _ in range(num_images):
        width = 1024  # Image size larger than 800x800
        height = 1024
        image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

        annotations = []

        # Positions
        positions = [
            'top_left', 'top_right', 'bottom_left', 'bottom_right', 'center',
            'left_edge', 'right_edge', 'top_edge', 'bottom_edge'
        ]
        random.shuffle(positions)

        # Keep track of polygons for overlapping
        existing_polygons = []

        # For each category, place one or more polygons
        for category in categories:
            category_id = category['id']
            shape = category_shapes[category_id]
            color = category_colors[category_id]

            num_polygons = random.randint(1, 3)  # Some images have more than one polygon per category

            for _ in range(num_polygons):
                if positions:
                    position = positions.pop()
                else:
                    position = None  # Random position
                polygon_coords = create_polygon_at_position(position, width, height, shape)
                polygon_shapely = Polygon(polygon_coords)

                # For overlapping polygons between focus and non-focus categories
                # Randomly decide to overlap with an existing polygon
                overlap_chance = 0.5  # 50% chance to create an overlapping polygon
                if existing_polygons and random.random() < overlap_chance:
                    overlapping_poly = random.choice(existing_polygons)
                    # Calculate the required offset to achieve the desired overlap percentage
                    overlap_area_target = overlapping_poly.area * overlap_percentage

                    # Create a scaled version of the overlapping polygon
                    scaled_poly = affinity.scale(overlapping_poly, xfact=1.0, yfact=1.0, origin='center')

                    # Move the scaled polygon over the existing polygon to achieve the desired overlap
                    # This is an approximation; exact overlap area calculation is complex
                    attempts = 0
                    max_attempts = 10
                    while attempts < max_attempts:
                        offset_x = random.randint(-20, 20)
                        offset_y = random.randint(-20, 20)
                        moved_poly = affinity.translate(scaled_poly, xoff=offset_x, yoff=offset_y)
                        intersection_area = overlapping_poly.intersection(moved_poly).area
                        if intersection_area >= overlap_area_target * 0.9 and intersection_area <= overlap_area_target * 1.1:
                            break  # Found a suitable overlap
                        attempts += 1

                    if attempts == max_attempts:
                        # If we couldn't find a suitable overlap, proceed without overlapping
                        logging.info(f"Could not achieve desired overlap percentage after {max_attempts} attempts.")
                        moved_poly = scaled_poly

                    polygon_coords = list(moved_poly.exterior.coords)
                    polygon_shapely = moved_poly

                    # Ensure the points are within the image boundaries
                    polygon_coords = [(max(0, min(width - 1, x)), max(0, min(height - 1, y))) for x, y in polygon_coords]
                    polygon_coords.append(polygon_coords[0])  # Close the polygon
                    polygon_shapely = Polygon(polygon_coords)

                existing_polygons.append(polygon_shapely)

                # Draw the object on the image with its category color
                cv2.fillPoly(image, [np.array(polygon_coords, dtype=np.int32)], color)

                # Initially, we assume the polygon is not clipped
                is_polygon_clipped = False

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
                # We need to store whether the polygon is clipped or not
                annotation.is_polygon_clipped = is_polygon_clipped  # Initially False
                annotations.append(annotation)
                annotation_id += 1

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

# Test function
def test_cutout_augmentation_edge_cases():
    """
    Test the CutoutAugmentation class with all configurations and edge cases for both detection and segmentation tasks.
    """
    # Create output directories
    output_dir = "cutout_augmentation_test"
    os.makedirs(output_dir, exist_ok=True)
    augmented_images_dir = os.path.join(output_dir, "augmented_images")
    os.makedirs(augmented_images_dir, exist_ok=True)
    visualizations_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)

    # Define tasks to test
    tasks = ['segmentation']

    # Define overlap percentages to test
    overlap_percentages = [0.0]

    for task in tasks:
        for overlap_percentage in overlap_percentages:
            # Create synthetic dataset
            dataset = create_synthetic_dataset_with_edge_cases(
                num_images=10,
                output_dir=output_dir,
                task=task,
                overlap_percentage=overlap_percentage
            )

            logging.info(f"Testing with overlap_percentage: {overlap_percentage}")

            cutout_configs = [
                # Targeted mode
                {
                    'modes': ['targeted'],
                    'focus_categories': ['category_heptagon', 'category_hendecagon'],
                    'config': {
                        'cutout_probability': 1.0,
                        'num_augmented_images': len(dataset.images),
                        'margin_percent': 0.05,
                        'max_shift_percent': 1.0,  # Increased to allow larger shifts
                        'shift_steps': 20,         # Increased for finer granularity
                        'max_clipped_area_per_category': None,
                        'random_seed': 42,
                        'enable_cutout': True,
                        'visualize_overlays': True,
                        'output_visualizations_dir': visualizations_dir,
                        'output_images_dir': augmented_images_dir,
                    }
                }
            ]

            for config_info in cutout_configs:
                modes = config_info['modes']
                focus_categories = config_info.get('focus_categories', [])
                cutout_config = config_info['config']
                augmenter = CutoutAugmentation(config=cutout_config, task=task, modes=modes, focus_categories=focus_categories)
                augmented_dataset = augmenter.apply(dataset)

                # Assertions
                assert len(augmented_dataset.images) > 0, f"No images were augmented for cutout in modes {modes} and task {task}."
                assert len(augmented_dataset.annotations) > 0, f"No annotations were augmented for cutout in modes {modes} and task {task}."

                for img in augmented_dataset.images:
                    # Load augmented image
                    image = load_image(img.file_name)
                    assert image is not None, f"Failed to load augmented image {img.file_name}."

                    # Get corresponding annotations
                    anns = [ann for ann in augmented_dataset.annotations if ann.image_id == img.id]
                    assert len(anns) > 0, f"No annotations found for image ID {img.id}."
                    for ann in anns:
                        polygon_coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
                        assert len(polygon_coords) >= 3, f"Invalid polygon with coordinates: {ann.polygon}"

                        # Draw the polygon on the image
                        polygon_np = np.array(polygon_coords, dtype=np.int32)
                        category_color = category_colors[ann.category_id]

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

                logging.info(f"Cutout augmentation tests completed successfully for modes '{modes}', task '{task}', overlap_percentage {overlap_percentage}.")

    logging.info("All cutout augmentation tests completed successfully.")

# Run the test function
if __name__ == "__main__":
    test_cutout_augmentation_edge_cases()
