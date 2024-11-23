# tests/test_augmentations.py

import os
import copy
import pytest
import random
import numpy as np
import cv2
from shapely.geometry import Polygon
from typing import List, Tuple, Optional, Dict

from augflow.augmentations.affine import AffineAugmentation
from augflow.augmentations.crop import CropAugmentation
from augflow.augmentations.cutout import CutoutAugmentation
from augflow.augmentations.flip import FlipAugmentation
from augflow.augmentations.mosaic import MosaicAugmentation
from augflow.augmentations.rotate import RotateAugmentation
from augflow.augmentations.scale import ScaleAugmentation
from augflow.augmentations.shear import ShearAugmentation
from augflow.augmentations.translate import TranslateAugmentation

from augflow.utils.unified_format import UnifiedDataset, UnifiedImage, UnifiedAnnotation
from augflow.utils.images import load_image, save_image, mosaic_visualize_transformed_overlays

from augflow.utils.configs import (
    affine_default_config,
    crop_default_config,
    cutout_default_config,
    flip_default_config,
    mosaic_default_config,
    rotate_default_config,
    scale_default_config,
    shear_default_config,
    translate_default_config
)

# Synthetic data generation functions
def generate_synthetic_image(width: int, height: int) -> np.ndarray:
    image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    return image

def add_gradient_purple_object(image: np.ndarray, polygon_coords: List[Tuple[int, int]]) -> None:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    polygon = np.array(polygon_coords, dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    
    # Create a gradient purple color
    purple_start = np.array([128, 0, 128], dtype=np.uint8)  # Dark purple
    purple_end = np.array([255, 128, 255], dtype=np.uint8)   # Light purple
    
    gradient = np.linspace(purple_start, purple_end, num=mask.shape[0], axis=0).astype(np.uint8)
    gradient = np.tile(gradient[:, np.newaxis, :], (1, mask.shape[1], 1))
    
    image[mask == 255] = gradient[mask == 255]

def create_non_overlapping_polygon(existing_polygons: List[Polygon], image_width: int, image_height: int, task: str) -> Tuple[List[Tuple[int, int]], Polygon]:
    max_attempts = 100
    for _ in range(max_attempts):
        if task == 'detection':
            # Create a rectangle
            x_min = random.randint(0, image_width // 2)
            y_min = random.randint(0, image_height // 2)
            x_max = random.randint(x_min + 50, image_width)
            y_max = random.randint(y_min + 50, image_height)
            polygon_coords = [
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
                (x_min, y_min)  # Close the polygon
            ]
        else:
            # Create a pentagon
            center_x = random.randint(100, image_width - 100)
            center_y = random.randint(100, image_height - 100)
            radius = random.randint(50, 100)
            num_points = 5
            angle_offset = random.uniform(0, 2 * np.pi)
            polygon_coords = []
            for i in range(num_points):
                angle = angle_offset + i * (2 * np.pi / num_points)
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                # Ensure the points are within the image boundaries
                x = max(0, min(image_width - 1, x))
                y = max(0, min(image_height - 1, y))
                polygon_coords.append((x, y))
            polygon_coords.append(polygon_coords[0])  # Close the polygon

        new_polygon = Polygon(polygon_coords)
        if not new_polygon.is_valid or new_polygon.area == 0:
            continue  # Try again if the polygon is invalid or has zero area

        # Check for overlap with existing polygons
        overlaps = False
        for existing_polygon in existing_polygons:
            if new_polygon.intersects(existing_polygon):
                overlaps = True
                break
        if not overlaps:
            return polygon_coords, new_polygon

    raise ValueError("Could not generate a non-overlapping polygon after multiple attempts.")

def create_synthetic_dataset(num_images: int, task: str, output_dir: str) -> UnifiedDataset:
    """
    Create a synthetic dataset with the specified number of images and task type.
    Each image will have up to four polygons.
    Polygons are closed areas (pentagons for segmentation, rectangles for detection).
    Polygons do not overlap within each image.
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset = UnifiedDataset()
    image_id = 1
    annotation_id = 1
    
    categories = [
        {'id': 1, 'name': 'rectangle_object', 'supercategory': 'shape'},
        {'id': 2, 'name': 'pentagon_object', 'supercategory': 'shape'},
    ]
    dataset.categories = categories
    
    for _ in range(num_images):
        width = random.randint(400, 1200)
        height = random.randint(400, 1200)
        image = generate_synthetic_image(width, height)
        
        num_objects = random.randint(1, 4)  # Up to 4 polygons per image
        annotations = []
        existing_polygons = []
        
        for _ in range(num_objects):
            try:
                polygon_coords, polygon_shapely = create_non_overlapping_polygon(existing_polygons, width, height, task)
                existing_polygons.append(polygon_shapely)
                
                # Draw the object on the image
                add_gradient_purple_object(image, polygon_coords)
                
                # Flatten the polygon coordinates
                polygon_flat = [coord for point in polygon_coords for coord in point]
                
                # Create annotation
                area = polygon_shapely.area
                if task == 'detection':
                    category_id = 1  # Rectangle object
                else:
                    category_id = 2  # Pentagon object
                annotation = UnifiedAnnotation(
                    id=annotation_id,
                    image_id=image_id,
                    category_id=category_id,
                    polygon=polygon_flat,
                    area=area
                )
                annotations.append(annotation)
                annotation_id += 1
            except ValueError:
                print(f"Could not place a non-overlapping polygon in image ID {image_id}.")
                break  # Stop adding polygons to this image if unable to find a non-overlapping one
        
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

# Pytest fixtures
@pytest.fixture(scope="module")
def synthetic_dataset_segmentation(tmp_path_factory):
    """
    Fixture to create a synthetic dataset for segmentation tasks.
    """
    output_dir = tmp_path_factory.mktemp("synthetic_segmentation")
    dataset = create_synthetic_dataset(num_images=5, task='segmentation', output_dir=output_dir)
    return dataset, output_dir

@pytest.fixture(scope="module")
def synthetic_dataset_detection(tmp_path_factory):
    """
    Fixture to create a synthetic dataset for detection tasks.
    """
    output_dir = tmp_path_factory.mktemp("synthetic_detection")
    dataset = create_synthetic_dataset(num_images=5, task='detection', output_dir=output_dir)
    return dataset, output_dir

# Test functions
def test_affine_augmentation(synthetic_dataset_segmentation):
    """
    Test the AffineAugmentation class.
    """
    dataset, output_dir = synthetic_dataset_segmentation
    affine_config = copy.deepcopy(affine_default_config)
    augmenter = AffineAugmentation(affine_config, 'segmentation')  # Positional Arguments
    augmented_dataset = augmenter.apply(dataset)
    
    # Assertions
    assert len(augmented_dataset.images) > 0, "No images were augmented for affine."
    assert len(augmented_dataset.annotations) > 0, "No annotations were augmented for affine."
    
    for img in augmented_dataset.images:
        # Load augmented image
        image = load_image(img.file_name)
        assert image is not None, f"Failed to load augmented image {img.file_name}."
        
        # Get corresponding annotations
        anns = [ann for ann in augmented_dataset.annotations if ann.image_id == img.id]
        for ann in anns:
            polygon_coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
            assert len(polygon_coords) >= 3, f"Invalid polygon with coordinates: {ann.polygon}"
            # Additional assertions can be added here
        
        # Visualize overlays
        visualization_dir = os.path.join(output_dir, 'affine_visualizations')
        os.makedirs(visualization_dir, exist_ok=True)
        visualization_filename = os.path.join(visualization_dir, f"{os.path.basename(img.file_name)}_viz.jpg")
        mosaic_visualize_transformed_overlays(
            transformed_image=image.copy(),
            cleaned_annotations=anns,
            output_visualizations_dir=visualization_dir,
            new_filename=os.path.basename(visualization_filename),
            task='segmentation'
        )

def test_mosaic_augmentation(synthetic_dataset_segmentation):
    """
    Test the MosaicAugmentation class.
    """
    dataset, output_dir = synthetic_dataset_segmentation
    mosaic_config = copy.deepcopy(mosaic_default_config)
    augmenter = MosaicAugmentation(mosaic_config, 'segmentation')  # Positional Arguments
    augmented_dataset = augmenter.apply(dataset)
    
    # Assertions
    assert len(augmented_dataset.images) > 0, "No images were augmented for mosaic."
    assert len(augmented_dataset.annotations) > 0, "No annotations were augmented for mosaic."
    
    for img in augmented_dataset.images:
        # Load augmented image
        image = load_image(img.file_name)
        assert image is not None, f"Failed to load augmented image {img.file_name}."
        
        # Get corresponding annotations
        anns = [ann for ann in augmented_dataset.annotations if ann.image_id == img.id]
        for ann in anns:
            polygon_coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
            assert len(polygon_coords) >= 3, f"Invalid polygon with coordinates: {ann.polygon}"
            # Additional assertions can be added here
        
        # Visualize overlays
        visualization_dir = os.path.join(output_dir, 'mosaic_visualizations')
        os.makedirs(visualization_dir, exist_ok=True)
        visualization_filename = os.path.join(visualization_dir, f"{os.path.basename(img.file_name)}_viz.jpg")
        mosaic_visualize_transformed_overlays(
            transformed_image=image.copy(),
            cleaned_annotations=anns,
            output_visualizations_dir=visualization_dir,
            new_filename=os.path.basename(visualization_filename),
            task='segmentation'
        )

def test_cutout_augmentation(synthetic_dataset_segmentation):
    """
    Test the CutoutAugmentation class.
    """
    dataset, output_dir = synthetic_dataset_segmentation
    cutout_config = copy.deepcopy(cutout_default_config)
    augmenter = CutoutAugmentation(cutout_config, 'segmentation')  # Positional Arguments
    augmented_dataset = augmenter.apply(dataset)
    
    # Assertions
    assert len(augmented_dataset.images) > 0, "No images were augmented for cutout."
    assert len(augmented_dataset.annotations) > 0, "No annotations were augmented for cutout."
    
    for img in augmented_dataset.images:
        # Load augmented image
        image = load_image(img.file_name)
        assert image is not None, f"Failed to load augmented image {img.file_name}."
        
        # Get corresponding annotations
        anns = [ann for ann in augmented_dataset.annotations if ann.image_id == img.id]
        for ann in anns:
            polygon_coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
            assert len(polygon_coords) >= 3, f"Invalid polygon with coordinates: {ann.polygon}"
            # Additional assertions can be added here
        
        # Visualize overlays
        visualization_dir = os.path.join(output_dir, 'cutout_visualizations')
        os.makedirs(visualization_dir, exist_ok=True)
        visualization_filename = os.path.join(visualization_dir, f"{os.path.basename(img.file_name)}_viz.jpg")
        mosaic_visualize_transformed_overlays(
            transformed_image=image.copy(),
            cleaned_annotations=anns,
            output_visualizations_dir=visualization_dir,
            new_filename=os.path.basename(visualization_filename),
            task='segmentation'
        )

def test_translate_augmentation(synthetic_dataset_segmentation):
    """
    Test the TranslateAugmentation class.
    """
    dataset, output_dir = synthetic_dataset_segmentation
    translate_config = copy.deepcopy(translate_default_config)
    augmenter = TranslateAugmentation(translate_config, 'segmentation')  # Positional Arguments
    augmented_dataset = augmenter.apply(dataset)
    
    # Assertions
    assert len(augmented_dataset.images) > 0, "No images were augmented for translate."
    assert len(augmented_dataset.annotations) > 0, "No annotations were augmented for translate."
    
    for img in augmented_dataset.images:
        # Load augmented image
        image = load_image(img.file_name)
        assert image is not None, f"Failed to load augmented image {img.file_name}."
        
        # Get corresponding annotations
        anns = [ann for ann in augmented_dataset.annotations if ann.image_id == img.id]
        for ann in anns:
            polygon_coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
            assert len(polygon_coords) >= 3, f"Invalid polygon with coordinates: {ann.polygon}"
            # Additional assertions can be added here
        
        # Visualize overlays
        visualization_dir = os.path.join(output_dir, 'translate_visualizations')
        os.makedirs(visualization_dir, exist_ok=True)
        visualization_filename = os.path.join(visualization_dir, f"{os.path.basename(img.file_name)}_viz.jpg")
        mosaic_visualize_transformed_overlays(
            transformed_image=image.copy(),
            cleaned_annotations=anns,
            output_visualizations_dir=visualization_dir,
            new_filename=os.path.basename(visualization_filename),
            task='segmentation'
        )

def test_flip_augmentation(synthetic_dataset_segmentation):
    """
    Test the FlipAugmentation class.
    """
    dataset, output_dir = synthetic_dataset_segmentation
    flip_config = copy.deepcopy(flip_default_config)
    augmenter = FlipAugmentation(flip_config,'segmentation') 
    augmented_dataset = augmenter.apply(dataset)
    
    # Assertions
    assert len(augmented_dataset.images) > 0, "No images were augmented for flip."
    assert len(augmented_dataset.annotations) > 0, "No annotations were augmented for flip."
    
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
            # Additional assertions can be added here
        
        # Visualize overlays
        visualization_dir = os.path.join(output_dir, 'flip_visualizations')
        os.makedirs(visualization_dir, exist_ok=True)
        visualization_filename = os.path.join(visualization_dir, f"{os.path.basename(img.file_name)}_viz.jpg")
        mosaic_visualize_transformed_overlays(
            transformed_image=image.copy(),
            cleaned_annotations=anns,
            output_visualizations_dir=visualization_dir,
            new_filename=os.path.basename(visualization_filename),
            task='segmentation'
        )

def test_rotate_augmentation(synthetic_dataset_segmentation):
    """
    Test the RotateAugmentation class.
    """
    dataset, output_dir = synthetic_dataset_segmentation
    rotate_config = copy.deepcopy(rotate_default_config)
    augmenter = RotateAugmentation(rotate_config, 'segmentation')  # Positional Arguments
    augmented_dataset = augmenter.apply(dataset)
    
    # Assertions
    assert len(augmented_dataset.images) > 0, "No images were augmented for rotate."
    assert len(augmented_dataset.annotations) > 0, "No annotations were augmented for rotate."
    
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
            # Additional assertions can be added here
        
        # Visualize overlays
        visualization_dir = os.path.join(output_dir, 'rotate_visualizations')
        os.makedirs(visualization_dir, exist_ok=True)
        visualization_filename = os.path.join(visualization_dir, f"{os.path.basename(img.file_name)}_viz.jpg")
        mosaic_visualize_transformed_overlays(
            transformed_image=image.copy(),
            cleaned_annotations=anns,
            output_visualizations_dir=visualization_dir,
            new_filename=os.path.basename(visualization_filename),
            task='segmentation'
        )

def test_scale_augmentation(synthetic_dataset_segmentation):
    """
    Test the ScaleAugmentation class.
    """
    dataset, output_dir = synthetic_dataset_segmentation
    scale_config = copy.deepcopy(scale_default_config)
    augmenter = ScaleAugmentation(scale_config, 'segmentation')  # Positional Arguments
    augmented_dataset = augmenter.apply(dataset)
    
    # Assertions
    assert len(augmented_dataset.images) > 0, "No images were augmented for scale."
    assert len(augmented_dataset.annotations) > 0, "No annotations were augmented for scale."
    
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
            # Additional assertions can be added here
        
        # Visualize overlays
        visualization_dir = os.path.join(output_dir, 'scale_visualizations')
        os.makedirs(visualization_dir, exist_ok=True)
        visualization_filename = os.path.join(visualization_dir, f"{os.path.basename(img.file_name)}_viz.jpg")
        mosaic_visualize_transformed_overlays(
            transformed_image=image.copy(),
            cleaned_annotations=anns,
            output_visualizations_dir=visualization_dir,
            new_filename=os.path.basename(visualization_filename),
            task='segmentation'
        )

def test_shear_augmentation(synthetic_dataset_detection):
    """
    Test the ShearAugmentation class.
    """
    dataset, output_dir = synthetic_dataset_detection
    shear_config = copy.deepcopy(shear_default_config)
    augmenter = ShearAugmentation(shear_config, 'detection')  # Positional Arguments
    augmented_dataset = augmenter.apply(dataset)
    
    # Assertions
    assert len(augmented_dataset.images) > 0, "No images were augmented for shear."
    assert len(augmented_dataset.annotations) > 0, "No annotations were augmented for shear."
    
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
            # Additional assertions can be added here
        
        # Visualize overlays
        visualization_dir = os.path.join(output_dir, 'shear_visualizations')
        os.makedirs(visualization_dir, exist_ok=True)
        visualization_filename = os.path.join(visualization_dir, f"{os.path.basename(img.file_name)}_viz.jpg")
        mosaic_visualize_transformed_overlays(
            transformed_image=image.copy(),
            cleaned_annotations=anns,
            output_visualizations_dir=visualization_dir,
            new_filename=os.path.basename(visualization_filename),
            task='detection'
        )

def test_crop_augmentation(synthetic_dataset_segmentation):
    """
    Test the CropAugmentation class.
    """
    dataset, output_dir = synthetic_dataset_segmentation
    crop_config = copy.deepcopy(crop_default_config)
    augmenter = CropAugmentation(crop_config, 'segmentation')  # Positional Arguments
    augmented_dataset = augmenter.apply(dataset)
    
    # Assertions
    assert len(augmented_dataset.images) > 0, "No images were augmented for crop."
    assert len(augmented_dataset.annotations) > 0, "No annotations were augmented for crop."
    
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
            # Additional assertions can be added here
        
        # Visualize overlays
        visualization_dir = os.path.join(output_dir, 'crop_visualizations')
        os.makedirs(visualization_dir, exist_ok=True)
        visualization_filename = os.path.join(visualization_dir, f"{os.path.basename(img.file_name)}_viz.jpg")
        mosaic_visualize_transformed_overlays(
            transformed_image=image.copy(),
            cleaned_annotations=anns,
            output_visualizations_dir=visualization_dir,
            new_filename=os.path.basename(visualization_filename),
            task='segmentation'
        )