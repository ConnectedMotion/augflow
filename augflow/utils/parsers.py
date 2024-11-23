import os
import json
import logging
from typing import List, Dict, Tuple
from .unified_format import UnifiedDataset, UnifiedImage, UnifiedAnnotation
from shapely.geometry import Polygon
import glob
import cv2
import yaml



class CocoParser:
    def __init__(self):
        logging.info("CocoParser initialized.")

    def load(self, dataset_path: str) -> Tuple[UnifiedDataset, str]:
        # Search for any .json file in dataset_path and subdirectories
        json_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith('.json'):
                    json_files.append(os.path.join(root, file))

        if not json_files:
            raise FileNotFoundError(f"No JSON annotations file found in '{dataset_path}'.")

        # Assuming the first JSON file is the annotations file
        annotations_path = json_files[0]
        logging.info(f"Using annotations file '{annotations_path}'.")

        with open(annotations_path, 'r') as f:
            coco = json.load(f)

        dataset = UnifiedDataset()
        dataset.categories = coco.get('categories', [])

        # Create a mapping from image ID to file name
        image_id_to_file_name = {}
        for img in coco.get('images', []):
            image_id_to_file_name[img['id']] = img['file_name']

        # Search for image files in dataset_path and subdirectories
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(dataset_path, '**', ext), recursive=True))

        # Create a mapping from file name to full path
        file_name_to_full_path = {os.path.basename(f): f for f in image_files}

        for img_id, img_file_name in image_id_to_file_name.items():
            # Get full image path
            img_full_path = file_name_to_full_path.get(img_file_name)
            if not img_full_path:
                logging.warning(f"Image file '{img_file_name}' not found in dataset. Skipping.")
                continue

            # Assuming width and height are present in the image info
            img_info = next((img for img in coco.get('images', []) if img['id'] == img_id), None)
            if not img_info:
                logging.warning(f"No image info found for image ID '{img_id}'. Skipping.")
                continue

            image = UnifiedImage(
                id=img_id,
                file_name=img_full_path,
                width=img_info['width'],
                height=img_info['height']
            )
            dataset.images.append(image)

        segmentation_present = False  # Flag to detect if any segmentation is present

        for ann in coco.get('annotations', []):
            segmentation = ann.get('segmentation', [])
            if segmentation and len(segmentation) > 0 and segmentation[0]:
                segmentation_present = True
                # Use the segmentation polygon
                polygon = segmentation[0]
            else:
                # Convert bbox to polygon
                x, y, w, h = ann['bbox']
                polygon = [
                    x, y,
                    x + w, y,
                    x + w, y + h,
                    x, y + h
                ]

            unified_ann = UnifiedAnnotation(
                id=ann['id'],
                image_id=ann['image_id'],
                category_id=ann['category_id'],
                polygon=polygon,
                iscrowd=ann.get('iscrowd', 0),
                area=ann.get('area', 0.0)
            )
            dataset.annotations.append(unified_ann)

        if segmentation_present:
            task = 'segmentation'
        else:
            task = 'detection'

        logging.info(f"Loaded {len(dataset.images)} images and {len(dataset.annotations)} annotations from COCO format with task '{task}'.")
        return dataset, task

    def save(self, dataset: UnifiedDataset, output_path: str, task: str):
        # Convert UnifiedDataset to COCO format
        coco = {
            'images': [],
            'annotations': [],
            'categories': dataset.categories
        }

        for img in dataset.images:
            coco_image = {
                'id': img.id,
                'file_name': os.path.basename(img.file_name),
                'width': img.width,
                'height': img.height
            }
            coco['images'].append(coco_image)

        for ann in dataset.annotations:
            # Convert polygon to segmentation and bbox
            segmentation = [ann.polygon]
            x_coords = ann.polygon[0::2]
            y_coords = ann.polygon[1::2]
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

            coco_ann = {
                'id': ann.id,
                'image_id': ann.image_id,
                'category_id': ann.category_id,
                'bbox': bbox,  # [x_min, y_min, width, height]
                'area': ann.area,
                'iscrowd': ann.iscrowd
            }
            if task == 'segmentation':
                coco_ann['segmentation'] = segmentation
            else:
                # For detection task, do not include segmentation
                coco_ann['segmentation'] = []
            coco['annotations'].append(coco_ann)

        # Save to annotations.json
        os.makedirs(output_path, exist_ok=True)
        annotations_file = os.path.join(output_path, 'annotations.json')
        with open(annotations_file, 'w') as f:
            json.dump(coco, f, indent=4)
        logging.info(f"Saved COCO annotations to '{annotations_file}'.")

        # Save images
        images_output_dir = os.path.join(output_path, 'images')
        os.makedirs(images_output_dir, exist_ok=True)
        for img in dataset.images:
            # Copy image to output_path/images
            src_path = img.file_name
            dst_path = os.path.join(images_output_dir, os.path.basename(img.file_name))
            if not os.path.exists(dst_path):
                import shutil
                try:
                    shutil.copy2(src_path, dst_path)
                    logging.info(f"Copied image '{src_path}' to '{dst_path}'.")
                except Exception as e:
                    logging.error(f"Failed to copy image '{src_path}' to '{dst_path}': {e}")





class YoloParser:
    def __init__(self):
        logging.info("YoloParser initialized.")

    def load(self, dataset_path: str) -> Tuple[UnifiedDataset, str]:
        # Find data.yaml file in dataset_path or one level above
        data_yaml_path = None
        for root, dirs, files in os.walk(os.path.abspath(os.path.join(dataset_path, '..'))):
            if 'data.yaml' in files:
                data_yaml_path = os.path.join(root, 'data.yaml')
                break
        if not data_yaml_path:
            for root, dirs, files in os.walk(dataset_path):
                if 'data.yaml' in files:
                    data_yaml_path = os.path.join(root, 'data.yaml')
                    break

        if not data_yaml_path or not os.path.exists(data_yaml_path):
            raise FileNotFoundError(f"YOLO data.yaml file not found in '{dataset_path}' or its parent directory.")

        with open(data_yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)

        num_classes = data_yaml.get('nc', 0)
        class_names = data_yaml.get('names', [])
        categories = [{'id': idx, 'name': name, 'supercategory': 'none'} for idx, name in enumerate(class_names)]

        dataset = UnifiedDataset()
        dataset.categories = categories

        # Find all image files in dataset_path and subdirectories
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(dataset_path, '**', ext), recursive=True))

        # Mapping from image base name to full path
        image_base_name_to_full_path = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}

        # Find all label files in dataset_path and subdirectories
        label_files = glob.glob(os.path.join(dataset_path, '**', '*.txt'), recursive=True)
        # Exclude data.yaml if present
        label_files = [lf for lf in label_files if os.path.basename(lf) != 'data.yaml']

        # Mapping from label base name to full path
        label_base_name_to_full_path = {os.path.splitext(os.path.basename(f))[0]: f for f in label_files}

        # Determine task type
        segmentation_detected = False
        detection_detected = False

        image_id = 1
        annotation_id = 1

        for base_name, img_path in image_base_name_to_full_path.items():
            # Assuming image_id is unique per image
            image = cv2.imread(img_path)
            if image is None:
                logging.error(f"Failed to read image '{img_path}'. Skipping.")
                continue
            height, width = image.shape[:2]
            img = UnifiedImage(
                id=image_id,
                file_name=img_path,
                width=width,
                height=height
            )
            dataset.images.append(img)

            # Find corresponding label file
            label_file = label_base_name_to_full_path.get(base_name)
            if not label_file:
                logging.warning(f"Label file for image '{img_path}' not found. Skipping annotations for this image.")
                image_id += 1
                continue

            with open(label_file, 'r') as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        # Less than 5 parts, must be segmentation
                        segmentation_detected = True
                        if len(parts) < 3:
                            logging.warning(f"Invalid segmentation annotation in '{label_file}': {line}")
                            continue
                        class_id = int(parts[0])
                        coords = list(map(float, parts[1:]))
                        # Convert normalized coordinates to absolute
                        polygon = [coord * width if i % 2 == 0 else coord * height for i, coord in enumerate(coords)]
                        area = Polygon(zip(polygon[0::2], polygon[1::2])).area
                        ann = UnifiedAnnotation(
                            id=annotation_id,
                            image_id=image_id,
                            category_id=class_id,
                            polygon=polygon,
                            iscrowd=0,
                            area=area
                        )
                        dataset.annotations.append(ann)
                        annotation_id += 1
                    else:
                        # 5 or more parts, could be detection or segmentation
                        if len(parts) == 5:
                            # Assume detection
                            detection_detected = True
                            class_id, x_center_norm, y_center_norm, width_norm, height_norm = parts
                            class_id = int(class_id)
                            x_center = float(x_center_norm) * width
                            y_center = float(y_center_norm) * height
                            w = float(width_norm) * width
                            h = float(height_norm) * height
                            x_min = x_center - w / 2
                            y_min = y_center - h / 2
                            x_max = x_center + w / 2
                            y_max = y_center + h / 2
                            polygon = [
                                x_min, y_min,
                                x_max, y_min,
                                x_max, y_max,
                                x_min, y_max
                            ]
                            area = w * h
                            ann = UnifiedAnnotation(
                                id=annotation_id,
                                image_id=image_id,
                                category_id=class_id,
                                polygon=polygon,
                                iscrowd=0,
                                area=area
                            )
                            dataset.annotations.append(ann)
                            annotation_id += 1
                        else:
                            # Assume segmentation
                            segmentation_detected = True
                            class_id = int(parts[0])
                            coords = list(map(float, parts[1:]))
                            # Convert normalized coordinates to absolute
                            polygon = [coord * width if i % 2 == 0 else coord * height for i, coord in enumerate(coords)]
                            area = Polygon(zip(polygon[0::2], polygon[1::2])).area
                            ann = UnifiedAnnotation(
                                id=annotation_id,
                                image_id=image_id,
                                category_id=class_id,
                                polygon=polygon,
                                iscrowd=0,
                                area=area
                            )
                            dataset.annotations.append(ann)
                            annotation_id += 1

            image_id += 1

        if segmentation_detected and not detection_detected:
            task = 'segmentation'
        elif detection_detected and not segmentation_detected:
            task = 'detection'
        else:
            task = 'detection'  # Default to detection if both are present

        logging.info(f"Loaded {len(dataset.images)} images and {len(dataset.annotations)} annotations from YOLO format with task '{task}'.")
        return dataset, task

    def save(self, dataset: UnifiedDataset, output_path: str, task: str):
        """
        Save the UnifiedDataset to YOLO format.

        Args:
            dataset (UnifiedDataset): The dataset to save.
            output_path (str): Path to save the YOLO dataset.
            task (str): 'detection' or 'segmentation'
        """
        images_dir = os.path.join(output_path, 'images')
        labels_dir = os.path.join(output_path, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Create a mapping from original category IDs to zero-based indices
        category_id_to_index = {cat['id']: idx for idx, cat in enumerate(sorted(dataset.categories, key=lambda x: x['id']))}
        # Update categories to have IDs starting from 0
        updated_categories = [{'id': idx, 'name': cat['name'], 'supercategory': cat.get('supercategory', 'none')} for idx, cat in enumerate(sorted(dataset.categories, key=lambda x: x['id']))]

        # Save data.yaml
        data_yaml = {
            'nc': len(updated_categories),
            'names': [cat['name'] for cat in updated_categories]
        }
        data_yaml_path = os.path.join(output_path, 'data.yaml')
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        logging.info(f"Saved YOLO data.yaml to '{data_yaml_path}'.")

        for img in dataset.images:
            src_image_path = img.file_name
            dst_image_path = os.path.join(images_dir, os.path.basename(src_image_path))
            if not os.path.exists(dst_image_path):
                import shutil
                try:
                    shutil.copy2(src_image_path, dst_image_path)
                    logging.info(f"Copied image '{src_image_path}' to '{dst_image_path}'.")
                except Exception as e:
                    logging.error(f"Failed to copy image '{src_image_path}' to '{dst_image_path}': {e}")

            # Find annotations for this image
            anns = [ann for ann in dataset.annotations if ann.image_id == img.id]

            # Always create a label file, even if there are no annotations
            label_file = os.path.join(labels_dir, os.path.splitext(os.path.basename(src_image_path))[0] + '.txt')
            with open(label_file, 'w') as lf:
                if not anns:
                    # Write nothing to create an empty label file
                    pass
                else:
                    for ann in anns:
                        # Map the category ID to zero-based index
                        class_id = category_id_to_index[ann.category_id]
                        if task == 'detection':
                            # Convert polygon to bbox
                            x_coords = ann.polygon[0::2]
                            y_coords = ann.polygon[1::2]
                            x_min = min(x_coords)
                            y_min = min(y_coords)
                            x_max = max(x_coords)
                            y_max = max(y_coords)
                            w = x_max - x_min
                            h = y_max - y_min
                            x_center = x_min + w / 2
                            y_center = y_min + h / 2
                            x_center_norm = x_center / img.width
                            y_center_norm = y_center / img.height
                            w_norm = w / img.width
                            h_norm = h / img.height
                            lf.write(f"{class_id} {x_center_norm} {y_center_norm} {w_norm} {h_norm}\n")
                        elif task == 'segmentation':
                            # Normalize polygon coordinates
                            segmentation_norm = []
                            for i in range(0, len(ann.polygon), 2):
                                x_norm = ann.polygon[i] / img.width
                                y_norm = ann.polygon[i+1] / img.height
                                segmentation_norm.extend([x_norm, y_norm])
                            segmentation_str = ' '.join(map(str, segmentation_norm))
                            lf.write(f"{class_id} {segmentation_str}\n")
            logging.info(f"Saved label file to '{label_file}'.")

        # Update the dataset's categories to have IDs starting from 0
        dataset.categories = updated_categories
