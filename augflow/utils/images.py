import cv2
import numpy as np
import logging
import os
from shapely.geometry import Polygon
from typing import List, Dict, Optional
from .unified_format import UnifiedAnnotation
from .annotations import reshape_segmentation

def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from a file path.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to read image '{image_path}'.")
        return image
    except Exception as e:
        logging.error(f"Exception occurred while reading image '{image_path}': {e}")
        return None

def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save an image to a file path.
    """
    try:
        success = cv2.imwrite(output_path, image)
        if success:
            logging.info(f"Saved image to '{output_path}'.")
        else:
            logging.error(f"Failed to save image to '{output_path}'.")
        return success
    except Exception as e:
        logging.error(f"Exception occurred while saving image '{output_path}': {e}")
        return False



def generate_affine_transform_matrix(
    image_size,
    rotation_deg=0,
    scale=(1.0, 1.0),
    shear_deg=(0, 0),
    translation=(0, 0),
    rot_point=None
):
    """
    Generate an affine transformation matrix based on rotation, scaling, shearing, and translation.

    Parameters:
    - image_size (tuple): Size of the image as (width, height).
    - rotation_deg (float): Rotation angle in degrees.
    - scale (tuple): Scaling factors as (scale_x, scale_y).
    - shear_deg (tuple): Shear angles in degrees as (shear_x_deg, shear_y_deg).
    - translation (tuple): Translation offsets as (tx, ty).
    - rot_point (tuple, optional): The point (x, y) to rotate around.

    Returns:
    - numpy.ndarray: A 2x3 affine transformation matrix.
    """
    width, height = image_size

    # Determine the rotation center
    if rot_point is not None:
        if not (isinstance(rot_point, tuple) and len(rot_point) == 2):
            raise ValueError("rot_point must be a tuple of two values (x, y).")
        center = rot_point
    else:
        center = (width / 2, height / 2)

    # Rotation matrix (2x3)
    M_rotation = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
    # Convert to 3x3 matrix
    M_rotation = np.vstack([M_rotation, [0, 0, 1]])

    # Shear matrix (3x3)
    if isinstance(shear_deg, tuple) and len(shear_deg) == 2:
        shear_x_deg, shear_y_deg = shear_deg
    else:
        shear_x_deg, shear_y_deg = shear_deg, 0

    shear_x = np.tan(np.radians(shear_x_deg))
    shear_y = np.tan(np.radians(shear_y_deg))

    M_shear = np.array([
        [1, shear_x, 0],
        [shear_y, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    # Scaling matrix (3x3)
    scale_x, scale_y = scale
    M_scale = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    # Translation matrix (3x3)
    tx, ty = translation
    M_translation = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float32)

    # Combine transformations: M = T * S * Sh * R
    M_combined = M_translation @ M_scale @ M_shear @ M_rotation

    # Return the affine transformation matrix (2x3)
    M_combined = M_combined[:2, :]

    return M_combined

def apply_affine_transform(image: np.ndarray, M: np.ndarray, output_size: tuple) -> np.ndarray:
    """
    Apply an affine transformation to the image using matrix M.
    """
    transformed_image = cv2.warpAffine(image, M, output_size, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return transformed_image

def mosaic_visualize_transformed_overlays(
    transformed_image: np.ndarray,
    cleaned_annotations: List[UnifiedAnnotation],
    output_visualizations_dir: str,
    new_filename: str,
    task: str = 'detection'  # Default to 'detection'
):
    """
    Visualize the transformed image with overlaid annotations.

    Args:
        transformed_image (np.ndarray): The transformed image.
        cleaned_annotations (List[UnifiedAnnotation]): List of cleaned annotations.
        output_visualizations_dir (str): Directory to save visualization images.
        new_filename (str): Name of the visualization file to save.
        task (str): 'detection' or 'segmentation' to control drawing.
    """
    # Create a copy of the image to draw overlays
    overlay = transformed_image.copy()
    alpha = 0.6  # Transparency factor

    for ann in cleaned_annotations:
        # Extract polygon coordinates
        coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
        if not coords:
            logging.warning(f"Annotation ID {ann.id} has invalid polygon coordinates. Skipping.")
            continue  # Skip if coordinates are invalid

        # Convert coordinates to integer coordinates for OpenCV
        pts = np.array(coords).reshape(-1, 1, 2).astype(int)

        # Determine color based on scaling and clipping status
        is_scaled = getattr(ann, 'is_polygon_scaled', False)
        is_clipped = getattr(ann, 'is_polygon_clipped', False)

        if not is_scaled and not is_clipped:
            color = (0, 255, 0)  # Green in BGR
        elif is_clipped:
            color = (0, 0, 255)  # Red in BGR
        elif is_scaled:
            color = (0, 165, 255)  # Orange in BGR
        else:
            color = (255, 255, 255)  # Default to white if status is unclear

        # Draw the polygon edges with thicker lines
        thickness = 6  # Thickness of the polygon edges

        # For detection, draw bounding boxes; for segmentation, draw polygons
        if task == 'detection':
            # For detection, compute the bounding box from the polygon
            x_coords = [pt[0] for pt in coords]
            y_coords = [pt[1] for pt in coords]
            x_min = int(min(x_coords))
            y_min = int(min(y_coords))
            x_max = int(max(x_coords))
            y_max = int(max(y_coords))

            # Draw the bounding box
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, thickness)

            # Optionally, add category name or ID
            category_name = str(ann.category_id)
            cv2.putText(
                overlay,
                category_name,
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )
        else:
            # For segmentation, draw the polygon
            cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=thickness)

            # Optionally, add category name or ID near the polygon
            # Calculate centroid for placing the text
            polygon_shape = Polygon(coords)
            if not polygon_shape.is_valid or polygon_shape.is_empty:
                logging.error(f"Invalid polygon for annotation ID {ann.id}. Skipping text placement.")
                continue
            centroid = polygon_shape.centroid
            text_x = int(centroid.x)
            text_y = int(centroid.y)
            # Ensure text is within image boundaries
            text_x = min(max(text_x, 10), transformed_image.shape[1] - 50)
            text_y = min(max(text_y, 10), transformed_image.shape[0] - 10)
            category_name = str(ann.category_id)
            cv2.putText(
                overlay,
                category_name,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, transformed_image, 1 - alpha, 0, transformed_image)

    # Save the visualization image
    visualization_path = os.path.join(output_visualizations_dir, new_filename)
    try:
        success = cv2.imwrite(visualization_path, transformed_image)
        if success:
            logging.info(f"Saved visualization image to '{visualization_path}'.")
        else:
            logging.error(f"Failed to save visualization image '{visualization_path}'.")
    except Exception as e:
        logging.error(f"Exception occurred while saving visualization image '{visualization_path}': {e}")




def save_visualization(
    transformed_image: np.ndarray,
    cleaned_annotations: List[UnifiedAnnotation],
    category_id_to_name: Dict[int, str],
    output_path: str,
    task: str,
    format: str = 'yolo',          # 'coco' or 'yolo'
    bbox_format: str = 'yolo'      # 'yolo', 'xyxy' for YOLO; ignored for COCO.
):
    """
    Save a visualization image with annotations overlaid.

    Args:
        transformed_image (np.ndarray): The image as a NumPy array.
        cleaned_annotations (List[UnifiedAnnotation]): List of annotations associated with the image.
        category_id_to_name (Dict[int, str]): Mapping from category IDs to category names.
        output_path (str): Path where the visualization image will be saved.
        task (str): 'detection' or 'segmentation'.
        format (str): 'coco' or 'yolo'.
        bbox_format (str): 'yolo' or 'xyxy' for YOLO; ignored for COCO.
    """
    overlay = transformed_image.copy()
    alpha = 0.4  # Transparency factor

    for ann in cleaned_annotations:
        # Extract polygon coordinates
        coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
        if not coords:
            logging.warning(f"Annotation ID {ann.id} has invalid polygon coordinates. Skipping.")
            continue  # Skip if coordinates are invalid

        # Convert coordinates to integer coordinates for OpenCV
        pts = np.array(coords).reshape(-1, 1, 2).astype(int)

        # Choose color
        color = (0, 255, 0)  # Green in BGR

        # For detection, draw bounding boxes; for segmentation, draw polygons
        if task == 'detection':
            # For detection, compute the bounding box from the polygon
            x_coords = [pt[0] for pt in coords]
            y_coords = [pt[1] for pt in coords]
            x_min = int(min(x_coords))
            y_min = int(min(y_coords))
            x_max = int(max(x_coords))
            y_max = int(max(y_coords))

            # Draw the bounding box
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, 2)

            # Add category name
            category_name = category_id_to_name.get(ann.category_id, 'Unknown')
            cv2.putText(
                overlay,
                category_name,
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )
        else:
            # For segmentation, draw the polygon
            cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)

            # Optionally, add category name or ID near the polygon
            # Calculate centroid for placing the text
            polygon_shape = Polygon(coords)
            if not polygon_shape.is_valid or polygon_shape.is_empty:
                logging.error(f"Invalid polygon for annotation ID {ann.id}. Skipping text placement.")
                continue
            centroid = polygon_shape.centroid
            text_x = int(centroid.x)
            text_y = int(centroid.y)
            # Ensure text is within image boundaries
            text_x = min(max(text_x, 10), transformed_image.shape[1] - 50)
            text_y = min(max(text_y, 10), transformed_image.shape[0] - 10)
            category_name = category_id_to_name.get(ann.category_id, 'Unknown')
            cv2.putText(
                overlay,
                category_name,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, transformed_image, 1 - alpha, 0, transformed_image)

    # Save the visualization image
    try:
        success = cv2.imwrite(output_path, transformed_image)
        if success:
            logging.info(f"Saved visualization image to '{output_path}'.")
        else:
            logging.error(f"Failed to save visualization image '{output_path}'.")
    except Exception as e:
        logging.error(f"Exception occurred while saving visualization image '{output_path}': {e}")


def crop_image( image, crop_coords, desired_output_size=None, clipping_mode='pad', padding_color=(0, 0, 0)):
        """
        Crop the image based on the specified coordinates and optionally pad to reach the desired output size.
        """
        x, y, w, h = crop_coords
        img_h, img_w = image.shape[:2]

        # Define the crop rectangle
        crop_rect = [x, y, x + w, y + h]  # [x1, y1, x2, y2]

        # Determine overlap with image boundaries
        x1, y1, x2, y2 = crop_rect
        overlap_x1 = max(x1, 0)
        overlap_y1 = max(y1, 0)
        overlap_x2 = min(x2, img_w)
        overlap_y2 = min(y2, img_h)

        # Check if the crop is completely outside the image
        if overlap_x1 >= overlap_x2 or overlap_y1 >= overlap_y2:
            logging.info("Crop is completely outside the image boundaries.")
            return None, None, None, (1, 1)

        # Calculate cropping coordinates
        cropped = image[overlap_y1:overlap_y2, overlap_x1:overlap_x2].copy()

        resize_factors = (1, 1)  # Default scaling factors

        pad_left = overlap_x1 - x1
        pad_top = overlap_y1 - y1
        pad_right = x2 - overlap_x2
        pad_bottom = y2 - overlap_y2

        pad_left_total = max(pad_left, 0)
        pad_top_total = max(pad_top, 0)
        pad_right_total = max(pad_right, 0)
        pad_bottom_total = max(pad_bottom, 0)

        if desired_output_size is None:
            # Handle clipping_mode even when no padding/resizing
            if clipping_mode == 'ignore':
                # If the crop is not fully within the image, skip it
                if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                    logging.info("Crop extends beyond image boundaries and clipping_mode is 'ignore'. Skipping crop.")
                    return None, None, None, (1, 1)
                else:
                    return cropped, pad_left_total, pad_top_total, (1, 1)
            elif clipping_mode == 'pad':
                # Pad the cropped image to maintain the original requested crop size
                if pad_left_total > 0 or pad_top_total > 0 or pad_right_total > 0 or pad_bottom_total > 0:
                    cropped = cv2.copyMakeBorder(
                        cropped,
                        pad_top_total,
                        pad_bottom_total,
                        pad_left_total,
                        pad_right_total,
                        borderType=cv2.BORDER_CONSTANT,
                        value=padding_color
                    )
                return cropped, pad_left_total, pad_top_total, (1, 1)
            else:
                logging.error(f"Unsupported clipping_mode '{clipping_mode}' when desired_output_size is None. Skipping crop.")
                return None, None, None, (1, 1)

        # Desired output size is specified
        desired_w, desired_h = desired_output_size

        # Calculate required padding to reach desired output size
        current_w = cropped.shape[1]
        current_h = cropped.shape[0]
        pad_right_desired = desired_w - current_w - pad_left_total
        pad_bottom_desired = desired_h - current_h - pad_top_total

        pad_right_total += max(pad_right_desired, 0)
        pad_bottom_total += max(pad_bottom_desired, 0)

        # Handle cases where padding is negative (i.e., crop is larger than desired size)
        if pad_right_desired < 0 or pad_bottom_desired < 0:
            if clipping_mode == 'resize_crop':
                # Calculate scaling factors
                scale_x = desired_w / (current_w + pad_left_total + pad_right_total)
                scale_y = desired_h / (current_h + pad_top_total + pad_bottom_total)
                resize_factors = (scale_x, scale_y)
                # Resize the cropped image to desired size
                cropped = cv2.resize(cropped, (desired_w, desired_h), interpolation=cv2.INTER_LINEAR)
                pad_left_total = int(pad_left_total * scale_x)
                pad_top_total = int(pad_top_total * scale_y)
            else:
                logging.info("Cropped image exceeds desired output size and clipping_mode is not 'resize_crop'. Skipping crop.")
                return None, None, None, (1, 1)
        else:
            if clipping_mode == 'pad':
                # Pad the cropped image to reach desired output size
                cropped = cv2.copyMakeBorder(
                    cropped,
                    pad_top_total,
                    pad_bottom_total,
                    pad_left_total,
                    pad_right_total,
                    borderType=cv2.BORDER_CONSTANT,
                    value=padding_color
                )
            elif clipping_mode == 'resize_crop':
                # Calculate scaling factors
                scale_x = desired_w / (current_w + pad_left_total + pad_right_total)
                scale_y = desired_h / (current_h + pad_top_total + pad_bottom_total)
                resize_factors = (scale_x, scale_y)
                # Resize the cropped image to the desired output size
                cropped = cv2.resize(cropped, (desired_w, desired_h), interpolation=cv2.INTER_LINEAR)
                pad_left_total = int(pad_left_total * scale_x)
                pad_top_total = int(pad_top_total * scale_y)
            elif clipping_mode == 'ignore':
                # Skip this crop as it cannot be padded to desired size without exceeding boundaries
                logging.info("Cannot pad to desired output size and clipping_mode is 'ignore'. Skipping crop.")
                return None, None, None, (1, 1)
            else:
                logging.error(f"Unsupported clipping_mode '{clipping_mode}'. Skipping crop.")
                return None, None, None, (1, 1)

        return cropped, pad_left_total, pad_top_total, resize_factors



def scale_image(image: np.ndarray, scale_x: float, scale_y: float, interpolation: int) -> np.ndarray:
        """
        Scale the image based on the provided scale factors.

        Args:
            image (np.ndarray): Original image.
            scale_x (float): Scaling factor for width.
            scale_y (float): Scaling factor for height.
            interpolation (int): OpenCV interpolation method.

        Returns:
            np.ndarray: Scaled image.
        """
        height, width = image.shape[:2]
        new_width = max(1, int(width * scale_x))
        new_height = max(1, int(height * scale_y))
        scaled_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        logging.debug(f"Image scaled to: width={new_width}, height={new_height}")
        return scaled_image