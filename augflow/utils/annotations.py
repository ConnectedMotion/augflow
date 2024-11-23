
import logging
import numpy as np
import copy
from shapely.geometry import Polygon, box
from typing import List, Dict
from .unified_format import UnifiedAnnotation
import cv2




def rotate_polygon(poly, M):
    """
    Rotate a Shapely polygon using the rotation matrix M.
    """
    coords = np.array(poly.exterior.coords)
    ones = np.ones(shape=(coords.shape[0], 1))
    coords_hom = np.hstack([coords, ones])
    rotated_coords = M.dot(coords_hom.T).T
    return Polygon(rotated_coords)


def shear_polygon(poly, M):
    """
    Shear a Shapely polygon using the shear matrix M.
    """
    coords = np.array(poly.exterior.coords)
    ones = np.ones(shape=(coords.shape[0], 1))
    coords_hom = np.hstack([coords, ones])
    sheared_coords = M.dot(coords_hom.T).T
    return Polygon(sheared_coords)




def transform_annotations(annotations: List[UnifiedAnnotation], M: np.ndarray) -> List[UnifiedAnnotation]:
    """
    Apply the affine transformation matrix M to the annotations.

    Args:
        annotations (List[UnifiedAnnotation]): List of annotations to transform.
        M (np.ndarray): Affine transformation matrix (2x3).

    Returns:
        List[UnifiedAnnotation]: List of transformed annotations.
    """
    transformed_annotations = []
    for ann in annotations:
        transformed_ann = copy.deepcopy(ann)
        # Transform polygon
        coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
        if not coords:
            continue
        coords_array = np.array(coords)
        ones = np.ones(shape=(coords_array.shape[0], 1))
        coords_hom = np.hstack([coords_array, ones])
        transformed_coords = M.dot(coords_hom.T).T
        transformed_ann.polygon = [coord for point in transformed_coords for coord in point]
        transformed_annotations.append(transformed_ann)
    return transformed_annotations



def reshape_segmentation(segmentation: List[float]) -> List[tuple]:
    """
    Reshape a flat list of coordinates into a list of (x, y) tuples.
    """
    try:
        if len(segmentation) % 2 != 0:
            raise ValueError("Segmentation list has an odd number of elements.")
        return [tuple(segmentation[i:i+2]) for i in range(0, len(segmentation), 2)]
    except Exception as e:
        logging.error(f"Error reshaping segmentation: {e}")
        return []





def compute_polygon_area(polygon_coords: List[float]) -> float:
    """
    Compute the area of a polygon given its coordinates.

    Args:
        polygon_coords (List[float]): List of polygon coordinates [x1, y1, x2, y2, ..., xn, yn].

    Returns:
        float: Area of the polygon.
    """
    coords = reshape_segmentation(polygon_coords)
    if len(coords) < 3:
        return 0.0  # Not a polygon
    poly = Polygon(coords)
    return poly.area


def get_new_bbox(segmentation):
    """
    Compute the bounding box from the segmentation.
    Segmentation is a list of coordinates [x1, y1, x2, y2, ..., xn, yn]
    """
    coords = np.array(segmentation).reshape(-1, 2)
    if len(coords) == 0:
        return [0, 0, 0, 0]
    x_min = coords[:,0].min()
    y_min = coords[:,1].min()
    x_max = coords[:,0].max()
    y_max = coords[:,1].max()
    return [x_min, y_min, x_max - x_min, y_max - y_min]



def calculate_area_reduction(original_area: float, new_area: float) -> float:
    """
    Calculate the percentage reduction in area, ensuring it's not negative.

    Args:
        original_area (float): Original area.
        new_area (float): New area after transformation.

    Returns:
        float: Area reduction percentage.
    """
    if original_area == 0:
        return 0.0
    reduction = (original_area - new_area) / original_area
    return max(reduction, 0.0)


def clean_annotations(
    transformed_annotations: List[UnifiedAnnotation],
    image_width: int,
    image_height: int,
    max_clipped_area_per_category: Dict[int, float]
) -> List[UnifiedAnnotation]:
    """
    Remove or adjust annotations that are partially or fully outside the image boundaries.

    Args:
        transformed_annotations (List[UnifiedAnnotation]): List of transformed annotations.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        max_clipped_area_per_category (Dict[int, float]): Max allowed area reduction per category.

    Returns:
        List[UnifiedAnnotation]: Cleaned annotations.
    """
    cleaned_annotations = []
    image_boundary = box(0, 0, image_width, image_height)
    
    for ann in transformed_annotations:
        category_id = ann.category_id
        max_allowed_reduction = max_clipped_area_per_category.get(category_id, None)

        if ann.polygon and len(ann.polygon) >= 6:  # At least 3 points (x, y pairs)
            # Convert flat list to list of (x, y) tuples
            coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
            poly = Polygon(coords)
            
            if not poly.is_valid or poly.area == 0:
                logging.warning(f"Annotation ID {ann.id} has an invalid polygon. Skipping.")
                continue  # Skip invalid polygons

            intersection = poly.intersection(image_boundary)
            original_area = poly.area
            new_area = intersection.area

            if intersection.is_empty:
                logging.info(f"Annotation ID {ann.id} is completely outside the image boundaries. Removing.")
                continue  # Skip annotations completely outside the image

            reduction = calculate_area_reduction(original_area, new_area)

            if max_allowed_reduction is not None and reduction > max_allowed_reduction:
                logging.info(f"Annotation ID {ann.id} has area reduction {reduction:.2f} exceeding the maximum allowed {max_allowed_reduction:.2f}. Removing.")
                continue  # Skip annotations with excessive area reduction

            # Update the annotation's polygon to the clipped polygon
            new_coords = list(intersection.exterior.coords)[:-1]  # Exclude the closing coordinate
            if len(new_coords) < 3:
                logging.warning(f"Clipped polygon for Annotation ID {ann.id} has less than 3 points. Removing.")
                continue  # Not a valid polygon after clipping

            # Flatten the list of tuples back to a flat list
            flipped_ann = copy.deepcopy(ann)
            flipped_ann.polygon = [coord for point in new_coords for coord in point]
            flipped_ann.area = new_area
            flipped_ann.is_polygon_clipped = True
            flipped_ann.area_reduction_due_to_clipping = reduction
            cleaned_annotations.append(flipped_ann)
        else:
            logging.warning(f"Annotation ID {ann.id} does not have a valid polygon. Skipping.")
            continue  # Skip annotations without a valid polygon

    return cleaned_annotations



def ensure_axis_aligned_rectangle(coords: List[tuple]) -> List[tuple]:
        """
        Ensure that the polygon is an axis-aligned rectangle.
        If not, approximate it to the bounding box.

        Args:
            coords (List[tuple]): List of (x, y) coordinates.

        Returns:
            List[tuple]: Coordinates of the axis-aligned rectangle, or empty list if not possible.
        """
        x_coords = [pt[0] for pt in coords]
        y_coords = [pt[1] for pt in coords]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        rectangle_coords = [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
            (x_min, y_min)  # Close the polygon
        ]
        return rectangle_coords


def calculate_iou(box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        Each box is a list [x, y, w, h]
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Convert to [x1, y1, x2, y2]
        box1_coords = [x1, y1, x1 + w1, y1 + h1]
        box2_coords = [x2, y2, x2 + w2, y2 + h2]

        # Calculate intersection
        xi1 = max(box1_coords[0], box2_coords[0])
        yi1 = max(box1_coords[1], box2_coords[1])
        xi2 = min(box1_coords[2], box2_coords[2])
        yi2 = min(box1_coords[3], box2_coords[3])
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0
        else:
            return inter_area / union_area



def flip_annotations(annotations: List[UnifiedAnnotation], image_width: int, image_height: int, flip_code: int) -> List[UnifiedAnnotation]:
    """
    Flip annotations based on the flip code.

    Parameters:
        annotations (List[UnifiedAnnotation]): Original annotations.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        flip_code (int): OpenCV flip code.

    Returns:
        List[UnifiedAnnotation]: Flipped annotations.
    """
    flipped_annotations = []
    for ann in annotations:
        flipped_ann = copy.deepcopy(ann)

        # Flip polygon
        flipped_coords = []
        coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))  # Convert flat list to list of (x, y) tuples
        for (x, y) in coords:
            if flip_code == 1:  # Horizontal
                flipped_x = image_width - x
                flipped_y = y
            elif flip_code == 0:  # Vertical
                flipped_x = x
                flipped_y = image_height - y
            elif flip_code == -1:  # Both
                flipped_x = image_width - x
                flipped_y = image_height - y
            else:
                raise ValueError(f"Invalid flip_code: {flip_code}")
            flipped_coords.extend([flipped_x, flipped_y])  # Flatten back to list

        flipped_ann.polygon = flipped_coords

        # Update bbox based on flipped polygon
        if flipped_coords:
            x_coords = flipped_coords[0::2]
            y_coords = flipped_coords[1::2]
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)
            flipped_ann.bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        else:
            # If no polygon, flip bbox directly
            x, y, w, h = ann.bbox
            if flip_code == 1:  # Horizontal
                flipped_x = image_width - (x + w)
                flipped_ann.bbox = [flipped_x, y, w, h]
            elif flip_code == 0:  # Vertical
                flipped_y = image_height - (y + h)
                flipped_ann.bbox = [x, flipped_y, w, h]
            elif flip_code == -1:  # Both
                flipped_x = image_width - (x + w)
                flipped_y = image_height - (y + h)
                flipped_ann.bbox = [flipped_x, flipped_y, w, h]

        # No change in area or category
        flipped_annotations.append(flipped_ann)

    return flipped_annotations



def passes_filter_scale(ann: UnifiedAnnotation, filter_scale: float) -> bool:
        """
        Check if a single annotation meets the minimum filter_scale.

        Args:
            ann (UnifiedAnnotation): The annotation to check.
            filter_scale (float): Minimum width and height for bounding boxes.

        Returns:
            bool: True if annotation meets the filter_scale, False otherwise.
        """
        coords = list(zip(ann.polygon[0::2], ann.polygon[1::2]))
        if not coords:
            return False
        x_coords = [pt[0] for pt in coords]
        y_coords = [pt[1] for pt in coords]
        w = max(x_coords) - min(x_coords)
        h = max(y_coords) - min(y_coords)
        if w < filter_scale or h < filter_scale:
            logging.debug(
                f"Annotation ID {ann.id} has bbox smaller than filter_scale ({w}x{h} < {filter_scale})."
            )
            return False
        return True