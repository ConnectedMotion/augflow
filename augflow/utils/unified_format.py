from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class UnifiedImage:
    id: int
    file_name: str
    width: int
    height: int

@dataclass
class UnifiedAnnotation:
    id: int
    image_id: int
    category_id: int
    # Polygon: List of [x, y] coordinates (required)
    polygon: List[float]
    iscrowd: int = 0
    area: Optional[float] = None
    is_polygon_scaled: bool = False
    is_polygon_clipped: bool = False
    area_reduction_due_to_scaling: Optional[float] = None
    area_reduction_due_to_clipping: Optional[float] = None

@dataclass
class UnifiedDataset:
    images: List[UnifiedImage] = field(default_factory=list)
    annotations: List[UnifiedAnnotation] = field(default_factory=list)
    categories: List[Dict] = field(default_factory=list)
