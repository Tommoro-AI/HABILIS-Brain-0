# Modifications by Tommoro: public runtime extraction and adaptation.
# This file differs from its original source; see manifest.json for source hashes.

from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict


class Point(TypedDict):
    point: List[float] # [x, y] coordinates
    occluded: Optional[bool] = False


class PointTrajectoryEntry(TypedDict):
    frame: int # frame index
    time: float # time in seconds
    points: Dict[str, Point] # object_id -> {'point': [x, y], 'occluded': bool}

