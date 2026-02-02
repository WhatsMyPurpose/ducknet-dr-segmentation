import json
import numpy as np

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import geometry as sg
from typing import Iterable,  List, Union

from copy import deepcopy



class Annotation:

    def __init__(self, geo_json: dict):
        self.geo_json = geo_json

    @staticmethod
    def _downsample_nested_coordinates(lst: Union[list, int], downsample_factor: int):
        """
        Recursively downsample a nested list by a given factor.
        
        Args:
            lst (list or int): The nested list or integer to downsample.
            downsample_factor (int): The factor by which to downsample.
        """
        if not isinstance(lst, Iterable):
            return lst // downsample_factor
        return [
            Annotation._downsample_nested_coordinates(sublst, downsample_factor)
            for sublst in lst
        ]

    def get_coordinates(self, downsample_factor: int) -> list:
        """Get downsampled coordinates of the geometry.
        
        Args:
            downsample_factor (int): The downsampling factor.
        """
        coordinates = deepcopy(self.geo_json["geometry"]["coordinates"])
        return Annotation._downsample_nested_coordinates(coordinates, downsample_factor)

    def get_polygons(self, downsample_factor: int) -> List[Polygon]:
        """Get list of shapely Polygons from the geometry.
        
        Args:
            downsample_factor (int): The downsampling factor.
        """
        coordinates = self.get_coordinates(downsample_factor)
        if self.geo_type == "Polygon":
            return [sg.Polygon(coordinates[0], coordinates[1:])]
        elif self.geo_type == "LineString":
            return [sg.Polygon(coordinates)]
        elif self.geo_type == "MultiPolygon":
            try:
                return list(unary_union(MultiPolygon(coordinates)).geoms)
            except:
                return list(MultiPolygon(coordinates).geoms)

        raise ValueError(f"Unsupported geometry type: {self.geo_type}")

    def get_geometry(self, downsample_factor: int):
        """Get shapely geometry object.
        
        Args:
            downsample_factor (int): The downsampling factor.
        """
        coordinates = self.get_coordinates(downsample_factor)
        geo_type = self.geo_type

        if not hasattr(sg, geo_type):
            raise ValueError(f"Unsupported geometry type: {geo_type}")

        if geo_type == "Polygon":
            return sg.Polygon(coordinates[0], coordinates[1:])
        return getattr(sg, geo_type)(coordinates)

    @property
    def id(self) -> str:
        return self.geo_json["id"]

    @property
    def name(self) -> str:
        properties = self.geo_json["properties"]
        return properties.get(
            "name", properties.get("classification", {}).get("name", "Unknown")
        )

    @property
    def geo_type(self) -> str:
        return self.geo_json["geometry"].get("type")

    @property
    def is_rectangle(self) -> bool:
        coords = np.array(self.get_coordinates(1)).squeeze()
        if coords.shape != (5, 2):
            return False
        if not np.array_equal(coords[0], coords[-1]):
            return False
        return True


class Annotations:

    def __init__(self, annotations: List[Annotation]):
        self.annotations = annotations

    def get_annotations(self) -> Annotation:
        return self.annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index: int) -> Annotation:
        return self.annotations[index]

    def __setitem__(self, index: int, value: Annotation):
        self.annotations[index] = value

    @staticmethod
    def _annotations_from_geojson(geo_json: dict) -> List[Annotation]:
        """Convert geojson dict to list of Annotation objects.
        
        Args:
            geo_json (dict): The geojson dictionary.
        """
        return [Annotation(geo_json) for geo_json in geo_json["features"]]

    @classmethod
    def from_geojson_file(cls, path: str) -> "Annotations":
        """
        Load annotations from a geojson file.
        """
        with open(path) as f:
            geo_json = json.load(f)
        return cls(Annotations._annotations_from_geojson(geo_json))

    def __iter__(self):
        return iter(self.annotations)
