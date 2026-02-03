import cv2
import openslide
import numpy as np
from rasterio.features import rasterize
from shapely.geometry import Polygon
from shapely.affinity import translate
from typing import Optional, Tuple, Union, Callable
from .annotations import Annotations
from biliary_seg.data.inference.masks import get_slide_bg_mask


def get_polygon_bounding_box(polygon: Polygon) -> Tuple[int, int, int, int]:
    """Calculate the bounding box of a polygon.
    
    Args:
        polygon (Polygon): The input polygon.
    """
    coords = np.array(polygon.exterior.coords, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h

def extract_patch_from_polygon(
    slide: Union[openslide.OpenSlide, np.ndarray], 
    polygon: Polygon, 
    downsample_factor: int = 1
):
    """Extract a patch from a slide image based on the bounding box of a polygon.
    
    Args:
        slide (Union[openslide.OpenSlide, np.ndarray]): The slide image or array.
        polygon (Polygon): The polygon defining the region of interest.
        downsample_factor (int, optional): The downsampling factor.
    """
    x, y, w, h = get_polygon_bounding_box(polygon)
    if isinstance(slide, np.ndarray):
        return slide[y : y + h, x : x + w]
    level = np.log2(downsample_factor).astype(int)
    return slide.read_region(
        (x * downsample_factor, y * downsample_factor), level, (w, h)
    )

def rasterize_annotations(
    slide: Union[openslide.OpenSlide, np.ndarray],
    annotations: Annotations,
    downsample_factor: int,
    bg_mask_func: Optional[Callable] = get_slide_bg_mask,
) -> np.ndarray:
    """
    Rasterize annotations onto a binary mask.
    
    Args:
        slide (Union[openslide.OpenSlide, np.ndarray]): The slide image or array.
        annotations (Annotations): The annotations to rasterize.
        downsample_factor (int): The downsampling factor.
        bg_mask_func (Optional[Callable], optional): Function to generate background mask. Defaults to get_slide_bg_mask.
    
    """
    
    if isinstance(slide, np.ndarray):
        height, width = slide.shape[:2]
        slide_dimensions = (width, height)
    else:
        level = int(np.log2(downsample_factor))
        slide_dimensions = slide.level_dimensions[level]
    
    mask = np.zeros(list(reversed(slide_dimensions)), dtype=bool)
    for annotation in annotations:
        polygons = annotation.get_polygons(downsample_factor)
        for polygon in polygons:
            translated_polygon = translate(
                polygon, xoff=-polygon.bounds[0], yoff=-polygon.bounds[1]
            )
            patch = extract_patch_from_polygon(slide, polygon, downsample_factor)
            patch = np.array(patch)

            polygon_mask = rasterize(
                [translated_polygon],
                (patch.shape[0], patch.shape[1]),
                fill=0,
                dtype="int16",
            ).astype(bool)

            if bg_mask_func is not None:
                background_mask = bg_mask_func(patch)
                polygon_background_mask = np.logical_and(
                    polygon_mask, background_mask
                )
                polygon_roi = np.logical_xor(polygon_mask, polygon_background_mask)
            else:
                polygon_roi = polygon_mask

            mask[
                int(polygon.bounds[1]) : int(polygon.bounds[3] + 1),
                int(polygon.bounds[0]) : int(polygon.bounds[2] + 1),
            ] |= polygon_roi
    return mask