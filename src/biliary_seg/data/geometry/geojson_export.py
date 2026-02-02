import cv2
import json
import numpy as np

def convert_contour_to_coords(contour: np.ndarray, scale_factor: int = 8) -> list:
    """Convert contour points to GeoJSON coordinates with upscaling
    
    Args:
        contour (np.ndarray): Contour points from cv2.findContours
        scale_factor (int, optional): Factor to scale up coordinates. Defaults to 8.
    """
    # Extract coordinates and flip y-axis
    coords = []
    for point in contour:
        x, y = point[0]
        # Apply scale factor and flip y-axis
        scaled_x, scaled_y = int(x * scale_factor), int(y * scale_factor)
        coords.append([scaled_x, scaled_y])

    # GeoJSON polygons must be closed (first and last points must be the same)
    if not np.array_equal(coords[0], coords[-1]):
        coords.append(coords[0])

    return coords


def mask_to_polygons(mask: np.ndarray, scale_factor: int = 8, min_points: int = 9) -> list:
    """
    Extract geographic coordinates from a binary mask with both outer and inner contours.

    Args:
        mask (np.ndarray): Binary mask to extract contours from
        scale_factor (int, optional): Factor to scale up coordinates. Defaults to 8.
        min_points (int, optional): Minimum number of points for a contour to be considered. Defaults to 9.

    Returns:
        list: List of polygon coordinate arrays for GeoJSON
    """
    # Find contours with hierarchy to get both outer and inner boundaries
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    polygon_coords = []

    # hierarchy[0] contains [next, prev, first_child, parent] for each contour
    for i, contour in enumerate(contours):
        # Skip if contour has too few points
        if len(contour) < min_points:
            continue

        # If it's an outer contour (parent = -1)
        if hierarchy[0][i][3] == -1:
            exterior_coords = convert_contour_to_coords(
                contour, scale_factor
            )

            # Find all inner contours (children)
            interior_coords = []
            child_idx = hierarchy[0][i][2]
            while child_idx != -1:
                child_contour = contours[child_idx]
                child_coords = convert_contour_to_coords(
                    child_contour, scale_factor
                )
                interior_coords.append(child_coords)
                child_idx = hierarchy[0][child_idx][0]  # Get next sibling

            if interior_coords:
                # Polygon with holes
                polygon_coords.append([exterior_coords] + interior_coords)
            else:
                # Simple polygon without holes
                polygon_coords.append([exterior_coords])

    return polygon_coords


def mask_to_geojson(
    mask: np.ndarray, classification: str = "Biliary", name: str = "BP", scale_factor=8, color="red"
) -> dict:
    """Create GeoJSON from binary mask with upscaling"""
    # Extract coordinates
    polygon_coords = mask_to_polygons(mask, scale_factor)

    # Define colors
    if color == "red":
        rgb_color = (255, 0, 0)
    elif color == "blue":
        rgb_color = (0, 0, 255)
    elif color == "green":
        rgb_color = (0, 255, 0)
    elif color == "yellow":
        rgb_color = (255, 255, 0)
    elif color == "purple":
        rgb_color = (255, 0, 255)
    elif color == "cyan":
        rgb_color = (0, 255, 255)
    elif color == "orange":
        rgb_color = (255, 165, 0)
    else:
        rgb_color = (255, 0, 0)

    # Create features
    features = []
    for i, coords in enumerate(polygon_coords):
        feature = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": coords},
            "properties": {
                "name": f"{name}_{i}",
                "classification": classification,
                "color": list(rgb_color),
            },
        }
        features.append(feature)

    # Create GeoJSON structure
    geojson = {"type": "FeatureCollection", "features": features}

    return geojson

def save_geojson(geojson: dict, filepath: str):
    """Save GeoJSON dictionary to a file.
    
    Args:
        geojson (dict): The GeoJSON data.
        filepath (str): The path to save the GeoJSON file.
    """
    with open(filepath, 'w') as f:
        json.dump(geojson, f)