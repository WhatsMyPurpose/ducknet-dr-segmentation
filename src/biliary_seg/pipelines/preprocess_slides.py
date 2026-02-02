import os
import openslide
import numpy as np
from PIL import Image
from typing import Optional
from src.biliary_seg.data.geometry.annotations import Annotations
from src.biliary_seg.data.geometry.rasterize import rasterize_annotations


def parse_slide(
    slide_id: str,
    slide_path: str,
    annotation_path: Optional[str] = None,
    output_dir: str = "./data",
    downsample_factor: int = 8,
    annotation_remove_tag: Optional[str] = "remove",
    annotation_roi_tag: Optional[str] = "roi",
) -> None:
    
    """Preprocess a slide by downsampling, rasterizing annotations, and saving images and masks.
    
    Args:
        slide_id (str): Identifier for the slide.
        slide_path (str): Path to the slide file.
        annotation_path (Optional[str], optional): Path to the annotation file. Defaults to None.
        output_dir (str, optional): Directory to save output images and masks. Defaults to "./data".
        downsample_factor (int, optional): Factor by which to downsample the slide. Defaults to 8.
        annotation_remove_tag (Optional[str], optional): Annotation name tag for regions to remove. Defaults to "remove".
        annotation_roi_tag (Optional[str], optional): Annotation name tag for regions of interest. Defaults to "roi".
    
    """
    
    # Create output directory
    for sub in ("images", "masks"):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
    
    # Get downsampled slide
    slide = openslide.open_slide(slide_path)
    downsample_level = slide.get_best_level_for_downsample(downsample_factor)
    downsampled_slide = slide.read_region((0, 0), downsample_level, slide.level_dimensions[downsample_level])
    
    if not annotation_path:
        # Save downsampled slide as image
        downsampled_slide.save(os.path.join(output_dir, "images", f"{slide_id}.png"))
        return
    
    annotations = Annotations.from_geojson_file(annotation_path)
    
    full_mask = rasterize_annotations(
        slid=slide,
        annotations=annotations,
        downsample_factor=downsample_factor,
    )
    
    # Handle annotations of area to remove from mask and image
    annotations_to_remove = [
        ann for ann in annotations if ann.name.lower() == annotation_remove_tag.lower()
    ]
    if annotations_to_remove:
        remove_mask = rasterize_annotations(
            slide=slide,
            annotations=Annotations(annotations_to_remove),
            downsample_factor=downsample_factor,
        )
        # Remove specified regions from both mask and image
        downsampled_slide_np = np.array(downsampled_slide)
        downsampled_slide_np[remove_mask] = 0
        downsampled_slide = Image.fromarray(downsampled_slide_np)

        full_mask[remove_mask] = 0
    
    # Handle ROI annotations
    roi_annotations = [
        ann for ann in annotations if ann.name.lower() == annotation_roi_tag.lower()
    ]
    if roi_annotations:
        print(f"Found {len(roi_annotations)} ROIs in slide {slide_id}.")
        for idx, roi_ann in enumerate(roi_annotations):
            x0, y0, x1, y1 = roi_ann.get_geometry(downsample_factor).bounds
            img_crop = downsampled_slide.crop((x0, y0, x1, y1))
            img_crop.save(
                os.path.join(output_dir, "images", f"{slide_id}_roi_{idx}.png")
            )
            mask_crop = full_mask[
                int(y0) : int(y1 + 1),
                int(x0) : int(x1 + 1),
            ]
            np.save(
                os.path.join(output_dir, "masks", f"{slide_id}_roi_{idx}.npy"),
                mask_crop,
            )
    else:
        # Save full downsampled slide and mask
        downsampled_slide.save(os.path.join(output_dir, "images", f"{slide_id}.png"))
        np.save(os.path.join(output_dir, "masks", f"{slide_id}.npy"), full_mask)