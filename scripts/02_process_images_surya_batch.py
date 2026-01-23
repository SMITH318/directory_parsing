#!/usr/bin/env python3
"""
Step 2: Robust Streaming OCR using Surya with Batch Processing
- Optimized for Google Colab with GPU batch processing
- Streaming output to JSONL (safe for Colab runtime disconnects)
- Handles large snippets without OOM crashes
- Uses Surya OCR engine for better accuracy on historical documents
"""

# Fix CUDA memory fragmentation - MUST be before torch import
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import gc
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Tuple

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

# =============================================================================
# CONFIGURATION - Adjust these for your Colab environment
# =============================================================================

# Batch size for GPU processing
# - Colab Free (T4, 15GB VRAM): 4-8 (conservative for large newspaper images)
# - Colab Pro (A100, 40GB VRAM): 16-32
# - M1 Mac: 4-8
# Note: Large historical newspaper snippets use MORE memory than typical images
BATCH_SIZE = 4

# Set to True to see progress for each image in batch
VERBOSE = True

# =============================================================================


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32, np.float32, np.float64)):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def init_surya():
    """Initialize Surya OCR predictors."""
    print("Initializing Surya OCR...")
    print("(First run will download model weights ~2GB)")
    
    # Foundation predictor is required for recognition
    foundation_predictor = FoundationPredictor()
    recognition_predictor = RecognitionPredictor(foundation_predictor)
    detection_predictor = DetectionPredictor()
    
    print("Surya OCR initialized successfully")
    return detection_predictor, recognition_predictor


def load_image_safe(img_path: str) -> Image.Image | None:
    """Safely load an image, returning None on failure."""
    try:
        img = Image.open(img_path)
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        print(f"  Warning: Could not load {img_path}: {e}")
        return None


def process_batch(
    batch_images: List[Image.Image],
    batch_metadata: List[Dict[str, Any]],
    detection_predictor: DetectionPredictor,
    recognition_predictor: RecognitionPredictor
) -> List[Dict[str, Any]]:
    """
    Process a batch of images with Surya OCR.
    
    Args:
        batch_images: List of PIL Images
        batch_metadata: List of metadata dicts for each image
        detection_predictor: Surya detection predictor
        recognition_predictor: Surya recognition predictor
    
    Returns:
        List of output entries ready for JSONL
    """
    entries = []
    
    try:
        # Run batch OCR
        predictions = recognition_predictor(
            batch_images,
            det_predictor=detection_predictor
        )
        
        # Process each result with its corresponding metadata
        for pred, meta in zip(predictions, batch_metadata):
            pdf_name = meta['pdf_name']
            page_num = meta['page_num']
            column = meta['column']
            x_off = meta['x_offset']
            y_off = meta['y_offset']
            
            for text_line in pred.text_lines:
                # Surya bbox is (x1, y1, x2, y2) format
                # Convert to [x, y, width, height] to match PaddleOCR output
                bbox = text_line.bbox
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                
                # Apply offsets to get global coordinates
                real_x = x1 + x_off
                real_y = y1 + y_off
                width = x2 - x1
                height = y2 - y1
                
                entry = {
                    "pub": pdf_name,
                    "page": page_num,
                    "col": column,
                    "text": text_line.text.strip(),
                    "conf": round(float(text_line.confidence), 4),
                    "bbox": [int(real_x), int(real_y), int(width), int(height)]
                }
                entries.append(entry)
        
        # Clear predictions from memory
        del predictions
                
    except Exception as e:
        print(f"  Batch processing error: {e}")
    
    finally:
        # Always try to clean up GPU memory after batch
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        gc.collect()
    
    return entries


def collect_all_snippets(all_metadata: List[Dict]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Collect all snippets from metadata into a flat list.
    
    Returns:
        List of (image_path, metadata_dict) tuples
    """
    snippets = []
    
    for pdf_meta in all_metadata:
        pdf_name = pdf_meta.get('source_pdf', 'unknown.pdf')
        
        for page_meta in pdf_meta.get('pages', []):
            page_num = page_meta.get('page_num', 0)
            
            for snip in page_meta.get('snippets', []):
                img_path = snip['path']
                
                if not os.path.exists(img_path):
                    continue
                
                meta = {
                    'pdf_name': pdf_name,
                    'page_num': page_num,
                    'column': snip.get('column', 0),
                    'x_offset': float(snip.get('x_offset', 0)),
                    'y_offset': float(snip.get('y_offset', 0)),
                }
                snippets.append((img_path, meta))
    
    return snippets


def main():
    # ==========================================================================
    # Setup Paths - Works in both Colab and local environments
    # ==========================================================================
    
    # Try to detect if we're in Colab
    try:
        from google.colab import drive
        IN_COLAB = True
        print("Running in Google Colab")
    except ImportError:
        IN_COLAB = False
        print("Running in local environment")
    
    # Determine project root
    if IN_COLAB:
        # Adjust this path for your Colab setup
        # If you mounted Google Drive: /content/drive/MyDrive/your_project
        # If you cloned a repo: /content/your_repo
        project_root = Path("/content/jan11_exp")
    else:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
    
    preprocessed_dir = project_root / "data" / "preprocessed"
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = preprocessed_dir / "all_metadata.json"
    if not metadata_path.exists():
        print(f"Error: Could not find metadata at {metadata_path}")
        print(f"Looked in: {metadata_path.absolute()}")
        return

    # ==========================================================================
    # Load metadata and collect all snippets
    # ==========================================================================
    
    print(f"Loading metadata from: {metadata_path}")
    with open(metadata_path, 'r') as f:
        all_metadata = json.load(f)

    snippets = collect_all_snippets(all_metadata)
    total_snippets = len(snippets)
    print(f"Found {total_snippets} image snippets to process")
    
    if total_snippets == 0:
        print("No snippets found. Check your metadata file and image paths.")
        return

    # ==========================================================================
    # Initialize Surya
    # ==========================================================================
    
    detection_predictor, recognition_predictor = init_surya()
    
    # ==========================================================================
    # Process in batches with streaming output
    # ==========================================================================
    
    output_file = output_dir / "ocr_output.jsonl"
    print(f"Streaming results to: {output_file}")
    print(f"Batch size: {BATCH_SIZE}")
    print("-" * 60)
    
    # Open in append mode for resume capability
    with open(output_file, 'a', encoding='utf-8') as f_out:
        batch_images = []
        batch_metadata = []
        processed = 0
        
        for idx, (img_path, meta) in enumerate(snippets):
            # Load image
            img = load_image_safe(img_path)
            if img is None:
                continue
            
            batch_images.append(img)
            batch_metadata.append(meta)
            
            # Process batch when full or at the end
            if len(batch_images) >= BATCH_SIZE or idx == total_snippets - 1:
                if batch_images:
                    batch_num = (processed // BATCH_SIZE) + 1
                    total_batches = (total_snippets + BATCH_SIZE - 1) // BATCH_SIZE
                    
                    print(f"Processing batch {batch_num}/{total_batches} "
                          f"({len(batch_images)} images) - "
                          f"[{processed + len(batch_images)}/{total_snippets}]")
                    
                    # Process the batch
                    entries = process_batch(
                        batch_images,
                        batch_metadata,
                        detection_predictor,
                        recognition_predictor
                    )
                    
                    # Write entries to JSONL
                    for entry in entries:
                        f_out.write(json.dumps(entry, cls=NumpyEncoder, ensure_ascii=False) + "\n")
                    
                    # Flush to disk (important for Colab disconnects)
                    f_out.flush()
                    
                    processed += len(batch_images)
                    
                    if VERBOSE:
                        print(f"  -> Wrote {len(entries)} text lines")
                    
                    # Cleanup batch
                    for img in batch_images:
                        img.close()
                    batch_images = []
                    batch_metadata = []
                    
                    # Aggressive memory cleanup
                    gc.collect()
                    
                    # Clear CUDA cache
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()  # Wait for all CUDA ops to complete
                    except ImportError:
                        pass

    print("-" * 60)
    print(f"Done! Processed {processed} images")
    print(f"Output saved to: {output_file}")


# =============================================================================
# Colab-specific helper functions
# =============================================================================

def mount_drive():
    """Helper to mount Google Drive in Colab."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted at /content/drive")
    except ImportError:
        print("Not running in Colab - drive mount not needed")


def install_surya():
    """Helper to install Surya in Colab."""
    import subprocess
    subprocess.run(["pip", "install", "-q", "surya-ocr"])
    print("Surya OCR installed")


def check_gpu():
    """Check GPU availability and print info."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name}")
            print(f"VRAM: {gpu_mem:.1f} GB")
            
            # Suggest batch size based on VRAM
            if gpu_mem >= 40:
                print(f"Suggested BATCH_SIZE: 64-128")
            elif gpu_mem >= 15:
                print(f"Suggested BATCH_SIZE: 16-32")
            else:
                print(f"Suggested BATCH_SIZE: 8-16")
        else:
            print("No GPU available - will use CPU (slower)")
    except ImportError:
        print("PyTorch not installed")


if __name__ == "__main__":
    main()