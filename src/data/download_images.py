import requests
import pandas as pd
from pathlib import Path
import tifffile
import numpy as np
from PIL import Image
import os
import io

# Setup paths
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed" / "patches"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = RAW_DIR / "image_manifest.csv"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"

def download_and_process_slide(file_id, file_name, patch_size=256, num_patches=20):
    """
    Downloads a single slide, extracts patches, deletes slide.
    """
    slide_path = RAW_DIR / file_name
    
    # Check if we already processed this
    slide_id = file_name.split('.')[0]
    slide_out_dir = PROCESSED_DIR / slide_id
    if slide_out_dir.exists() and len(list(slide_out_dir.glob("*.png"))) >= num_patches:
        print(f"Skipping {file_name}, already processed.")
        return

    # 1. Download
    print(f"Downloading {file_name} from GDC...")
    response = requests.get(f"{GDC_DATA_ENDPOINT}/{file_id}", stream=True)
    if response.status_code != 200:
        print(f"Failed to download {file_id}")
        return

    with open(slide_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024*1024):
            f.write(chunk)
    
    print(f"Downloaded {file_name}. Processing...")

    try:
        # 2. Open with tifffile
        with tifffile.TiffFile(slide_path) as tif:
            # SVS files usually have multiple pages (pyramid). 
            # Page 0 is highest res.
            # We can read regions without loading the whole image.
            
            # Use zarr for memory mapping if possible, or just read standard
            # But SVS is striped/tiled. Tifffile handles this well.
            
            # Access the high-res level (usually series 0)
            # Tifffile might return a `tifffile.ZarrStore` or similar if using asarray(out='memmap')
            # But for simple patching, let's try reading the dimensions first.
            
            page = tif.pages[0]
            width = page.imagewidth
            height = page.imagelength
            
            print(f"Slide dimensions: {width}x{height}")
            
            slide_out_dir.mkdir(exist_ok=True)
            
            patches_saved = 0
            attempts = 0
            
            # Simple random sampling for now, with tissue detection
            while patches_saved < num_patches and attempts < num_patches * 5:
                attempts += 1
                
                # Pick random coordinate
                x = np.random.randint(0, width - patch_size)
                y = np.random.randint(0, height - patch_size)
                
                # Read region
                # Note: Tifffile reads efficiently if tiled
                # [y:y+h, x:x+w]
                patch = page.asarray(out=None, key=slice(y, y+patch_size), squeeze=True)
                # If the above fails to slice due to page structure, we might need a different approach
                # But typically page.asarray() loads the whole thing which is BAD for RAM.
                # However, tifffile supports tiled access if the image is tiled.
                
                # Alternative: Use key argument in asarray to read a crop? 
                # tifffile's asarray doesn't always support ROI slicing for all formats easily without loading.
                # Let's use `page.crop` or `imread` with selection if supported.
                # Actually, standard tifffile `input[slice]` works if it's a memory-mapped array or compatible.
                # Let's try the safer `page.read_texture` or `zarr` interface if available, 
                # but `asarray` with slicing is the standard attempt.
                
                # REVISION: To be safe with RAM on a laptop, we shouldn't load the whole 1GB+ image.
                # `tifffile.imread(file, key=...)`
                # If `page.is_tiled`:
                # We can iterate tiles, but we want random patches.
                
                # Let's hope the `key` param or slicing works on the persistent object.
                # If not, for this demo, we might fail on 16GB implementations if we aren't careful.
                # Let's try to assume we can read the thumbnail first to find tissue?
                pass 
            
            # BETTER APPROACH for "Manageable Patches" without OpenSlide on Laptop:
            # Read a lower resolution level to find tissue, then map coordinates.
            # But Tifffile might not expose levels as easily as Openslide.
            # Typically SVS levels are in other pages. Use `tif.series` or `tif.pages`.
            
            # Let's write a simplified robust loop. 
            # We will read specific tiles.
            
            w, h = patch_size, patch_size
            
            for i in range(num_patches):
                # Random coords
                rx = np.random.randint(0, width - w)
                ry = np.random.randint(0, height - h)
                
                # Read patch
                try:
                    # Tifffile slicing requires the image data to be accessible.
                    # This reads only the necessary bytes if properly implemented for SVS
                    raw_patch = tif.pages[0].asarray()[ry:ry+h, rx:rx+w] 
                    # WAIT: tif.pages[0].asarray() loads the WHOLE image into memory. DANGEROUS.
                    # Correct way with Tifffile for SVS partial read:
                    # z = zarr.open(tif.pages[0].zarr_store) --> requires zarr
                    # Or use `tifffile.imread(..., selection=...)`
                    pass
                except:
                    continue

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
    finally:
        # 3. Clean up
        if slide_path.exists():
            os.remove(slide_path)
            print(f"Deleted source slide {file_name}")

def safe_patch_extraction_demo():
    """
    Since efficient random access in SVS without OpenSlide is tricky with just tifffile 
    (it often wants to load the strip/tile), we will try a different strategy:
    
    Download the slide.
    Use `tifffile.imread(path, key=0)` but that loads it all.
    
    Actually, let's look for a library that supports this. `imagecodecs` is usually needed by `tifffile`.
    
    Alternative: Just download the 'Thumbnail' or 'Slide Image' from GDC if available?
    GDC allows downloading `clincical` and `biospecimen`. 
    
    Wait, for the purpose of this task (GenAI), we need high res.
    
    Let's try using `tifffile.memmap`.
    """
    df = pd.read_csv(MANIFEST_PATH)
    
    # Process all slides in manifest
    for i, row in df.iterrows():
        file_id = row['file_id']
        file_name = row['file_name']
        
        print(f"Processing {file_name}...")
        
        # Download
        slide_path = RAW_DIR / file_name
        if not slide_path.exists():
            print("Downloading...")
            resp = requests.get(f"{GDC_DATA_ENDPOINT}/{file_id}", stream=True)
            with open(slide_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
        
        try:
            # Open with OpenSlide
            import openslide
            slide = openslide.OpenSlide(str(slide_path))
            
            # Level 0 is highest res
            w, h = slide.dimensions
            print(f"Slide dimensions: {w}x{h}")
            
            count = 0
            attempts = 0
            
            slide_out = PROCESSED_DIR / file_name.replace(".svs", "")
            slide_out.mkdir(parents=True, exist_ok=True)
            
            print(f"Sampling from {h}x{w}...")
            
            while count < 20 and attempts < 200:
                attempts += 1
                y = np.random.randint(0, h - 256)
                x = np.random.randint(0, w - 256)
                
                try:
                    # OpenSlide read_region returns RGBA
                    patch = slide.read_region((x, y), 0, (256, 256))
                    patch = patch.convert("RGB")
                    patch_arr = np.array(patch)
                    
                    # Tissue detection
                    mean_val = np.mean(patch_arr)
                    std_val = np.std(patch_arr)
                    
                    if attempts < 5:
                        print(f"Debug Patch {attempts}: Mean={mean_val:.2f}, Std={std_val:.2f}")

                    # Standard background is ~240-255 (White). Tissue < 235?
                    if mean_val < 235 and std_val > 5:
                        patch.save(slide_out / f"patch_{x}_{y}.png")
                        count += 1
                        if count % 5 == 0:
                            print(f"  Saved {count} patches...")
                    
                except Exception as ie:
                    continue
            
            print(f"Extracted {count} patches.")
            slide.close()
            
        except Exception as e:
            print(f"Failed to read/process {file_name}: {e}")
            import traceback
            traceback.print_exc()
            
        except Exception as e:
            print(f"Failed to read/process {file_name}: {e}")
            import traceback
            traceback.print_exc()
        
        # Cleanup
        if slide_path.exists():
            os.remove(slide_path)
            print("Deleted SVS.")

if __name__ == "__main__":
    if MANIFEST_PATH.exists():
        safe_patch_extraction_demo()
    else:
        print("Manifest not found.")
