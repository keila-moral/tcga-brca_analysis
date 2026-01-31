import tifffile
import numpy as np
from pathlib import Path
import zarr

def check_pixels():
    # Find a downloaded SVS
    raw_dir = Path("data/raw")
    svs_files = list(raw_dir.glob("*.svs"))
    
    if not svs_files:
        print("No SVS files found.")
        return
        
    f = svs_files[0]
    print(f"Checking {f.name}")
    
    with tifffile.TiffFile(f) as tif:
        print(f"Pages: {len(tif.pages)}")
        for i, p in enumerate(tif.pages):
            print(f"Page {i}: {p.shape}, {p.dtype}")
            
        # Try reading center of Page 0
        p0 = tif.pages[0]
        store = p0.aszarr()
        z = zarr.open(store, mode='r')
        
        h, w = z.shape[0], z.shape[1]
        cy, cx = h//2, w//2
        
        print(f"Reading center crop at {cx}, {cy}")
        patch = z[cy:cy+256, cx:cx+256]
        
        print(f"Patch shape: {patch.shape}")
        print(f"Mean: {np.mean(patch)}")
        print(f"Std: {np.std(patch)}")
        print(f"Min: {np.min(patch)}, Max: {np.max(patch)}")

if __name__ == "__main__":
    check_pixels()
