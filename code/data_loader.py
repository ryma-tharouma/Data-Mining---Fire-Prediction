"""
load_data.py
------------
Utility functions to explore and load datasets for the Data Mining project.

"""

import os
from pathlib import Path
import zipfile


# === Configuration ===
DATA_DIR = Path("../Data mining/data")

# === Utility: List all files ===
def list_data_files(base_dir: Path = DATA_DIR):
    """Recursively list all files under data/ directory."""
    print(f"Listing files in: {base_dir.resolve()}\n" + "=" * 60)
    for path in sorted(base_dir.rglob("*")):
        if path.is_dir():
            print(f"📁 {path.relative_to(base_dir)}")
        else:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"📄 {path.relative_to(base_dir)} ({size_mb:.1f} MB)")
    print("=" * 60)


# === Placeholder: Load raster data (to fill later with rasterio) ===
def load_raster_data(file_path: Path):
    """
    Placeholder for raster data loading.
    Will later use rasterio.open(file_path) once rasterio is verified.
    """
    print(f"[INFO] Raster loader not yet implemented for: {file_path.name}")
    return None


# === Placeholder: Load vector (shapefile) data (to fill later with geopandas) ===
def load_vector_data(file_path: Path):
    """
    Placeholder for vector data loading.
    Will later use geopandas.read_file(file_path).
    """
    print(f"[INFO] Vector loader not yet implemented for: {file_path.name}")
    return None


# === Optional: Unzip helper (for .zip datasets) ===
def unzip_file(zip_path: Path, extract_to: Path = None):
    """Extracts a ZIP file to the given folder."""
    extract_to = extract_to or zip_path.parent
    print(f"[INFO] Extracting {zip_path.name} → {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("[INFO] Extraction complete.")


# === Run as script ===
if __name__ == "__main__":
    list_data_files()
