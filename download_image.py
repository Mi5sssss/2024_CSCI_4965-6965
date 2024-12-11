from tqdm import tqdm  # Progress bar
from concurrent.futures import ThreadPoolExecutor
import planetary_computer as pc
from pystac_client import Client
import rioxarray
import os

# Open Planetary Computer STAC API
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

# Query Sentinel-2 images
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[-73.68, 42.73, -73.60, 42.80],  # Example bounding box
    datetime="2023-01-01/2023-12-31",      # Date range
    query={"eo:cloud_cover": {"lt": 10}}  # Filter by cloud cover
)

# Retrieve items from the search results
items = list(search.get_all_items())

# Output directory
output_dir = "./HAB_images"
os.makedirs(output_dir, exist_ok=True)

# Function to download an image
def download_image(item):
    try:
        asset = item.assets["visual"]
        signed_href = pc.sign(asset.href)
        image = rioxarray.open_rasterio(signed_href)
        output_path = os.path.join(output_dir, f"{item.id}.tif")
        image.rio.to_raster(output_path)
        return item.id, True
    except Exception as e:
        return item.id, False

# Parallel download with a progress bar
with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust the number of workers as needed
    with tqdm(total=len(items), desc="Downloading images") as pbar:
        futures = [executor.submit(download_image, item) for item in items]
        for future in tqdm(futures, desc="Processing downloads"):
            item_id, success = future.result()
            if not success:
                print(f"Failed to download {item_id}")
            pbar.update(1)
