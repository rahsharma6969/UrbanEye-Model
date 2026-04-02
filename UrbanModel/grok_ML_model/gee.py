# =============================================
# SAMPLE CODE: Check Google Earth Engine Image Quality
# For Bhayandar / Aarey-Thane area
# =============================================

import ee
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import requests
import rasterio as rio
from rasterio.plot import show

# ─── 1. Initialize GEE (use your project) ───
ee.Initialize(project='urbaneye-476904')  # ← your project ID

# ─── 2. Define Area of Interest (Bhayandar / Aarey) ───
# 5 km buffer around central point
aoi = ee.Geometry.Point([72.85, 19.3]).buffer(5000)

# ─── 3. Get a recent Sentinel-2 image (2025 example) ───
s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(aoi)
      .filterDate('2025-01-01', '2025-06-30')
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
      .median()  # or .qualityMosaic('B8') for NIR best
      .select(['B4', 'B3', 'B2', 'B8']))  # RGB + NIR

print("Image bands:", s2.bandNames().getInfo())

# ─── 4. Visualization parameters (for true color & NIR) ───
true_color_vis = {
    'bands': ['B4', 'B3', 'B2'],
    'min': 0,
    'max': 3000,
    'gamma': 1.4
}

nir_vis = {
    'bands': ['B8', 'B4', 'B3'],
    'min': 0,
    'max': 5000,
    'gamma': 1.2
}

# ─── 5. Quick Method 1: Get thumbnail (fast preview quality check) ───
thumbnail_url = s2.getThumbURL({
    'region': aoi,
    'dimensions': 800,          # pixels wide
    'format': 'png',
    **true_color_vis
})

print("Thumbnail URL (open in browser):", thumbnail_url)

# Download and show thumbnail
response = requests.get(thumbnail_url)
if response.status_code == 200:
    img = Image.open(io.BytesIO(response.content))
    plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.title("GEE Thumbnail - True Color (Quick Quality Check)")
    plt.axis('off')
    plt.show()
else:
    print("Thumbnail download failed:", response.status_code)

# ─── 6. Method 2: Export full image & check detailed quality ───
# Export small region for quality inspection
task = ee.batch.Export.image.toDrive(
    image=s2.visualize(**true_color_vis),
    description='bhayandar_s2_quality_check',
    folder='UrbanEye',
    region=aoi,
    scale=10,
    maxPixels=1e9
)
task.start()

print("Export task started! Check status: https://code.earthengine.google.com/tasks")
print("Wait 5–20 min, then download from Drive > UrbanEye folder")

# ─── 7. After download: Load file and do quality checks ───
# Replace with your actual downloaded filename
filename = r"C:\Users\Rahul\Downloads\UrbanEye\bhayandar_s2_quality_check.tif"  # ← CHANGE THIS

try:
    with rio.open(filename) as src:
        data = src.read()  # shape: (bands, height, width)
        profile = src.profile
        
        print("\nImage Metadata:")
        print("Shape:", data.shape)
        print("CRS:", src.crs)
        print("Resolution:", src.res)
        print("Band count:", src.count)
        
        # Value range per band
        for i, band_name in enumerate(['Red', 'Green', 'Blue', 'NIR']):
            band = data[i]
            print(f"{band_name} band → Min: {band.min():.2f}, Max: {band.max():.2f}, Mean: {band.mean():.2f}")
        
        # Plot RGB
        rgb = np.moveaxis(data[:3,:,:], 0, -1)  # B4,B3,B2 → RGB
        rgb = np.clip(rgb / 3000, 0, 1)  # stretch
        
        plt.figure(figsize=(10,10))
        plt.imshow(rgb)
        plt.title("Downloaded Sentinel-2 Image - True Color")
        plt.axis('off')
        plt.show()
        
        # Histogram (check contrast / data distribution)
        plt.figure(figsize=(10,4))
        for i, color in zip(range(3), ['red','green','blue']):
            plt.hist(data[i].flatten(), bins=100, color=color, alpha=0.5, label=['R','G','B'][i])
        plt.title("Histogram of RGB Bands (Quality Check)")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

except Exception as e:
    print("Error loading file:", e)
    print("Make sure the file is downloaded and path is correct.")