#!/usr/bin/env python3
"""
Cloud Map Generator
Generates 2:1 aspect ratio cloud cover images using satellite data sources.
"""

import os
import sys
import json
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import logging
import io
from urllib.parse import urlencode
from numba import njit

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Numba-optimized functions
@njit
def interpolate(from_a, to_a, from_b, to_b, input_val):
    """Interpolate between two ranges."""
    if input_val < from_a:
        return from_b
    if input_val > to_a:
        return to_b
    proportion = (input_val - from_a) / (to_a - from_a)
    return from_b + (to_b - from_b) * proportion

@njit
def screen_blend(a, b):
    """Screen blend mode: 1 - (1-a) * (1-b)"""
    a_norm = a / 255.0
    b_norm = b / 255.0
    return (1 - (1 - a_norm) * (1 - b_norm)) * 255

@njit
def gamma_correction(gamma, input_val):
    """Apply gamma correction."""
    gamma_correction = 1 / gamma
    return 255 * (input_val / 255) ** gamma_correction

@njit
def process_cloud_pixels(ir_array, dust_array, visible_array, source_width, source_height):
    """Process cloud pixels using optimized Numba functions."""
    cloud_array = np.zeros((source_height, source_width, 3), dtype=np.uint8)
    
    for y in range(source_height):
        for x in range(source_width):
            # Get dust values
            dust_r = dust_array[y, x, 0]
            dust_b = 255 - dust_array[y, x, 2]
            
            # Process dust data
            dust_r_masked = (dust_r / 255.0) * (dust_b / 255.0) * 255
            dust = screen_blend(dust_r_masked, 0.5 * dust_b)
            
            # Process IR data
            ir = interpolate(72, 178, 0, 255, ir_array[y, x, 0])
            ir_gamma = gamma_correction(1.46, ir)
            
            # Combine IR and dust
            output_value = gamma_correction(2.0, screen_blend(dust, ir_gamma * 0.77))
            
            # Process visible data for additional detail
            visible_r = visible_array[y, x, 0]
            visible_g = visible_array[y, x, 1]
            visible_b = visible_array[y, x, 2]
            
            # Check if visible data looks greyscale
            visible_diff = max(visible_r, visible_g, visible_b) - min(visible_r, visible_g, visible_b)
            visible_gb_diff = abs(visible_g - visible_b)
            
            if visible_diff < 25 or (visible_r < visible_g and visible_gb_diff < 11 and visible_g > 150):
                # Apply visible enhancement
                visible_value = gamma_correction(1.5, max(visible_r, visible_g, visible_b))
                
                # Fade visible data near antimeridian
                fade_start = source_width * 0.4
                fade_end = source_width * 0.6
                if fade_start < x < fade_end and visible_value < 255:
                    am_adjustment = abs(interpolate(fade_start, fade_end, -1, 1, x))
                    final_value = max(visible_value * am_adjustment, output_value)
                else:
                    final_value = max(visible_value, output_value)
            else:
                final_value = output_value
            
            # Set pixel values
            cloud_array[y, x, 0] = int(final_value)
            cloud_array[y, x, 1] = int(final_value)
            cloud_array[y, x, 2] = int(final_value)
    
    return cloud_array

@njit
def fill_horizontal_gaps(cloud_array, source_width, source_height):
    """Fill horizontal gaps using interpolation."""
    height_to_process = source_height - source_height // 4
    
    for y in range(source_height // 8, height_to_process):
        gap_start_x = -1
        gap_end_x = -1
        gap_start_val = 0
        gap_end_val = 0
        after_gap_counter = 0
        gap_buffer = 3
        
        for x in range(source_width):
            pixel_val = cloud_array[y, x, 0]
            
            if pixel_val == 255 and gap_start_x == -1:
                gap_start_x = x - gap_buffer
                if gap_start_x >= 0:
                    gap_start_val = cloud_array[y, gap_start_x, 0]
            elif gap_start_x != -1 and pixel_val < 255 and after_gap_counter < gap_buffer:
                after_gap_counter += 1
            elif pixel_val < 255 and gap_start_x != -1 and after_gap_counter == gap_buffer:
                gap_end_x = x
                gap_end_val = pixel_val
                
                # Interpolate across the gap
                if gap_end_x > gap_start_x:
                    added_per_step = (gap_end_val - gap_start_val) / (gap_end_x - gap_start_x)
                    for gap_x in range(gap_start_x, gap_end_x + 1):
                        if 0 <= gap_x < source_width:
                            pixel_index = gap_x - gap_start_x
                            new_val = gap_start_val + pixel_index * added_per_step
                            new_val = max(0, min(255, new_val))
                            
                            cloud_array[y, gap_x, 0] = int(new_val)
                            cloud_array[y, gap_x, 1] = int(new_val)
                            cloud_array[y, gap_x, 2] = int(new_val)
                
                # Reset for next gap
                gap_start_x = -1
                gap_end_x = -1
                gap_start_val = 0
                gap_end_val = 0
                after_gap_counter = 0

@njit
def fill_vertical_gaps(cloud_array, source_width, source_height):
    """Fill vertical gaps using interpolation."""
    width_to_process = source_width - source_width // 4
    
    for x in range(source_width // 8, width_to_process):
        gap_start_y = -1
        gap_end_y = -1
        gap_start_val = 0
        gap_end_val = 0
        after_gap_counter = 0
        gap_buffer = 3
        
        for y in range(source_height):
            pixel_val = cloud_array[y, x, 0]
            
            if pixel_val == 255 and gap_start_y == -1:
                gap_start_y = y - gap_buffer
                if gap_start_y >= 0:
                    gap_start_val = cloud_array[gap_start_y, x, 0]
            elif gap_start_y != -1 and pixel_val < 255 and after_gap_counter < gap_buffer:
                after_gap_counter += 1
            elif pixel_val < 255 and gap_start_y != -1 and after_gap_counter == gap_buffer:
                gap_end_y = y
                gap_end_val = pixel_val
                
                # Interpolate across the gap
                if gap_end_y > gap_start_y:
                    added_per_step = (gap_end_val - gap_start_val) / (gap_end_y - gap_start_y)
                    for gap_y in range(gap_start_y, gap_end_y + 1):
                        if 0 <= gap_y < source_height:
                            pixel_index = gap_y - gap_start_y
                            new_val = gap_start_val + pixel_index * added_per_step
                            new_val = max(0, min(255, new_val))
                            
                            cloud_array[gap_y, x, 0] = int(new_val)
                            cloud_array[gap_y, x, 1] = int(new_val)
                            cloud_array[gap_y, x, 2] = int(new_val)
                
                # Reset for next gap
                gap_start_y = -1
                gap_end_y = -1
                gap_start_val = 0
                gap_end_val = 0
                after_gap_counter = 0

@njit
def fill_large_gaps(cloud_array, source_width, source_height):
    """Fill larger gaps using a more aggressive approach."""
    # Process the entire image for large gaps
    for y in range(source_height):
        for x in range(source_width):
            if cloud_array[y, x, 0] == 255:
                # Find the nearest non-white pixels in all directions
                left_val = 0
                right_val = 0
                top_val = 0
                bottom_val = 0
                
                # Look left
                for left_x in range(x-1, -1, -1):
                    if cloud_array[y, left_x, 0] < 255:
                        left_val = cloud_array[y, left_x, 0]
                        break
                
                # Look right
                for right_x in range(x+1, source_width):
                    if cloud_array[y, right_x, 0] < 255:
                        right_val = cloud_array[y, right_x, 0]
                        break
                
                # Look up
                for top_y in range(y-1, -1, -1):
                    if cloud_array[top_y, x, 0] < 255:
                        top_val = cloud_array[top_y, x, 0]
                        break
                
                # Look down
                for bottom_y in range(y+1, source_height):
                    if cloud_array[bottom_y, x, 0] < 255:
                        bottom_val = cloud_array[bottom_y, x, 0]
                        break
                
                # Calculate average of found values
                valid_vals = []
                if left_val > 0:
                    valid_vals.append(left_val)
                if right_val > 0:
                    valid_vals.append(right_val)
                if top_val > 0:
                    valid_vals.append(top_val)
                if bottom_val > 0:
                    valid_vals.append(bottom_val)
                
                if valid_vals:
                    avg_val = sum(valid_vals) / len(valid_vals)
                    avg_val = max(0, min(255, avg_val))
                    
                    cloud_array[y, x, 0] = int(avg_val)
                    cloud_array[y, x, 1] = int(avg_val)
                    cloud_array[y, x, 2] = int(avg_val)

@njit
def fill_antimeridian_gaps(cloud_array, source_width, source_height):
    """Fill gaps at the antimeridian using both horizontal and vertical interpolation."""
    # Fill horizontal gaps first
    fill_horizontal_gaps(cloud_array, source_width, source_height)
    
    # Then fill vertical gaps
    fill_vertical_gaps(cloud_array, source_width, source_height)
    
    # Run a second pass to catch any remaining gaps
    fill_horizontal_gaps(cloud_array, source_width, source_height)
    fill_vertical_gaps(cloud_array, source_width, source_height)
    
    # Fill any remaining large gaps
    fill_large_gaps(cloud_array, source_width, source_height)

@njit
def mirror_poles(cloud_array, source_width, source_height):
    """Mirror the image at top and bottom to fill polar gaps."""
    height_to_mirror = source_height // 8
    
    # Mirror bottom to top
    for y in range(height_to_mirror):
        for x in range(source_width):
            source_y = source_height - height_to_mirror*2 + y
            if source_y < source_height:
                cloud_array[y, x, 0] = cloud_array[source_y, x, 0]
                cloud_array[y, x, 1] = cloud_array[source_y, x, 1]
                cloud_array[y, x, 2] = cloud_array[source_y, x, 2]
    
    # Mirror top to bottom
    for y in range(source_height - height_to_mirror, source_height):
        for x in range(source_width):
            source_y = height_to_mirror + (y - (source_height - height_to_mirror))
            if source_y < source_height:
                cloud_array[y, x, 0] = cloud_array[source_y, x, 0]
                cloud_array[y, x, 1] = cloud_array[source_y, x, 1]
                cloud_array[y, x, 2] = cloud_array[source_y, x, 2]

class CloudMapGenerator:
    def __init__(self, output_dir="output", config_file=None):
        """
        Initialize the Cloud Map Generator.
        
        Args:
            output_dir (str): Directory to save generated images
            config_file (str): Path to configuration file
        """
        self.output_dir = output_dir
        self.config = self._load_config(config_file)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Image dimensions for 2:1 aspect ratio
        self.source_width = 8192
        self.source_height = 4096
        self.width = self.source_width
        self.height = self.source_height
        
        # EUMETSAT WMS base URL
        self.wms_base_url = "https://view.eumetsat.int/geoserver/ows"
        
    def _load_config(self, config_file):
        """Load configuration from file or use defaults."""
        default_config = {
            "wms_layers": {
                "ir108": "mumi:worldcloudmap_ir108",
                "dust_rgb": "mumi:wideareacoverage_rgb_dust", 
                "natural_rgb": "mumi:wideareacoverage_rgb_natural"
            },
            "image_settings": {
                "default_width": 8192,
                "default_height": 4096,
                "output_formats": ["jpg", "png"],
                "quality": 95
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Could not load config file: {e}")
        
        return default_config
    
    def fetch_wms_data(self, layer_type="ir108", hemisphere="left"):
        """
        Fetch data from EUMETSAT WMS service.
        
        Args:
            layer_type (str): Type of layer to fetch (ir108, dust_rgb, natural_rgb)
            hemisphere (str): "left" or "right" hemisphere
        
        Returns:
            PIL.Image: Fetched image
        """
        layer_name = self.config["wms_layers"][layer_type]
        
        # Define bounding boxes for left and right hemispheres
        if hemisphere == "left":
            bbox = "-90,-180,90,0"  # Western hemisphere
        else:
            bbox = "-90,0,90,180"   # Eastern hemisphere
        
        # WMS request parameters
        params = {
            "service": "WMS",
            "request": "GetMap",
            "version": "1.3.0",
            "layers": layer_name,
            "styles": "",
            "format": "image/png",
            "crs": "EPSG:4326",
            "bbox": bbox,
            "width": self.source_height,  # Square image for each hemisphere
            "height": self.source_height
        }
        
        url = f"{self.wms_base_url}?{urlencode(params)}"
        
        try:
            logger.info(f"Fetching {layer_type} {hemisphere} hemisphere from EUMETSAT WMS")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Convert response to PIL Image
            image = Image.open(io.BytesIO(response.content))
            return image
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch WMS data: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to process WMS image: {e}")
            return None
    
    
    def multiply_blend(self, a, b):
        """Multiply blend mode: a * b"""
        a_norm = a / 255.0
        b_norm = b / 255.0
        return a_norm * b_norm * 255
    
    def process_cloud_data(self, ir_west, ir_east, dust_west, dust_east, visible_west, visible_east):
        """
        
        Args:
            ir_west, ir_east: IR images for western and eastern hemispheres
            dust_west, dust_east: Dust RGB images for western and eastern hemispheres  
            visible_west, visible_east: Visible RGB images for western and eastern hemispheres
        
        Returns:
            PIL.Image: Processed cloud cover image
        """
        logger.info("Processing cloud data using advanced algorithms")
        
        # Combine western and eastern images 
        ir_map = Image.new('RGB', (self.source_width, self.source_height))
        ir_map.paste(ir_east, (0, 0))  # IR_MAP_RIGHT 
        ir_map.paste(ir_west, (self.source_width//2, 0))  # IR_MAP_LEFT 
        
        dust_map = Image.new('RGB', (self.source_width, self.source_height))
        dust_map.paste(dust_east, (0, 0))
        dust_map.paste(dust_west, (self.source_width//2, 0))
        
        visible_map = Image.new('RGB', (self.source_width, self.source_height))
        visible_map.paste(visible_east, (0, 0))
        visible_map.paste(visible_west, (self.source_width//2, 0))
        
        # Create cloud map canvas
        cloud_map = Image.new('RGB', (self.source_width, self.source_height), (255, 0, 0))
        
        # Convert to numpy arrays for processing
        ir_array = np.array(ir_map)
        dust_array = np.array(dust_map)
        visible_array = np.array(visible_map)
        
        # Process cloud pixels using optimized Numba function
        cloud_array = process_cloud_pixels(ir_array, dust_array, visible_array, self.source_width, self.source_height)
        
        # Fill gaps at antimeridian using optimized function
        fill_antimeridian_gaps(cloud_array, self.source_width, self.source_height)
        
        # Save the right half (western hemisphere)
        right_half = cloud_array[:, self.source_width//2:].copy()
        
        # Move left half (eastern hemisphere) to the right side
        cloud_array[:, self.source_width//2:] = cloud_array[:, :self.source_width//2]
        
        # Put the saved right half (western hemisphere) on the left side
        cloud_array[:, :self.source_width//2] = right_half
        
        # Mirror poles to fill gaps using optimized function
        mirror_poles(cloud_array, self.source_width, self.source_height)
        
        return Image.fromarray(cloud_array.astype(np.uint8))
    
    
    
    
    def save_image(self, image, filename, format="JPEG"):
        """
        Save image to file.
        
        Args:
            image (PIL.Image): Image to save
            filename (str): Output filename
            format (str): Image format
        """
        if image is None:
            logger.error("No image to save")
            return False
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            image.save(filepath, format=format, quality=95)
            logger.info(f"Saved image: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return False
    
    def generate_cloud_maps(self, data_source="eumetsat", product_type="ir108"):
        """
        Generate cloud maps using EUMETSAT WMS service.
        
        Args:
            data_source (str): Data source to use (currently only "eumetsat" supported)
            product_type (str): Product type for EUMETSAT data
        """
        logger.info(f"Generating cloud maps using EUMETSAT WMS service")
        
        # Fetch all required images
        logger.info("Fetching IR images...")
        ir_west = self.fetch_wms_data("ir108", "left")  # Western hemisphere (-180 to 0)
        ir_east = self.fetch_wms_data("ir108", "right")  # Eastern hemisphere (0 to 180)
        
        logger.info("Fetching Dust RGB images...")
        dust_west = self.fetch_wms_data("dust_rgb", "left")
        dust_east = self.fetch_wms_data("dust_rgb", "right")
        
        logger.info("Fetching Natural Colour RGB images...")
        visible_west = self.fetch_wms_data("natural_rgb", "left")
        visible_east = self.fetch_wms_data("natural_rgb", "right")
        
        # Check if all images were fetched successfully
        images = [ir_west, ir_east, dust_west, dust_east, visible_west, visible_east]
        if any(img is None for img in images):
            logger.error("Failed to fetch some satellite images")
            return False
        
        # Process cloud data using the advanced algorithm
        cloud_map = self.process_cloud_data(ir_west, ir_east, dust_west, dust_east, visible_west, visible_east)
        if cloud_map is None:
            logger.error("Failed to process cloud data")
            return False
        
        # Save only the cloud map (overwrites previous image)
        success = self.save_image(cloud_map, "clouds.jpg")
        
        if success:
            logger.info("Successfully generated cloud map")
        else:
            logger.error("Failed to save cloud map")
        
        return success
    

def main():
    """Main function to run the cloud map generator."""
    parser = argparse.ArgumentParser(description="Generate 2:1 cloud cover images using EUMETSAT WMS")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--source", choices=["eumetsat"], default="eumetsat", 
                       help="Data source to use (currently only EUMETSAT WMS supported)")
    parser.add_argument("--product", choices=["ir108", "dust_rgb", "natural_rgb"], default="ir108",
                       help="Primary product type (all products are fetched)")
    parser.add_argument("--width", type=int, default=8192, help="Image width")
    parser.add_argument("--height", type=int, default=4096, help="Image height")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = CloudMapGenerator(args.output_dir, args.config)
    
    # Set custom dimensions
    generator.source_width = args.width
    generator.source_height = args.height
    generator.width = args.width
    generator.height = args.height
    
    # Generate cloud maps
    success = generator.generate_cloud_maps(args.source, args.product)
    
    if success:
        print(f"Cloud map generated successfully in {args.output_dir}")
        print("Generated file:")
        print("- clouds.jpg - Cloud cover map")
    else:
        print("Failed to generate cloud map")
        sys.exit(1)

if __name__ == "__main__":
    main()
