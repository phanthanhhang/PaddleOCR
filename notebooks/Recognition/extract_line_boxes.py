import json
import cv2
import numpy as np
import os
from glob import glob
from pathlib import Path
import traceback

def load_ocr_data(json_path):
    """Load OCR data from JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure all text content is properly encoded
        def ensure_utf8_encoding(obj):
            if isinstance(obj, dict):
                return {k: ensure_utf8_encoding(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [ensure_utf8_encoding(item) for item in obj]
            elif isinstance(obj, str):
                return obj.encode('utf-8', errors='replace').decode('utf-8')
            else:
                return obj
        
        return ensure_utf8_encoding(data)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Error loading OCR data from {json_path}: {e}")
        return None

def load_image(image_path):
    """Load image using OpenCV."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def get_polygon_bbox(polygon):
    """Get bounding box from polygon coordinates."""
    if len(polygon) < 8:
        return None
    
    # Reshape polygon to (n, 2) format
    points = np.array(polygon).reshape(-1, 2)
    
    # Get bounding box coordinates
    x_min = int(np.min(points[:, 0]))
    y_min = int(np.min(points[:, 1]))
    x_max = int(np.max(points[:, 0]))
    y_max = int(np.max(points[:, 1]))
    
    return x_min, y_min, x_max, y_max

def compute_rotation_leftright_points(polygon):
    """Method 1: Compute rotation using leftmost and rightmost points."""
    points = np.array(polygon).reshape(-1, 2)
    
    # Find the leftmost and rightmost points
    leftmost_idx = np.argmin(points[:, 0])
    rightmost_idx = np.argmax(points[:, 0])
    
    leftmost_point = points[leftmost_idx]
    rightmost_point = points[rightmost_idx]
    
    # Calculate angle from horizontal
    dx = rightmost_point[0] - leftmost_point[0]
    dy = rightmost_point[1] - leftmost_point[1]
    
    # Avoid division by zero
    if dx == 0:
        angle = 90  # Vertical line
    else:
        angle = abs(np.degrees(np.arctan(dy / dx)))
    
    return angle

def compute_rotation_pca(polygon):
    """Method 2: Compute rotation using Principal Component Analysis (manual implementation)."""
    points = np.array(polygon).reshape(-1, 2)
    
    # Center the data
    mean_point = np.mean(points, axis=0)
    centered_points = points - mean_point
    
    # Compute covariance matrix
    cov_matrix = np.cov(centered_points.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Get the principal component (eigenvector with largest eigenvalue)
    principal_idx = np.argmax(eigenvalues)
    principal_axis = eigenvectors[:, principal_idx]
    
    # Calculate angle from horizontal
    angle = abs(np.degrees(np.arctan2(principal_axis[1], principal_axis[0])))
    
    return angle

def compute_rotation_min_area_rect(polygon):
    """Method 3: Compute rotation using minimum area rectangle."""
    points = np.array(polygon).reshape(-1, 2).astype(np.float32)
    
    # Find minimum area rectangle
    rect = cv2.minAreaRect(points)
    
    # Get the angle of the rectangle
    angle = abs(rect[2])
    
    # OpenCV returns angle in range [-90, 0], normalize to [0, 90]
    if angle > 45:
        angle = 90 - angle
    
    return angle

def compute_rotation_least_squares(polygon):
    """Method 4: Compute rotation using least squares line fitting (manual implementation)."""
    points = np.array(polygon).reshape(-1, 2)
    
    # Manual least squares implementation
    x = points[:, 0]
    y = points[:, 1]
    
    # Calculate slope using least squares formula
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if denominator == 0:
        angle = 90  # Vertical line
    else:
        slope = numerator / denominator
        angle = abs(np.degrees(np.arctan(slope)))
    
    return angle

def compute_rotation_top_bottom_edges(polygon):
    """Method 5: Compute rotation using top and bottom edges."""
    points = np.array(polygon).reshape(-1, 2)
    
    # Find top and bottom points
    top_idx = np.argmin(points[:, 1])
    bottom_idx = np.argmax(points[:, 1])
    
    # For each, find the point with similar y-coordinate but different x
    top_y = points[top_idx, 1]
    bottom_y = points[bottom_idx, 1]
    
    # Find points close to top and bottom y-coordinates
    top_threshold = 10  # pixels
    bottom_threshold = 10  # pixels
    
    top_points = points[np.abs(points[:, 1] - top_y) <= top_threshold]
    bottom_points = points[np.abs(points[:, 1] - bottom_y) <= bottom_threshold]
    
    if len(top_points) >= 2 and len(bottom_points) >= 2:
        # Calculate angle from top edge
        top_left = top_points[np.argmin(top_points[:, 0])]
        top_right = top_points[np.argmax(top_points[:, 0])]
        
        dx = top_right[0] - top_left[0]
        dy = top_right[1] - top_left[1]
        
        if dx == 0:
            angle = 90
        else:
            angle = abs(np.degrees(np.arctan(dy / dx)))
    else:
        # Fallback to leftmost-rightmost method
        angle = compute_rotation_leftright_points(polygon)
    
    return angle

def compute_all_rotation_methods(polygon):
    """Compute rotation using all available methods and return a summary."""
    methods = {
        'leftright_points': compute_rotation_leftright_points,
        'pca': compute_rotation_pca,
        'min_area_rect': compute_rotation_min_area_rect,
        'least_squares': compute_rotation_least_squares,
        'top_bottom_edges': compute_rotation_top_bottom_edges
    }
    
    results = {}
    for method_name, method_func in methods.items():
        try:
            angle = method_func(polygon)
            results[method_name] = round(angle, 2)
        except Exception as e:
            print(f"Error in {method_name}: {e}")
            results[method_name] = None
    
    return results

def is_horizontal_line(polygon, min_aspect_ratio=2.0, min_width=50, max_angle=15, rotation_method='pca'):
    """
    Determine if a line is horizontal based on its polygon coordinates.
    
    Args:
        polygon: List of polygon coordinates
        min_aspect_ratio: Minimum width/height ratio to consider horizontal
        min_width: Minimum width in pixels to consider as a valid line
        max_angle: Maximum angle in degrees from horizontal (0°) to consider as horizontal
        rotation_method: Method to use for rotation calculation ('leftright_points', 'pca', 'min_area_rect', 'least_squares', 'top_bottom_edges')
    
    Returns:
        bool: True if the line is horizontal, False otherwise
    """
    bbox = get_polygon_bbox(polygon)
    if bbox is None:
        return False
    
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    print(f"Width: {width}, Height: {height}")
    
    # Avoid division by zero
    if height == 0:
        return width >= min_width
    
    aspect_ratio = width / height
    print(f"Aspect ratio: {aspect_ratio}")
    
    # Check basic aspect ratio and minimum width first
    if aspect_ratio < min_aspect_ratio or width < min_width:
        return False
    
    # Calculate angle using specified method
    rotation_methods = {
        'leftright_points': compute_rotation_leftright_points,
        'pca': compute_rotation_pca,
        'min_area_rect': compute_rotation_min_area_rect,
        'least_squares': compute_rotation_least_squares,
        'top_bottom_edges': compute_rotation_top_bottom_edges
    }
    
    if rotation_method not in rotation_methods:
        print(f"Unknown rotation method: {rotation_method}, using PCA")
        rotation_method = 'pca'
    
    try:
        angle = rotation_methods[rotation_method](polygon)
        print(f"Angle ({rotation_method}): {angle}")
        
        # For comparison, show all methods
        all_angles = compute_all_rotation_methods(polygon)
        print(f"All rotation methods: {all_angles}")
        
    except Exception as e:
        print(f"Error computing rotation with {rotation_method}: {e}")
        # Fallback to leftright method
        angle = compute_rotation_leftright_points(polygon)
        print(f"Angle (fallback): {angle}")
    
    # Check if angle is within acceptable range for horizontal lines
    is_horizontal = angle <= max_angle
    print(f"Is horizontal: {is_horizontal}")
    
    return is_horizontal

def crop_line_from_image(image, polygon, padding=5):
    """Crop line region from image based on polygon coordinates."""
    if image is None or not polygon:
        return None
    
    bbox = get_polygon_bbox(polygon)
    if bbox is None:
        return None
    
    x_min, y_min, x_max, y_max = bbox
    
    # Add padding
    height, width = image.shape[:2]
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)
    
    # Crop the image
    cropped = image[y_min:y_max, x_min:x_max]
    
    return cropped

def sanitize_filename(filename):
    """Sanitize filename to remove invalid characters."""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove extra spaces and dots
    filename = filename.strip().replace('..', '.')
    
    # Limit filename length
    if len(filename) > 200:
        filename = filename[:200]
    
    return filename

def process_folder(annotation_folder, image_folder, output_folder, folder_name, rotation_method='pca'):
    """Process all JSON files in a folder and extract horizontal line boxes."""
    print(f"\nProcessing folder: {folder_name}")
    print(f"Annotation folder: {annotation_folder}")
    print(f"Image folder: {image_folder}")
    print(f"Using rotation method: {rotation_method}")
    
    # Get all JSON files
    json_files = glob(os.path.join(annotation_folder, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {annotation_folder}")
        return
    
    processed_count = 0
    total_lines_extracted = 0
    total_horizontal_lines = 0
    
    for json_path in json_files[0:1]:
        try:
            # Load OCR data
            ocr_data = load_ocr_data(json_path)
            if ocr_data is None:
                continue
            
            # Get corresponding image path
            json_filename = os.path.basename(json_path)
            image_filename = json_filename.replace('.json', '.jpeg')
            image_path = os.path.join(image_folder, image_filename)
            
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
            
            # Load image
            image = load_image(image_path)
            if image is None:
                continue
            
            # Get image name without extension
            image_name = os.path.splitext(image_filename)[0]
            
            # Extract lines from OCR data
            pages = ocr_data.get('pages', [])
            if not pages:
                print(f"No pages found in {json_path}")
                continue
            
            page = pages[0]  # Process first page
            lines = page.get('lines', [])
            
            if not lines:
                print(f"No lines found in {json_path}")
                continue
            else:
                print(f"{len(lines)} Lines found in {json_path}")
            
            # Filter horizontal lines first
            horizontal_lines = []
            for line_idx, line in enumerate(lines):
                polygon = line.get('polygon', [])
                content = line.get('content', '').strip()
                
                if not polygon or not content:
                    continue
                if len(content) > 100 or 'ppa' in content.lower():
                    continue
                
                # Check if line is horizontal (not slanted or vertical)
                print(f"Checking line {line_idx} with content: {content}, length: {len(content)}")
                if len(content) < 3:
                    min_aspect_ratio = 0.6
                    min_width = 10
                    max_angle = 20
                else:
                    min_aspect_ratio = 1.0
                    min_width = 20
                    max_angle = 15
                print(f"Min aspect ratio: {min_aspect_ratio}, Min width: {min_width}, Max angle: {max_angle}")
                if is_horizontal_line(polygon, min_aspect_ratio=min_aspect_ratio, min_width=min_width, max_angle=max_angle, rotation_method=rotation_method):
                    horizontal_lines.append((line_idx, line))
                    total_horizontal_lines += 1
                print('='*100)
            
            print(f"Found {len(horizontal_lines)} horizontal lines out of {len(lines)} total lines")
            
            # Process each horizontal line
            for line_idx, line in horizontal_lines:
                polygon = line.get('polygon', [])
                content = line.get('content', '').strip()
                
                # Crop line from image
                cropped_line = crop_line_from_image(image, polygon, padding=5)
                
                if cropped_line is None or cropped_line.size == 0:
                    continue
                
                # Get dimensions and angle for logging
                bbox = get_polygon_bbox(polygon)
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    width = x_max - x_min
                    height = y_max - y_min
                    aspect_ratio = width / height if height > 0 else 0
                    
                    # Calculate angle using the chosen method
                    rotation_methods = {
                        'leftright_points': compute_rotation_leftright_points,
                        'pca': compute_rotation_pca,
                        'min_area_rect': compute_rotation_min_area_rect,
                        'least_squares': compute_rotation_least_squares,
                        'top_bottom_edges': compute_rotation_top_bottom_edges
                    }
                    
                    if rotation_method in rotation_methods:
                        angle = rotation_methods[rotation_method](polygon)
                    else:
                        angle = compute_rotation_pca(polygon)  # Default to PCA
                
                # Create output filename
                # Format: FOLDER__imagename__horizontal_line_number.jpg
                output_filename = f"{folder_name}__{image_name}__horizontal_line_{line_idx:03d}.jpg"
                output_filename = sanitize_filename(output_filename)
                output_path = os.path.join(output_folder, output_filename)
                
                # Save cropped line
                try:
                    cv2.imwrite(output_path, cropped_line)
                    total_lines_extracted += 1
                    
                    # Print progress for first few lines of each image
                    if total_lines_extracted <= 5:
                        print(f"  Extracted horizontal line {line_idx}: '{content[:50]}...' -> {output_filename}")
                        print(f"    Dimensions: {width}x{height}, Aspect ratio: {aspect_ratio:.2f}, Angle: {angle:.1f}°")
                        
                except Exception as e:
                    print(f"Error saving {output_path}: {e}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            traceback.print_exc()
    
    print(f"Completed folder {folder_name}: {processed_count} files processed")
    print(f"Total horizontal lines found: {total_horizontal_lines}")
    print(f"Total horizontal lines extracted: {total_lines_extracted}")

def main():
    """Main function to extract horizontal line boxes from all annotation folders."""
    # Base paths
    annotation_base = "/data1/hang/Stellantis/text_recognition/annotations"
    image_base = "/data1/stellantis/images"
    output_base = "/data1/hang/Stellantis/PaddleOCR/notebooks/Recognition/extracted_horizontal_line_boxes"
    
    # Annotation folders that exist
    annotation_folders = ["ZBII", "VDN", "INVOICE", "ID"]
    
    # Choose rotation method: 'leftright_points', 'pca', 'min_area_rect', 'least_squares', 'top_bottom_edges'
    rotation_method = 'pca'  # You can change this to test different methods
    
    # Create output directory
    os.makedirs(output_base, exist_ok=True)
    
    print("Starting HORIZONTAL line box extraction from OCR annotations...")
    print(f"Output directory: {output_base}")
    print(f"Rotation calculation method: {rotation_method}")
    print("Available rotation methods:")
    print("  - leftright_points: Uses leftmost and rightmost points")
    print("  - pca: Uses Principal Component Analysis")
    print("  - min_area_rect: Uses minimum area rectangle")
    print("  - least_squares: Uses least squares line fitting")
    print("  - top_bottom_edges: Uses top and bottom edges")
    print("Filtering criteria (excluding slanted and vertical boxes):")
    print("  - Minimum aspect ratio (width/height): 1.0 (or 0.6 for short content)")
    print("  - Minimum width: 20 pixels (or 10 for short content)")
    print("  - Maximum angle from horizontal: 30 degrees (or 20 for short content)")
    
    total_extracted = 0
    
    for folder_name in annotation_folders[0:1]:
        annotation_folder = os.path.join(annotation_base, folder_name)
        image_folder = os.path.join(image_base, folder_name)
        
        # Check if both annotation and image folders exist
        if not os.path.exists(annotation_folder):
            print(f"Warning: Annotation folder not found: {annotation_folder}")
            continue
        
        if not os.path.exists(image_folder):
            print(f"Warning: Image folder not found: {image_folder}")
            continue
        
        # Process this folder
        process_folder(annotation_folder, image_folder, output_base, folder_name, rotation_method)
    
    print(f"\n" + "="*60)
    print("HORIZONTAL LINE BOX EXTRACTION COMPLETE")
    print("="*60)
    print(f"All extracted horizontal line boxes saved to: {output_base}")
    print(f"Naming convention: FOLDER__imagename__horizontal_line_XXX.jpg")
    print(f"Rotation method used: {rotation_method}")
    print("="*60)

def test_rotation_methods():
    """Test function to demonstrate all rotation methods on sample data."""
    print("Testing rotation calculation methods...")
    
    # Test with different polygon shapes
    test_polygons = [
        # Horizontal line
        [100, 100, 200, 100, 200, 120, 100, 120],
        # Slightly tilted line
        [100, 100, 200, 110, 200, 130, 100, 120],
        # More tilted line
        [100, 100, 200, 130, 200, 150, 100, 120],
        # Vertical line
        [100, 100, 120, 100, 120, 200, 100, 200],
        # Rectangle (should be horizontal)
        [50, 50, 150, 50, 150, 80, 50, 80]
    ]
    
    descriptions = [
        "Horizontal line",
        "Slightly tilted line",
        "More tilted line", 
        "Vertical line",
        "Rectangle"
    ]
    
    for i, (polygon, desc) in enumerate(zip(test_polygons, descriptions)):
        print(f"\n{i+1}. {desc}: {polygon}")
        angles = compute_all_rotation_methods(polygon)
        print(f"   Rotation angles: {angles}")
        
        # Test horizontality with different methods
        for method in ['leftright_points', 'pca', 'min_area_rect', 'least_squares', 'top_bottom_edges']:
            try:
                is_horiz = is_horizontal_line(polygon, min_aspect_ratio=1.0, min_width=20, max_angle=30, rotation_method=method)
                print(f"   {method}: horizontal = {is_horiz}")
            except Exception as e:
                print(f"   {method}: ERROR - {e}")
    
    print("\nRotation method testing complete.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_rotation_methods()
    else:
        main() 