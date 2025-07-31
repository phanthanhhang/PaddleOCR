import json
import cv2
import numpy as np
import os
from glob import glob
from pathlib import Path
import traceback

def load_train_test_splits(split_dir):
    """Load train/test split information from files."""
    splits = {'train': {}, 'test': {}}
    
    # Categories to process
    categories = ['ID', 'INVOICE', 'VDN', 'ZBII', 'OTHERS']
    
    for category in categories:
        # Load train files
        train_file = os.path.join(split_dir, f'train_{category}.txt')
        if os.path.exists(train_file):
            with open(train_file, 'r') as f:
                train_files = [line.strip() for line in f.readlines() if line.strip()]
                splits['train'][category] = set(train_files)
        else:
            splits['train'][category] = set()
        
        # Load test files
        test_file = os.path.join(split_dir, f'test_{category}.txt')
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                test_files = [line.strip() for line in f.readlines() if line.strip()]
                splits['test'][category] = set(test_files)
        else:
            splits['test'][category] = set()
    
    return splits

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

def get_original_rotation_info(ocr_data):
    """Get original rotation information from OCR data."""
    if not ocr_data:
        return None
    
    # Handle different OCR data structures
    pages = None
    
    # Try different possible structures
    if 'pages' in ocr_data:
        pages = ocr_data.get('pages', [])
    elif 'analyzeResult' in ocr_data and 'pages' in ocr_data['analyzeResult']:
        pages = ocr_data['analyzeResult'].get('pages', [])
    elif 'result' in ocr_data and 'pages' in ocr_data['result']:
        pages = ocr_data['result'].get('pages', [])
    
    if pages and len(pages) > 0:
        page = pages[0]
        return {
            'angle': page.get('angle', 0),
            'width': page.get('width', 0),
            'height': page.get('height', 0),
            'unit': page.get('unit', 'pixel')
        }
    return None

def determine_best_rotation(original_angle):
    """Determine the best rotation angle to align text properly."""
    print(f"Original detected rotation: {original_angle}°")
    
    # Normalize angle to be between -180 and 180 degrees
    normalized_angle = original_angle % 360
    if normalized_angle > 180:
        normalized_angle -= 360
    
    print(f"Normalized angle: {normalized_angle}°")
    
    # Determine the best correction angle
    if abs(normalized_angle) <= 45:
        # Close to 0 degrees - no rotation needed
        correction_angle = 0
        print("Text is already properly aligned (close to 0°)")
    elif normalized_angle > 45 and normalized_angle <= 135:
        # Rotate 90 degrees counterclockwise
        correction_angle = 90
        print("Applying 90° counterclockwise rotation")
    elif normalized_angle > 135 or normalized_angle <= -135:
        # Rotate 180 degrees
        correction_angle = 180
        print("Applying 180° rotation")
    elif normalized_angle > -135 and normalized_angle <= -45:
        # Rotate 90 degrees clockwise
        correction_angle = -90
        print("Applying 90° clockwise rotation")
    else:
        # Default to no rotation
        correction_angle = 0
        print("Defaulting to no rotation")
    
    return correction_angle

def rotate_image_and_coordinates(image, angle_degrees, ocr_data):
    """Rotate image and transform OCR coordinates accordingly."""
    if image is None or angle_degrees == 0:
        return image, ocr_data
    
    print(f"Rotating image by {angle_degrees}°")
    
    # Get image dimensions
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    
    # Calculate new image dimensions
    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])
    new_width = int((height * sin_val) + (width * cos_val))
    new_height = int((height * cos_val) + (width * sin_val))
    
    # Adjust rotation matrix for new dimensions
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # Rotate image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
    
    # Transform OCR coordinates
    transformed_ocr_data = transform_ocr_coordinates(ocr_data, rotation_matrix, width, height, new_width, new_height)
    
    return rotated_image, transformed_ocr_data

def transform_ocr_coordinates(ocr_data, rotation_matrix, orig_width, orig_height, new_width, new_height):
    """Transform OCR polygon coordinates based on rotation matrix."""
    if not ocr_data:
        return ocr_data
    
    # Create a copy of the OCR data
    transformed_data = json.loads(json.dumps(ocr_data))
    
    # Handle different OCR data structures
    pages = None
    if 'pages' in transformed_data:
        pages = transformed_data['pages']
    elif 'analyzeResult' in transformed_data and 'pages' in transformed_data['analyzeResult']:
        pages = transformed_data['analyzeResult']['pages']
    elif 'result' in transformed_data and 'pages' in transformed_data['result']:
        pages = transformed_data['result']['pages']
    
    if not pages:
        return transformed_data
    
    def transform_polygon(polygon):
        """Transform a polygon using the rotation matrix."""
        if len(polygon) >= 8:
            # Reshape polygon to (n, 2) format
            points = np.array(polygon).reshape(-1, 2)
            
            # Transform each point
            transformed_points = []
            for point in points:
                # Apply rotation transformation
                x, y = point[0], point[1]
                new_x = rotation_matrix[0, 0] * x + rotation_matrix[0, 1] * y + rotation_matrix[0, 2]
                new_y = rotation_matrix[1, 0] * x + rotation_matrix[1, 1] * y + rotation_matrix[1, 2]
                transformed_points.extend([new_x, new_y])
            
            return transformed_points
        return polygon
    
    for page in pages:
        # Transform words polygons
        page_words = page.get('words', [])
        for word in page_words:
            polygon = word.get('polygon', [])
            if polygon:
                word['polygon'] = transform_polygon(polygon)
        
        # Transform lines polygons
        page_lines = page.get('lines', [])
        for line in page_lines:
            polygon = line.get('polygon', [])
            if polygon:
                line['polygon'] = transform_polygon(polygon)
    
    # Update page dimensions and reset angle
    if pages:
        pages[0]['width'] = new_width
        pages[0]['height'] = new_height
        pages[0]['angle'] = 0  # Reset angle after correction
    
    return transformed_data

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

def compute_rotation_pca(polygon):
    """Compute rotation using Principal Component Analysis (manual implementation)."""
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

def is_horizontal_line(polygon, min_aspect_ratio=1.0, min_width=20, max_angle=15):
    """
    Determine if a line is horizontal based on its polygon coordinates.
    
    Args:
        polygon: List of polygon coordinates
        min_aspect_ratio: Minimum width/height ratio to consider horizontal
        min_width: Minimum width in pixels to consider as a valid line
        max_angle: Maximum angle in degrees from horizontal (0°) to consider as horizontal
    
    Returns:
        bool: True if the line is horizontal, False otherwise
    """
    bbox = get_polygon_bbox(polygon)
    if bbox is None:
        return False
    
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    
    # Avoid division by zero
    if height == 0:
        return width >= min_width
    
    aspect_ratio = width / height
    
    # Check basic aspect ratio and minimum width first
    if aspect_ratio < min_aspect_ratio or width < min_width:
        return False
    
    # Calculate angle using PCA method
    try:
        angle = compute_rotation_pca(polygon)
        
        # Check if angle is within acceptable range for horizontal lines
        is_horizontal = angle <= max_angle
        
        return is_horizontal
    except Exception as e:
        print(f"Error computing rotation: {e}")
        return False

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

def test_single_image():
    """Test the line extraction on a single specific image."""
    # Specific image to test
    test_image_filename = "9175601242_miscellaneous_fbc94bac6c4c1d3bd05d89b302a45323_page_3.jpeg"
    
    # Paths
    annotation_base = "/data1/hang/Stellantis/text_recognition/annotations"
    image_base = "/data1/stellantis/images"
    split_dir = "/data1/hang/Stellantis/text_recognition/train_test_filenames"
    output_base = "/data1/hang/Stellantis/PaddleOCR/notebooks/Recognition/test_single_output"
    
    # Create output directory
    os.makedirs(output_base, exist_ok=True)
    
    # Load train/test splits
    print("Loading train/test splits...")
    splits = load_train_test_splits(split_dir)
    
    # Test with ZBII folder
    folder_name = "ZBII"
    annotation_folder = os.path.join(annotation_base, folder_name)
    image_folder = os.path.join(image_base, folder_name)
    
    # Construct paths
    image_path = os.path.join(image_folder, test_image_filename)
    json_filename = test_image_filename.replace('.jpeg', '.json')
    json_path = os.path.join(annotation_folder, json_filename)
    
    print(f"\nTesting single image: {test_image_filename}")
    print(f"Image path: {image_path}")
    print(f"JSON path: {json_path}")
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return
    
    if not os.path.exists(json_path):
        print(f"ERROR: JSON annotation not found at {json_path}")
        return
    
    # Load OCR data
    print("\nLoading OCR data...")
    ocr_data = load_ocr_data(json_path)
    if ocr_data is None:
        print("ERROR: Failed to load OCR data")
        return
    
    # Load image
    print("Loading image...")
    image = load_image(image_path)
    if image is None:
        print("ERROR: Failed to load image")
        return
    
    print(f"Original image shape: {image.shape}")
    
    # Check train/test split
    is_train = test_image_filename in splits['train'].get(folder_name, set())
    is_test = test_image_filename in splits['test'].get(folder_name, set())
    
    print(f"Image is in train set: {is_train}")
    print(f"Image is in test set: {is_test}")
    
    # Get rotation information and apply correction
    print("\nAnalyzing rotation...")
    original_rotation = get_original_rotation_info(ocr_data)
    if original_rotation:
        print(f"Original rotation info: {original_rotation}")
        original_angle = original_rotation['angle']
        best_rotation = determine_best_rotation(original_angle)
        
        # Apply rotation correction
        if best_rotation != 0:
            print(f"Applying rotation correction of {best_rotation}°")
            image, ocr_data = rotate_image_and_coordinates(image, best_rotation, ocr_data)
            if image is None:
                print("ERROR: Failed to rotate image")
                return
            print(f"Rotated image shape: {image.shape}")
        else:
            print("No rotation correction needed")
    else:
        print("No rotation information found, proceeding without rotation")
    
    # Extract lines
    print("\nExtracting lines...")
    pages = ocr_data.get('pages', [])
    if not pages:
        print("ERROR: No pages found in OCR data")
        return
    
    page = pages[0]
    lines = page.get('lines', [])
    
    if not lines:
        print("ERROR: No lines found in OCR data")
        return
    
    print(f"Total lines found: {len(lines)}")
    
    # Filter horizontal lines
    horizontal_lines = []
    for line_idx, line in enumerate(lines):
        polygon = line.get('polygon', [])
        content = line.get('content', '').strip()
        
        if not polygon or not content:
            continue
        
        # Skip very long content or content with 'ppa'
        if len(content) > 100 or 'ppa' in content.lower():
            continue
        
        # Adjust parameters based on content length
        if len(content) < 3:
            min_aspect_ratio = 0.6
            min_width = 10
            max_angle = 20
        else:
            min_aspect_ratio = 1.0
            min_width = 20
            max_angle = 15
        
        # Check if line is horizontal
        if is_horizontal_line(polygon, min_aspect_ratio=min_aspect_ratio, min_width=min_width, max_angle=max_angle):
            horizontal_lines.append((line_idx, line))
            print(f"  Horizontal line {line_idx}: '{content[:50]}...' (len={len(content)})")
    
    print(f"\nHorizontal lines found: {len(horizontal_lines)} out of {len(lines)} total lines")
    
    # Extract and save line boxes
    extracted_count = 0
    for line_idx, line in horizontal_lines:
        polygon = line.get('polygon', [])
        content = line.get('content', '').strip()
        
        # Crop line from image
        cropped_line = crop_line_from_image(image, polygon, padding=5)
        
        if cropped_line is None or cropped_line.size == 0:
            print(f"  Failed to crop line {line_idx}")
            continue
        
        # Create output filename
        image_name = os.path.splitext(test_image_filename)[0]
        output_filename = f"{folder_name}__{image_name}__horizontal_line_{line_idx:03d}.jpg"
        output_filename = sanitize_filename(output_filename)
        output_path = os.path.join(output_base, output_filename)
        
        # Save cropped line
        try:
            cv2.imwrite(output_path, cropped_line)
            extracted_count += 1
            print(f"  Saved line {line_idx}: {output_filename} (shape: {cropped_line.shape})")
        except Exception as e:
            print(f"  Error saving line {line_idx}: {e}")
    
    print(f"\nTest completed!")
    print(f"Successfully extracted {extracted_count} horizontal line boxes")
    print(f"Output directory: {output_base}")
    print(f"Total files saved: {len(os.listdir(output_base))}")

if __name__ == "__main__":
    test_single_image() 