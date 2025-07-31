#!/usr/bin/env python3

import os
import math
import random
from collections import defaultdict

def split_file_balanced(file_path, output_dir, num_parts=16):
    """
    Split a file into equal parts with balanced distribution of document types.
    Only include lines where the corresponding image file exists.
    
    Args:
        file_path: Path to the input file
        output_dir: Directory to save the split files
        num_parts: Number of parts to split into (default 16)
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get file name without extension
    file_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_name)[0]
    
    # Old and new path prefixes
    old_prefix = "/data1/hang/Stellantis/PaddleOCR/notebooks/Recognition/line_boxes_dataset/images/"
    new_prefix = "/mnt/ssd1/hang/OCR_Recoginition_data/line_boxes_dataset/images/"
    target_images_dir = "/mnt/ssd1/hang/OCR_Recoginition_data/line_boxes_dataset/images"
    
    # Read all lines and group by document type
    print(f"Reading and categorizing lines from {file_name}...")
    doc_types = defaultdict(list)
    total_lines_read = 0
    missing_images = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines_read += 1
            
            # Update image path
            original_line = line
            if line.startswith(old_prefix):
                line = line.replace(old_prefix, new_prefix, 1)
            
            # Extract image path from the line (first part before the tab/space)
            parts = line.strip().split('\t')
            if len(parts) < 2:
                parts = line.strip().split(' ', 1)
            
            if len(parts) >= 2:
                image_path = parts[0]
                
                # Check if image file exists
                if os.path.exists(image_path):
                    # Determine document type from the image path
                    if "VDN__" in line:
                        doc_types["VDN"].append(line)
                    elif "ZBII__" in line:
                        doc_types["ZBII"].append(line)
                    elif "INVOICE__" in line:
                        doc_types["INVOICE"].append(line)
                    elif "ID__" in line:
                        doc_types["ID"].append(line)
                    else:
                        # If no clear document type, assign to a general category
                        doc_types["OTHER"].append(line)
                else:
                    # Image file doesn't exist, record it
                    missing_images.append(image_path)
                    if len(missing_images) <= 10:  # Only print first 10 missing images
                        print(f"WARNING: Image not found: {image_path}")
    
    # Print distribution and missing images summary
    valid_lines = sum(len(lines) for lines in doc_types.values())
    print(f"Total lines read: {total_lines_read}")
    print(f"Valid lines (with existing images): {valid_lines}")
    print(f"Missing images: {len(missing_images)}")
    
    if missing_images:
        print(f"First few missing images:")
        for i, missing_path in enumerate(missing_images[:5]):
            print(f"  {i+1}. {missing_path}")
        if len(missing_images) > 5:
            print(f"  ... and {len(missing_images) - 5} more")
    
    for doc_type, lines in doc_types.items():
        print(f"{doc_type}: {len(lines)} lines")
    
    # Randomly shuffle each document type
    print("Shuffling document types...")
    for doc_type in doc_types:
        random.shuffle(doc_types[doc_type])
    
    # Initialize output files
    output_files = {}
    for part_num in range(num_parts):
        part_file = os.path.join(output_dir, f"{base_name}_{part_num:02d}.txt")
        output_files[part_num] = open(part_file, 'w', encoding='utf-8')
    
    # Distribute each document type equally across parts
    print("Distributing lines across parts...")
    part_counts = [0] * num_parts
    
    for doc_type, lines in doc_types.items():
        if not lines:
            continue
            
        lines_per_part = len(lines) // num_parts
        remainder = len(lines) % num_parts
        
        print(f"Distributing {doc_type}: {lines_per_part} lines per part, {remainder} extra")
        
        line_idx = 0
        for part_num in range(num_parts):
            # Calculate how many lines this part gets
            lines_for_this_part = lines_per_part
            if part_num < remainder:
                lines_for_this_part += 1
            
            # Write lines to this part
            for i in range(lines_for_this_part):
                if line_idx < len(lines):
                    output_files[part_num].write(lines[line_idx])
                    part_counts[part_num] += 1
                    line_idx += 1
    
    # Close all output files
    for part_num in range(num_parts):
        output_files[part_num].close()
        part_file = os.path.join(output_dir, f"{base_name}_{part_num:02d}.txt")
        print(f"Created {part_file} with {part_counts[part_num]} lines")
    
    return len(missing_images)

def verify_split_results(output_dir, base_name, num_parts=16):
    """
    Verify that all image paths in the split files exist.
    """
    print(f"\nVerifying split results in {output_dir}...")
    
    for part_num in range(num_parts):
        part_file = os.path.join(output_dir, f"{base_name}_{part_num:02d}.txt")
        
        if not os.path.exists(part_file):
            print(f"WARNING: Part file not found: {part_file}")
            continue
        
        missing_in_part = 0
        total_in_part = 0
        
        with open(part_file, 'r', encoding='utf-8') as f:
            for line in f:
                total_in_part += 1
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    parts = line.strip().split(' ', 1)
                
                if len(parts) >= 2:
                    image_path = parts[0]
                    if not os.path.exists(image_path):
                        missing_in_part += 1
        
        if missing_in_part > 0:
            print(f"WARNING: {part_file} has {missing_in_part}/{total_in_part} missing images")
        else:
            print(f"✓ {part_file}: All {total_in_part} images exist")

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define paths
    dataset_dir = "/mnt/ssd1/hang/OCR_Recoginition_data/line_boxes_dataset"
    train_file = os.path.join(dataset_dir, "train.txt")
    test_file = os.path.join(dataset_dir, "test.txt")
    
    # Create output directories
    train_output_dir = os.path.join(dataset_dir, "train_split")
    test_output_dir = os.path.join(dataset_dir, "test_split")
    
    # Check if images directory exists
    images_dir = os.path.join(dataset_dir, "images")
    if not os.path.exists(images_dir):
        print(f"ERROR: Images directory not found: {images_dir}")
        return
    
    print(f"Images directory exists: {images_dir}")
    
    # Split train.txt
    print("Splitting train.txt with balanced distribution...")
    train_missing = split_file_balanced(train_file, train_output_dir, 16)
    
    # Split test.txt
    print("\nSplitting test.txt with balanced distribution...")
    test_missing = split_file_balanced(test_file, test_output_dir, 16)
    
    print(f"\nSplitting complete!")
    print(f"Total missing images in train.txt: {train_missing}")
    print(f"Total missing images in test.txt: {test_missing}")
    
    # Verify the results
    verify_split_results(train_output_dir, "train", 16)
    verify_split_results(test_output_dir, "test", 16)
    
    print("\nAll split files have been verified. Only lines with existing images are included.")

if __name__ == "__main__":
    main()