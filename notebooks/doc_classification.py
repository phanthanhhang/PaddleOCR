import sys
sys.path.append("/data1/hang/Stellantis/PaddleOCR")
from paddleocr import DocImgOrientationClassification
from pprint import pprint
import time
import cv2
import os

model = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")
img_path = "/data1/hang/Stellantis/VDN_annotated/9377075037_VDN_20250510165525_1.jpg"
time_start = time.time()
output = model.predict(img_path, batch_size=1)
time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")
for res in output:
    input_path = res['input_path']
    input_img = res['input_img']
    print(input_img.shape)
    rotate = int(res['label_names'][0])
    conf = res['scores'][0]
    print(f"rotate: {rotate}, conf: {conf}")
    print('--------------------------------')
    res.print(json_format=False)
    
    # If rotation is needed, correct the image and save it
    if rotate != 0:
        # Get the base filename and directory
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        ext = os.path.splitext(os.path.basename(input_path))[1]
        output_filename = f"{base_name}_rotated{ext}"
        
        # Apply rotation correction based on detected angle
        if rotate == 90:
            # Rotate 90 degrees counterclockwise to correct
            print('rotate 90 degrees counterclockwise')
            corrected_img = cv2.rotate(input_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotate == 180:
            # Rotate 180 degrees to correct
            print('rotate 180 degrees')
            corrected_img = cv2.rotate(input_img, cv2.ROTATE_180)
        elif rotate == 270:
            # Rotate 90 degrees clockwise to correct (or 270 counterclockwise)
            print('rotate 90 degrees clockwise')
            corrected_img = cv2.rotate(input_img, cv2.ROTATE_90_CLOCKWISE)
        else:
            # For other angles, use general rotation
            print('rotate other angles')
            height, width = input_img.shape[:2]
            center = (width // 2, height // 2)
            c = cv2.getRotationMatrix2D(center, -rotate, 1.0)
            corrected_img = cv2.warpAffine(input_img, c, (width, height))
        
        # Convert from RGB to BGR for cv2.imwrite (PaddleOCR uses RGB, OpenCV expects BGR)
        corrected_img_bgr = cv2.cvtColor(corrected_img, cv2.COLOR_RGB2BGR)
        
        # Save the corrected image to current directory
        cv2.imwrite(output_filename, corrected_img_bgr)
        print(f"Corrected image saved as: {output_filename}")
    
    # res.save_to_img("./output/demo.png")
    # res.save_to_json("./output/res.json")