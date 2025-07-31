import sys
sys.path.append("/data1/hang/Stellantis/PaddleOCR")
from paddleocr import PaddleOCR
import time
import cv2  
import numpy as np
import os
import random
ocr = PaddleOCR(
    text_detection_model_name = "PP-OCRv5_server_det",
    text_recognition_model_dir = "/data1/hang/Stellantis/PaddleOCR/output/export_model_german_custom_PP-OCRv5_server_rec_19072025",
    # text_detection_model_name = "ch_PP-OCRv5_det",
    use_doc_orientation_classify=True,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    # # lang = "german",
    # text_det_limit_side_len=640,
    # text_det_limit_type="max"
    )
img_paths = [l.strip() for l in open("/data1/hang/Stellantis/PaddleOCR/notebooks/Recognition/whole_doc_img_paths_00.txt").readlines()]
# img_path = "/data1/hang/Stellantis/VDN_annotated/9377075037_VDN_20250510165525_1.jpg"
# img_path = "/data1/stellantis/images/ID/9376717257_miscellaneous_93c03169ef11a2406ba577361c46e554_page_0.jpeg"
# img_path = "/data1/stellantis/images/ID/9376271786_IDD1_20250428_page_0.jpeg"
# img_path = "/data1/stellantis/images/VDN/9376878657_miscellaneous_43265ee846048c623c971559d614d8e1.pdf_page_3.jpeg"
# img_path = "/data1/stellantis/images/ID/937643835_IDD1_20250508_page_0.jpeg"
# img_path = "/data1/stellantis/images/VDN/9376836987_miscellaneous_eb40da98e206dfbc3d10e2ffe41f16d5_page_30.jpeg"
# img_path = "/data1/stellantis/images/VDN/9376836247_miscellaneous_c6afb1241e5362d130259a634c4410c5.pdf_page_25.jpeg"
# img_path = "/data1/stellantis/images/VDN/9176789007_miscellaneous_ff23a60e32fbf0d2d94650bd11add2df_page_21.jpeg"
# img_path = "/data1/stellantis/images/VDN/9376716467_miscellaneous_474a033f2f131487595b7179369e8aea_page_26.jpeg"
# img_path = "/data1/stellantis/images/VDN/9376701227_miscellaneous_7cf2545ee8d06296cc71e95f48c318fa.pdf_page_12.jpeg"
# img_path = "/data1/stellantis/images/INVOICE/9376967051_miscellaneous_860718460bdadd2d463ac6b8ffe1c9a6.pdf_page_0.jpeg"
# img_path = "/data1/hang/Stellantis/VDN_annotated/9376869727_miscellaneous_7c59c6e1ec34fe958c66f4f749653244_page_19.jpeg"
# img_path = "/data1/hang/Stellantis/VDN_annotated/9376600486_miscellaneous_d168fdf3855caf5de6408e603b484cbc_page_23.jpeg"

# img_path = "/data1/stellantis/images/ZBII/9376052411_miscellaneous_b2bcd6610e4529f7f5dc91a5d2766a2c_page_3.jpeg"


index = random.randint(0, len(img_paths))
img_path = img_paths[index]
# img_path = "/data1/stellantis/images/ZBII/9175954842_miscellaneous_7f251bf7ee973264ad52bf56237950f4.pdf_page_34.jpeg"

time_start = time.time()
result = ocr.predict(input=img_path, text_rec_score_thresh=0.7)
time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")
for res in result:
    print('keys:', res.keys())
    angle = int(res['doc_preprocessor_res']['angle'])
    print(f"number of texts: {len(res['rec_texts'])}")
    print(f"number of polygons: {len(res['dt_polys'])}")
    print(f"number of rec_polys: {len(res['rec_polys'])}")
    print(f"number of rec_boxes: {len(res['rec_boxes'])}")
    text_rect_scores = res['rec_scores']
    
    input_img = cv2.imread(res['input_path'])
    if angle == 90:
            # Rotate 90 degrees counterclockwise to correct
            print('rotate 90 degrees counterclockwise')
            corrected_img = cv2.rotate(input_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
            # Rotate 180 degrees to correct
            print('rotate 180 degrees')
            corrected_img = cv2.rotate(input_img, cv2.ROTATE_180)
    elif angle == 270:
            # Rotate 90 degrees clockwise to correct (or 270 counterclockwise)
            print('rotate 90 degrees clockwise')
            corrected_img = cv2.rotate(input_img, cv2.ROTATE_90_CLOCKWISE)
    else:
        corrected_img = input_img
        
    # Draw bounding boxes on the corrected image
    img_with_boxes = corrected_img.copy()
    rec_boxes = res['rec_boxes']
    rec_texts = res['rec_texts']
    rec_scores = res['rec_scores']
    
    # Draw each bounding box
    for i, box in enumerate(rec_boxes):
        # rec_boxes format: [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = box
        
        # Draw rectangle (image, top_left, bottom_right, color, thickness)
        cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Add recognized text and score above the box
        if i < len(rec_texts) and i < len(rec_scores):
            text = rec_texts[i]
            score = rec_scores[i]
            # Format: "text (confidence: 0.95)"
            display_text = f"{text} ({score:.2f})"
            
            # Use smaller font size and add background for better readability
            font_scale = 0.4
            thickness = 1
            
            # Get text size to create background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(img_with_boxes, 
                         (x_min, y_min - text_height - baseline - 5), 
                         (x_min + text_width, y_min - 2), 
                         (0, 255, 0), -1)
            
            # Draw text on background
            cv2.putText(img_with_boxes, display_text, (x_min, y_min - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    
    # Save the image with boxes
    output_path = "./output_german/corrected_img_with_boxes.jpg"
    cv2.imwrite(output_path, img_with_boxes)
    print(f"Image with bounding boxes saved to: {output_path}")
    
    # Draw detection polygons on a separate image
    img_with_polys = corrected_img.copy()
    dt_polys = res['dt_polys']
    
    # Draw each detection polygon
    for i, poly in enumerate(dt_polys):
        # Convert polygon points to numpy array format for cv2.polylines
        pts = np.array(poly, np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Draw polygon outline (isClosed=True for closed polygon)
        cv2.polylines(img_with_polys, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        
        # Optional: Fill polygon with semi-transparent color
        # cv2.fillPoly(img_with_polys, [pts], color=(255, 0, 0, 50))
        
        # Add polygon index at the centroid
        M = cv2.moments(pts)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(img_with_polys, str(i), (cx, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Save the image with detection polygons
    poly_output_path = "./output_german/corrected_img_with_dt_polys.jpg"
    cv2.imwrite(poly_output_path, img_with_polys)
    print(f"Image with detection polygons saved to: {poly_output_path}")
    
    res.print()
    res.save_to_img("./output_german/")
    # print save image path
    print(f"Image saved to: ./output_german/{os.path.basename(img_path).replace('.jpeg', '_ocr_res_img.jpeg')}")
    res.save_to_json("./output_german/res.json")