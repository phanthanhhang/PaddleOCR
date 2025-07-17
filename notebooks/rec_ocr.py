import sys
sys.path.append("/data1/hang/Stellantis/PaddleOCR")
from paddleocr import TextRecognition
import time
import cv2  
import numpy as np
import os
import random
model = TextRecognition(model_dir = "/data1/hang/Stellantis/PaddleOCR/output/export_model_german_custom_PP-OCRv5_server_rec_17072025")
paths_gt = [l.strip().split("\t") for l in open("/mnt/ssd1/hang/OCR_Recoginition_data/line_boxes_dataset/test_split/test_00.txt", "r").readlines()]

index = random.randint(0, len(paths_gt) - 1)

path = paths_gt[index][0]
gt = paths_gt[index][1]

# path = "/mnt/ssd1/hang/OCR_Recoginition_data/line_boxes_dataset/images/ZBII__9376951832_miscellaneous_006f3eeda9ad804c685f32cadc9a7863_page_11__horizontal_line_056.jpg"

print(path)
print('gt: ', gt)

output = model.predict(path, batch_size=1)
for res in output:
    res.print()