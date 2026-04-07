import json
import cv2
import numpy as np
import os

def draw_rotated_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    cx, cy, width, height, angle = bbox
    angle = angle
    rect = ((cx, cy), (width, height), angle)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(image, [box], 0, color, thickness)


json_file_path = "/home/tyb/home/tyb/project/AxisAnchor-main/Insulator.json"
output_dir = "/home/tyb/home/tyb/project/AxisAnchor-main/result"
os.makedirs(output_dir, exist_ok=True)


score_threshold = 0.9

with open(json_file_path, 'r') as f:
    data = json.load(f)

for entry in data:
    image_path = entry["filename"][0]
    height = entry["height"]
    width = entry["width"]
    rotated_bboxes = entry.get("rotated_bboxes", [])
    line_scores = entry.get("line_scores", [])


    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图片: {image_path}")
        continue

    image = cv2.resize(image, (width, height))


    for bbox, score in zip(rotated_bboxes, line_scores):
        if score > score_threshold:
            draw_rotated_bbox(image, bbox, color=(0, 0, 255))


    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"处理后的图片已保存到: {output_path}")
