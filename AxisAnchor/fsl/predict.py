import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from AxisAnchor.base.utils.logger import setup_logger
from AxisAnchor.fsl.config import cfg
from AxisAnchor.fsl.dataset.build import build_transform
from AxisAnchor.fsl.model.build import build_model
import cv2

class ImageList(IterableDataset):
    def __init__(self, image_paths, transform):
        super().__init__()
        self.image_paths = image_paths
        self.transform = transform

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError("Single worker only.")
        for image_path in self.image_paths:
            im = Image.open(image_path)
            w, h = im.size
            meta = {
                "filename": image_path,
                "height": h,
                "width": w,
            }
            yield self.transform(np.array(im)), meta


def parse_args():
    parser = argparse.ArgumentParser(description="HAWP Testing")
    parser.add_argument("config", help="the path of config file")
    parser.add_argument("images_dir", help="the directory containing images")
    parser.add_argument("--ckpt", type=str, required=True)

    parser.add_argument(
        "--j2l", default=None, type=float, help="the threshold for junction-line attraction"
    )
    parser.add_argument("--rscale", default=2, type=int, help="the residual scale")

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output", default=None, help="the path of outputs")

    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def bresenham(x1, y1, x2, y2):
    pixels = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    while True:
        pixels.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return pixels

def get_line_s(line, s_matrix, height, width):
    x1, y1, x2, y2 = line

    x1 = int(round(float(x1) * 127 / (width - 1)))
    y1 = int(round(float(y1) * 127 / (height - 1)))
    x2 = int(round(float(x2) * 127 / (width - 1)))
    y2 = int(round(float(y2) * 127 / (height - 1)))

    line_pixels = bresenham(x1, y1, x2, y2)

    s_values = []
    for x, y in line_pixels:
        if 0 <= x < s_matrix.shape[1] and 0 <= y < s_matrix.shape[0]:
            s_value = s_matrix[y, x]
            s_values.append(s_value)

    line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    if s_values:
        avg_s_normalized = np.mean(s_values)
        actual_s = avg_s_normalized * line_length
    else:
        actual_s = 0.0

    return actual_s

def line_to_rotated_bbox(line, s, height, width):

    x1, y1, x2, y2 = line
    x1 = float(x1)
    y1 = float(y1)
    x2 = float(x2)
    y2 = float(y2)

    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # 弧度转角度

    return [x_center, y_center, line_length, s, angle]


def rotated_rect_iou(rect_a, rect_b):

    width_a, height_a = rect_a[1]
    width_b, height_b = rect_b[1]

    # 计算旋转矩形的交集区域
    result, inter_pts = cv2.rotatedRectangleIntersection(rect_a, rect_b)

    if result == cv2.INTERSECT_NONE:
        return 0.0
    elif result == cv2.INTERSECT_FULL:
        return 1.0

    if inter_pts is not None and len(inter_pts) > 2:
        inter_area = cv2.contourArea(inter_pts)
    else:
        inter_area = 0.0

    area_a = width_a * height_a
    area_b = width_b * height_b

    return inter_area / (area_a + area_b - inter_area)


def NMSBoxes(bboxes, scores, score_threshold, nms_threshold, eta=1, top_k=0):

    score_index_vec = [(score, idx) for idx, score in enumerate(scores) if score >= score_threshold]
    score_index_vec.sort(reverse=True, key=lambda x: x[0])

    if top_k > 0:
        score_index_vec = score_index_vec[:top_k]

    indices = []
    adaptive_threshold = nms_threshold

    for i in range(len(score_index_vec)):
        idx = score_index_vec[i][1]
        keep = True

        for kept_idx in indices:
            overlap = rotated_rect_iou(bboxes[idx], bboxes[kept_idx])
            if overlap > adaptive_threshold:
                keep = False
                break

        if keep:
            indices.append(idx)

            if eta < 1 and adaptive_threshold > 0.5:
                adaptive_threshold *= eta

    return indices


def process_lines_with_nms(output, nms_threshold=0.1, score_threshold=0.9):

    lines_pred = output["lines_pred"]
    lines_score = output["lines_score"].cpu().tolist()
    s_matrix = output["s"].sigmoid().cpu().numpy()[0, 0]
    height = int(output["height"].item())
    width = int(output["width"].item())
    filename = output["filename"]
    lines_pred_list = lines_pred.cpu().tolist()

    lines_pred_list = lines_pred.cpu().tolist()
    rotated_bboxes = []
    for line in lines_pred_list:
        s = get_line_s(line, s_matrix, height, width)
        rotated_bbox = line_to_rotated_bbox(line, s, height, width)
        rotated_bboxes.append(rotated_bbox)

    rotated_rects = []
    for bbox in rotated_bboxes:
        center = (bbox[0], bbox[1])
        size = (bbox[2], bbox[3])
        angle = bbox[4]
        rotated_rect = (center, size, angle)
        rotated_rects.append(rotated_rect)

    indices = NMSBoxes(rotated_rects, lines_score, score_threshold, nms_threshold)

    final_bboxes = [rotated_bboxes[i] for i in indices]
    final_scores = [lines_score[i] for i in indices]
    final_lines = [lines_pred_list[i] for i in indices]

    return {
        "filename": filename,
        "height": height,
        "width": width,
        "lines": final_lines,
        "line_scores": final_scores,
        "rotated_bboxes": final_bboxes,
        "lines_raw": lines_pred_list,
    }


def main():
    args = parse_args()

    config_path = args.config
    cfg.merge_from_file(config_path)

    root = args.output
    if root is None:
        root = str(Path(args.ckpt).parent)

    logger = setup_logger("Insulator.predict", root)
    logger.info(args)
    logger.info(f"Loaded configuration file {config_path}")

    set_random_seed(args.seed)

    device = cfg.MODEL.DEVICE
    logger.info(f"Running on device {device}")
    model = build_model(cfg).to(device)

    if args.rscale is not None:
        model.use_residual = args.rscale

    if args.j2l:
        model.j2l_threshold = args.j2l

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.eval()


    transform = build_transform(cfg)

    image_paths = [os.path.join(args.images_dir, filename) for filename in os.listdir(args.images_dir) if filename.lower().endswith(('.jpg', '.png'))]

    dataset = ImageList(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True)

    outputs = []
    timings = defaultdict(float)
    # time1 = time.time()
    for tensor, meta in tqdm(dataloader, total=len(image_paths)):
        with torch.no_grad():
            output, extra_info = model(tensor.to(device), [meta])

        if output is not None:
            outputs.append(process_lines_with_nms(output))
            for key, value in extra_info.items():
                timings[key] += value

    # time2 = time.time()
    # print(time2-time1)
    logger.info(f"Timings : {dict(timings)}")

    out_path = Path(root) / "Insulator.json"
    logger.info(f"Writing outputs to {out_path}")
    with out_path.open("w") as f:
        json.dump(outputs, f, indent=4)

    print(" saved to Insulator.json")


if __name__ == "__main__":
    main()
