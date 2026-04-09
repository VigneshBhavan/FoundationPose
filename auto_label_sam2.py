#!/usr/bin/env python3
"""Auto-label gripper images using YOLO (rough box) + SAM2 (precise polygon).

For each image:
1. YOLO detects the gripper → bounding box
2. Box center is used as a SAM2 point prompt
3. SAM2 produces a precise segmentation mask
4. Mask contour is saved as a labelme polygon annotation

Images that already have a .json label are skipped (unless --overwrite).
Images where YOLO finds nothing are flagged for manual labeling.

Usage:
    python auto_label_sam2.py
    python auto_label_sam2.py --overwrite   # re-label everything
    python auto_label_sam2.py --yolo_only   # skip SAM2, just use YOLO boxes
"""

import argparse
import base64
import glob
import json
import os

import cv2
import numpy as np
import osam.apis
import osam.types
from ultralytics import YOLO


def mask_to_polygon(mask, simplify_eps=2.0):
    """Convert a binary mask to the largest polygon (list of [x, y] points)."""
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Take the largest contour
    largest = max(contours, key=cv2.contourArea)
    # Simplify to reduce point count
    approx = cv2.approxPolyDP(largest, simplify_eps, True)
    points = approx.reshape(-1, 2).tolist()

    if len(points) < 3:
        return None

    return [[float(x), float(y)] for x, y in points]


def make_labelme_json(image_path, img_bgr, shapes):
    """Create a labelme-compatible annotation dict."""
    h, w = img_bgr.shape[:2]

    _, buf = cv2.imencode(".png", img_bgr)
    image_data = base64.b64encode(buf).decode("utf-8")

    annotation = {
        "version": "5.11.4",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": image_data,
        "imageHeight": h,
        "imageWidth": w,
    }
    return annotation


def box_to_polygon(box):
    """Convert a YOLO xyxy box to a labelme rectangle polygon."""
    x1, y1, x2, y2 = [float(v) for v in box]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def main():
    parser = argparse.ArgumentParser(
        description="Auto-label gripper images with YOLO + SAM2."
    )
    parser.add_argument("--images_dir", type=str, default="gripper_images",
                        help="Directory containing .png images to label.")
    parser.add_argument("--yolo_model", type=str,
                        default="gripper_dataset/runs/gripper_detect11/weights/best.pt",
                        help="Path to YOLO model for initial detection.")
    parser.add_argument("--conf_thresh", type=float, default=0.25,
                        help="YOLO confidence threshold (lower to catch blurry frames).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-label images that already have a .json.")
    parser.add_argument("--yolo_only", action="store_true",
                        help="Use YOLO bounding boxes directly, skip SAM2 refinement.")
    parser.add_argument("--simplify_eps", type=float, default=2.0,
                        help="Polygon simplification epsilon (higher = fewer points).")
    args = parser.parse_args()

    # Find all PNGs
    png_paths = sorted(glob.glob(os.path.join(args.images_dir, "*.png")))
    print(f"Found {len(png_paths)} images in {args.images_dir}/")

    # Filter out already-labeled images
    if not args.overwrite:
        unlabeled = []
        for p in png_paths:
            json_path = p.replace(".png", ".json")
            if not os.path.exists(json_path):
                unlabeled.append(p)
        print(f"  Already labeled: {len(png_paths) - len(unlabeled)}")
        print(f"  Need labeling: {len(unlabeled)}")
        png_paths = unlabeled

    if not png_paths:
        print("Nothing to label.")
        return

    # Load YOLO
    print(f"\nLoading YOLO model: {args.yolo_model}")
    yolo = YOLO(args.yolo_model)

    # Load SAM2 (once — reuse for all images)
    if not args.yolo_only:
        print("Loading SAM2 model...")
        # Warm up SAM2 with a dummy image
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        osam.apis.generate(osam.types.GenerateRequest(
            model="sam2:latest",
            image=dummy,
            prompt=osam.types.Prompt(
                points=np.array([[320, 240]]),
                point_labels=np.array([1]),
            ),
        ))
        print("SAM2 ready.\n")

    labeled = 0
    no_detection = []

    for i, img_path in enumerate(png_paths):
        basename = os.path.basename(img_path)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  [{i+1}/{len(png_paths)}] {basename} — SKIP (can't read)")
            continue

        # Run YOLO
        results = yolo(img_bgr, verbose=False, conf=args.conf_thresh)
        det = results[0]
        boxes = det.boxes.xyxy.cpu().numpy() if len(det.boxes) > 0 else np.array([])
        scores = det.boxes.conf.cpu().numpy() if len(det.boxes) > 0 else np.array([])

        if len(boxes) == 0:
            no_detection.append(basename)
            print(f"  [{i+1}/{len(png_paths)}] {basename} — NO DETECTION")
            continue

        # Keep only the highest-confidence detection
        best_idx = np.argmax(scores)
        boxes = boxes[best_idx:best_idx+1]
        scores = scores[best_idx:best_idx+1]

        shapes = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box

            if args.yolo_only:
                # Use YOLO box directly
                points = box_to_polygon(box)
                shape_type = "polygon"
                desc = f"yolo conf={score:.3f}"
            else:
                # Use box center as SAM2 point prompt
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                resp = osam.apis.generate(osam.types.GenerateRequest(
                    model="sam2:latest",
                    image=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                    prompt=osam.types.Prompt(
                        points=np.array([[cx, cy]]),
                        point_labels=np.array([1]),
                    ),
                ))

                if resp.annotations:
                    mask = resp.annotations[0].mask
                    polygon = mask_to_polygon(mask, args.simplify_eps)
                    if polygon:
                        points = polygon
                        shape_type = "polygon"
                        desc = f"sam2+yolo conf={score:.3f}"
                    else:
                        # SAM2 mask too small, fall back to YOLO box
                        points = box_to_polygon(box)
                        shape_type = "polygon"
                        desc = f"yolo-fallback conf={score:.3f}"
                else:
                    points = box_to_polygon(box)
                    shape_type = "polygon"
                    desc = f"yolo-fallback conf={score:.3f}"

            shapes.append({
                "label": "Gripper",
                "points": points,
                "group_id": None,
                "description": desc,
                "shape_type": shape_type,
                "flags": {},
                "mask": None,
            })

        # Save labelme JSON
        annotation = make_labelme_json(img_path, img_bgr, shapes)
        json_path = img_path.replace(".png", ".json")
        with open(json_path, "w") as f:
            json.dump(annotation, f, indent=2)

        method = "YOLO" if args.yolo_only else "SAM2"
        print(f"  [{i+1}/{len(png_paths)}] {basename} — {len(shapes)} gripper(s) [{method}]")
        labeled += 1

    # Summary
    print(f"\n=== Auto-Label Summary ===")
    print(f"  Labeled: {labeled}")
    print(f"  No detection: {len(no_detection)}")
    if no_detection:
        nd_path = os.path.join(args.images_dir, "no_detection_list.txt")
        with open(nd_path, "w") as f:
            for item in no_detection:
                f.write(item + "\n")
        print(f"  No-detection list: {nd_path}")
    print(f"==========================")
    print(f"\nNext: review in labelme, then retrain:")
    print(f"  labelme {args.images_dir}/")
    print(f"  python train_yolo_gripper.py")


if __name__ == "__main__":
    main()
