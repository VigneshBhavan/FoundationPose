#!/usr/bin/env python3
"""Fast visual review of auto-labeled images.

Shows each flagged image with its bounding box. Controls:
    ENTER/SPACE  - Accept label (keep as-is)
    d            - Delete image + label (bad image, remove from dataset)
    e            - Open in Labelme for editing
    q            - Quit (remaining images stay flagged)

Images with no detection are shown without a box — press 'e' to label in Labelme.
"""

import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np


def load_and_draw(img_dir, png_name):
    """Load image and draw bounding boxes from its JSON if it exists."""
    img_path = os.path.join(img_dir, png_name)
    json_path = os.path.join(img_dir, png_name.replace(".png", ".json"))

    img = cv2.imread(img_path)
    if img is None:
        return None, False

    has_label = os.path.exists(json_path)
    if has_label:
        with open(json_path) as f:
            ann = json.load(f)
        for shape in ann.get("shapes", []):
            pts = shape["points"]
            desc = shape.get("description", "")
            if len(pts) >= 2:
                x1, y1 = int(pts[0][0]), int(pts[0][1])
                x2, y2 = int(pts[2][0]), int(pts[2][1])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(img, desc, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

    return img, has_label


def main():
    parser = argparse.ArgumentParser(description="Review auto-labeled images.")
    parser.add_argument("--img_dir", default="gripper_images")
    parser.add_argument("--review_list", default="gripper_images/review_list.txt")
    args = parser.parse_args()

    if not os.path.exists(args.review_list):
        print("No review list found. Nothing to review.")
        return

    with open(args.review_list) as f:
        items = [line.strip() for line in f if line.strip()]

    # Separate filename from notes like "(no detection)"
    review_files = []
    for item in items:
        png = item.split(" ")[0]
        no_det = "(no detection)" in item
        review_files.append((png, no_det))

    print(f"\n{len(review_files)} images to review")
    print("  ENTER/SPACE = accept | d = delete | e = open in Labelme | q = quit\n")

    remaining = []
    accepted = 0
    deleted = 0
    edited = 0

    for i, (png, no_det) in enumerate(review_files):
        img, has_label = load_and_draw(args.img_dir, png)
        if img is None:
            print(f"  Skipping {png} (file not found)")
            continue

        tag = "[NO LABEL]" if no_det else "[LOW CONF]"
        status = f"[{i+1}/{len(review_files)}] {png} {tag}  |  ENTER=accept  d=delete  e=labelme  q=quit"
        cv2.putText(img, status, (5, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imshow("Review", img)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (13, 32):  # ENTER or SPACE
                accepted += 1
                print(f"  {png}: accepted")
                break
            elif key == ord("d"):
                # Delete image and label
                img_path = os.path.join(args.img_dir, png)
                json_path = os.path.join(args.img_dir, png.replace(".png", ".json"))
                if os.path.exists(img_path):
                    os.remove(img_path)
                if os.path.exists(json_path):
                    os.remove(json_path)
                deleted += 1
                print(f"  {png}: deleted")
                break
            elif key == ord("e"):
                # Open in Labelme
                img_path = os.path.join(args.img_dir, png)
                print(f"  Opening {png} in Labelme...")
                try:
                    subprocess.run(["labelme", img_path], check=False)
                except FileNotFoundError:
                    print("  ERROR: labelme not found. Install with: pip install labelme")
                edited += 1
                print(f"  {png}: edited in Labelme")
                break
            elif key == ord("q"):
                # Save remaining as still needing review
                remaining = [
                    f"{p} (no detection)" if nd else p
                    for p, nd in review_files[i:]
                ]
                print("Quitting review early.")
                break

        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    # Update review list with remaining items
    with open(args.review_list, "w") as f:
        for item in remaining:
            f.write(item + "\n")

    print(f"\n=== Review Summary ===")
    print(f"  Accepted: {accepted}")
    print(f"  Deleted:  {deleted}")
    print(f"  Edited:   {edited}")
    print(f"  Remaining: {len(remaining)}")
    print(f"======================")


if __name__ == "__main__":
    main()
