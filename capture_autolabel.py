#!/usr/bin/env python3
"""Capture images from RealSense and auto-label with existing YOLO model.

Modes:
    SPACE  - Save single frame
    b      - Toggle burst mode (auto-save every N frames)
    r      - Toggle review mode (only save low-confidence detections)
    q      - Quit

Low-confidence detections (below --review_thresh) are flagged in a
review list so you can fix them in Labelme afterwards.
"""

import argparse
import base64
import json
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


def make_labelme_json(image_path, img_bgr, boxes, scores, review_thresh):
    """Create a labelme-compatible annotation dict from YOLO detections."""
    h, w = img_bgr.shape[:2]

    # Encode image as base64 PNG for labelme
    _, buf = cv2.imencode(".png", img_bgr)
    image_data = base64.b64encode(buf).decode("utf-8")

    shapes = []
    needs_review = False

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        # Clamp to image bounds
        x1 = max(0, min(w, float(x1)))
        y1 = max(0, min(h, float(y1)))
        x2 = max(0, min(w, float(x2)))
        y2 = max(0, min(h, float(y2)))

        if score < review_thresh:
            needs_review = True

        shapes.append({
            "label": "Gripper",
            "points": [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ],
            "group_id": None,
            "description": f"auto conf={score:.3f}",
            "shape_type": "polygon",
            "flags": {},
            "mask": None,
        })

    annotation = {
        "version": "5.11.4",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": image_data,
        "imageHeight": h,
        "imageWidth": w,
    }

    return annotation, needs_review


def main():
    parser = argparse.ArgumentParser(
        description="Capture + auto-label images for YOLO training."
    )
    parser.add_argument("--output_dir", type=str, default="gripper_images",
                        help="Directory to save images + labels.")
    parser.add_argument("--yolo_model", type=str,
                        default="gripper_dataset/runs/gripper_detect11/weights/best.pt",
                        help="Path to existing YOLO model for auto-labeling.")
    parser.add_argument("--conf_thresh", type=float, default=0.3,
                        help="Min confidence to include a detection.")
    parser.add_argument("--review_thresh", type=float, default=0.7,
                        help="Detections below this are flagged for review.")
    parser.add_argument("--burst_interval", type=int, default=10,
                        help="In burst mode, save every N frames.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load YOLO model
    print(f"Loading YOLO model: {args.yolo_model}")
    model = YOLO(args.yolo_model)

    # Find next available index (don't overwrite existing)
    existing = [f for f in os.listdir(args.output_dir) if f.endswith(".png")]
    save_idx = len(existing)
    print(f"Found {save_idx} existing images, starting at index {save_idx:04d}")

    # Init RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    pipeline.start(config)

    burst_mode = False
    frame_count = 0
    session_saved = 0
    review_list = []

    print(f"\nCapture ready at {args.width}x{args.height} @ {args.fps}fps")
    print(f"  SPACE = save one | b = toggle burst (every {args.burst_interval} frames)")
    print(f"  r = show review list | q = quit\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_rgb = np.asanyarray(color_frame.get_data())
            bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
            frame_count += 1

            # Run YOLO on current frame for display
            results = model(bgr, verbose=False, conf=args.conf_thresh)
            det = results[0]
            boxes_xyxy = det.boxes.xyxy.cpu().numpy() if len(det.boxes) > 0 else []
            scores = det.boxes.conf.cpu().numpy() if len(det.boxes) > 0 else []

            # Draw detections on display copy
            display = bgr.copy()
            for box, score in zip(boxes_xyxy, scores):
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0) if score >= args.review_thresh else (0, 165, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, f"{score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Status bar
            mode_str = "BURST" if burst_mode else "MANUAL"
            det_str = f"{len(boxes_xyxy)} det" if len(boxes_xyxy) > 0 else "no det"
            status = f"[{mode_str}] saved:{session_saved} total:{save_idx} {det_str} frame:{frame_count}"
            cv2.putText(display, status, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if len(review_list) > 0:
                cv2.putText(display, f"{len(review_list)} need review",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            cv2.imshow("Capture + AutoLabel", display)

            # Decide whether to save this frame
            should_save = False
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("b"):
                burst_mode = not burst_mode
                print(f"Burst mode: {'ON' if burst_mode else 'OFF'}")
            elif key == ord("r"):
                if review_list:
                    print(f"\n--- {len(review_list)} images need review ---")
                    for f in review_list:
                        print(f"  {f}")
                    print("---")
                else:
                    print("No images need review.")
            elif key == ord(" "):
                should_save = True

            if burst_mode and frame_count % args.burst_interval == 0:
                should_save = True

            # Save frame + auto-label
            if should_save and len(boxes_xyxy) > 0:
                filename = f"{save_idx:04d}.png"
                img_path = os.path.join(args.output_dir, filename)
                json_path = os.path.join(args.output_dir, f"{save_idx:04d}.json")

                # Save clean image (no overlay)
                cv2.imwrite(img_path, bgr)

                # Generate and save labelme annotation
                annotation, needs_review = make_labelme_json(
                    img_path, bgr, boxes_xyxy, scores, args.review_thresh
                )
                with open(json_path, "w") as f:
                    json.dump(annotation, f, indent=2)

                if needs_review:
                    review_list.append(filename)
                    print(f"  Saved {filename} [REVIEW - low conf]")
                else:
                    print(f"  Saved {filename} [OK]")

                save_idx += 1
                session_saved += 1

            elif should_save and len(boxes_xyxy) == 0:
                # No detection — save anyway but flag for manual labeling
                filename = f"{save_idx:04d}.png"
                img_path = os.path.join(args.output_dir, filename)
                cv2.imwrite(img_path, bgr)
                # No JSON — needs manual labeling
                review_list.append(filename + " (no detection)")
                print(f"  Saved {filename} [NO DETECTION - needs manual label]")
                save_idx += 1
                session_saved += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

        print(f"\n=== Session Summary ===")
        print(f"  Saved: {session_saved} images")
        print(f"  Total in {args.output_dir}/: {save_idx}")
        if review_list:
            review_path = os.path.join(args.output_dir, "review_list.txt")
            with open(review_path, "w") as f:
                for item in review_list:
                    f.write(item + "\n")
            print(f"  Need review: {len(review_list)} (saved to {review_path})")
        else:
            print(f"  Need review: 0")
        print(f"=======================")


if __name__ == "__main__":
    main()
