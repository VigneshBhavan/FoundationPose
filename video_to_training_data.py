#!/usr/bin/env python3
"""Extract frames from a video (or record one from RealSense) and auto-label
with an existing YOLO model.

This captures motion blur that static frame capture misses, improving
detection robustness during fast gripper movement.

Modes:
    1. From existing video file:
        python video_to_training_data.py --video path/to/video.mp4

    2. Record from RealSense first, then extract:
        python video_to_training_data.py --record --duration 30

Controls (during recording):
    q  - Stop recording early

Controls (during review):
    Shows each extracted frame with YOLO detections overlaid.
    SPACE/ENTER - Accept and continue
    q           - Stop (remaining frames still saved)
"""

import argparse
import base64
import json
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO


def make_labelme_json(image_path, img_bgr, boxes, scores, review_thresh):
    """Create a labelme-compatible annotation dict from YOLO detections."""
    h, w = img_bgr.shape[:2]

    _, buf = cv2.imencode(".png", img_bgr)
    image_data = base64.b64encode(buf).decode("utf-8")

    shapes = []
    needs_review = False

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        x1 = max(0, min(w, float(x1)))
        y1 = max(0, min(h, float(y1)))
        x2 = max(0, min(w, float(x2)))
        y2 = max(0, min(h, float(y2)))

        if score < review_thresh:
            needs_review = True

        shapes.append({
            "label": "Gripper",
            "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
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


def record_video(output_path, duration, width, height, fps):
    """Record video from RealSense camera."""
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    pipeline.start(config)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Recording at {width}x{height} @ {fps}fps for up to {duration}s")
    print("Press 'q' to stop early\n")

    start = time.time()
    frame_count = 0

    try:
        while time.time() - start < duration:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_rgb = np.asanyarray(color_frame.get_data())
            bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            frame_count += 1

            elapsed = time.time() - start
            cv2.putText(bgr, f"REC {elapsed:.1f}s / {duration}s  frames:{frame_count}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("Recording", bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Recorded {frame_count} frames ({time.time() - start:.1f}s) -> {output_path}")
    return output_path


def extract_and_label(video_path, model, output_dir, extract_fps, conf_thresh,
                      review_thresh, skip_review):
    """Extract frames from video at target FPS, auto-label with YOLO."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, round(video_fps / extract_fps))

    print(f"Video: {video_fps:.1f} fps, {total_frames} frames")
    print(f"Extracting every {frame_interval} frames (~{extract_fps} fps output)")

    os.makedirs(output_dir, exist_ok=True)

    # Find next available index
    existing = [f for f in os.listdir(output_dir) if f.endswith(".png")]
    save_idx = len(existing)
    print(f"Found {save_idx} existing images, starting at index {save_idx:04d}")

    extracted = []
    frame_num = 0
    session_saved = 0
    review_list = []

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if frame_num % frame_interval == 0:
            # Run YOLO
            results = model(bgr, verbose=False, conf=conf_thresh)
            det = results[0]
            boxes = det.boxes.xyxy.cpu().numpy() if len(det.boxes) > 0 else []
            scores = det.boxes.conf.cpu().numpy() if len(det.boxes) > 0 else []

            filename = f"{save_idx:04d}.png"
            img_path = os.path.join(output_dir, filename)
            json_path = os.path.join(output_dir, f"{save_idx:04d}.json")

            # Save image
            cv2.imwrite(img_path, bgr)

            if len(boxes) > 0:
                # Auto-label
                annotation, needs_review = make_labelme_json(
                    img_path, bgr, boxes, scores, review_thresh
                )
                with open(json_path, "w") as f:
                    json.dump(annotation, f, indent=2)

                tag = "[REVIEW]" if needs_review else "[OK]"
                if needs_review:
                    review_list.append(filename)
            else:
                tag = "[NO DET]"
                review_list.append(filename + " (no detection)")

            print(f"  {filename} {tag}  ({len(boxes)} det, frame {frame_num}/{total_frames})")
            extracted.append((img_path, bgr, boxes, scores))
            save_idx += 1
            session_saved += 1

        frame_num += 1

    cap.release()

    # Quick visual review if not skipped
    if not skip_review and extracted:
        print(f"\n--- Review {len(extracted)} extracted frames ---")
        print("SPACE/ENTER = next | d = delete frame | q = done\n")

        for img_path, bgr, boxes, scores in extracted:
            display = bgr.copy()
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0) if score >= review_thresh else (0, 165, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, f"{score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            basename = os.path.basename(img_path)
            cv2.putText(display, basename, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Review Extracted Frames", display)

            key = cv2.waitKey(0) & 0xFF
            if key == ord("d"):
                # Delete this frame
                json_path = img_path.replace(".png", ".json")
                if os.path.exists(img_path):
                    os.remove(img_path)
                if os.path.exists(json_path):
                    os.remove(json_path)
                session_saved -= 1
                print(f"  Deleted {basename}")
            elif key == ord("q"):
                break

        cv2.destroyAllWindows()

    # Save review list
    if review_list:
        review_path = os.path.join(output_dir, "review_list.txt")
        # Append to existing review list
        mode = "a" if os.path.exists(review_path) else "w"
        with open(review_path, mode) as f:
            for item in review_list:
                f.write(item + "\n")

    print(f"\n=== Summary ===")
    print(f"  Video frames: {total_frames}")
    print(f"  Extracted: {session_saved}")
    print(f"  Need review: {len(review_list)}")
    print(f"  Output dir: {output_dir}")
    print(f"===============")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video and auto-label for YOLO training."
    )
    parser.add_argument("--video", type=str, default=None,
                        help="Path to existing video file. If omitted, records from RealSense.")
    parser.add_argument("--record", action="store_true",
                        help="Record a new video from RealSense before extracting.")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Recording duration in seconds (with --record).")
    parser.add_argument("--record_output", type=str, default="gripper_motion.mp4",
                        help="Output path for recorded video.")
    parser.add_argument("--extract_fps", type=float, default=40.0,
                        help="Target FPS for frame extraction (match policy deployment rate).")
    parser.add_argument("--output_dir", type=str, default="gripper_images",
                        help="Directory to save extracted images + labels.")
    parser.add_argument("--yolo_model", type=str,
                        default="gripper_dataset/runs/gripper_detect11/weights/best.pt",
                        help="Path to existing YOLO model for auto-labeling.")
    parser.add_argument("--conf_thresh", type=float, default=0.3,
                        help="Min confidence to include a detection.")
    parser.add_argument("--review_thresh", type=float, default=0.7,
                        help="Detections below this are flagged for review.")
    parser.add_argument("--skip_review", action="store_true",
                        help="Skip interactive review of extracted frames.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    # Step 1: Get video
    if args.record:
        video_path = record_video(args.record_output, args.duration,
                                  args.width, args.height, args.fps)
    elif args.video:
        video_path = args.video
    else:
        parser.error("Provide --video <path> or --record to capture from RealSense.")

    # Step 2: Load YOLO model
    print(f"\nLoading YOLO model: {args.yolo_model}")
    model = YOLO(args.yolo_model)

    # Step 3: Extract frames and auto-label
    extract_and_label(video_path, model, args.output_dir, args.extract_fps,
                      args.conf_thresh, args.review_thresh, args.skip_review)


if __name__ == "__main__":
    main()
