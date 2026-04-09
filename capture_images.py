#!/usr/bin/env python3
"""Capture training images from a RealSense camera for YOLO training.

Controls:
    SPACE  - Save current frame as PNG
    q      - Quit
"""

import argparse
import os

import cv2
import numpy as np
import pyrealsense2 as rs


def main():
    parser = argparse.ArgumentParser(description="Capture images from RealSense for YOLO training.")
    parser.add_argument("--output_dir", type=str, default="captured_images",
                        help="Directory to save captured images.")
    parser.add_argument("--prefix", type=str, default="",
                        help="Prefix prepended to each filename (e.g. 'obj1_').")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Count existing images so we never overwrite previous captures.
    existing = [f for f in os.listdir(args.output_dir)
                if f.startswith(args.prefix) and f.endswith(".png")]
    save_idx = len(existing)

    # Initialize RealSense -- color only, matching run_realsense.py resolution.
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 424, 240, rs.format.rgb8, 60)
    pipeline.start(config)

    frame_count = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())  # RGB from RealSense
            bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

            # Overlay status text.
            cv2.putText(bgr, f"Frame: {frame_count}  Saved: {save_idx}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(bgr, "SPACE=save  q=quit",
                        (10, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            cv2.imshow("Capture", bgr)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                filename = f"{args.prefix}{save_idx:03d}.png"
                path = os.path.join(args.output_dir, filename)
                # Save the clean RGB frame (no overlay) as PNG.
                cv2.imwrite(path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
                print(f"Saved {path}")
                save_idx += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"Done. {save_idx} total images in {args.output_dir}/")


if __name__ == "__main__":
    main()
