"""
Convert labelme polygon annotations to YOLO segmentation format,
split into train/val, create dataset.yaml, and train YOLOv8-seg.
"""

import argparse
import glob
import json
import os
import random
import shutil

import yaml
from ultralytics import YOLO

CLASS_MAP = {"Gripper": 0}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert labelme polygon annotations to YOLO segmentation format and train YOLOv8-seg."
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="gripper_images_good",
        help="Directory containing labelme .json and .png files with polygon annotations.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="gripper_seg_dataset",
        help="Output directory for YOLO dataset and training runs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s-seg.pt",
        help="Pretrained YOLO segmentation model to fine-tune.",
    )
    return parser.parse_args()


def labelme_to_yolo_seg(json_path):
    """Read a labelme JSON file and return YOLO segmentation format strings.

    Each string is: ``class_id x1 y1 x2 y2 ... xn yn`` with values normalized to [0, 1].
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    lines = []
    for shape in data["shapes"]:
        label = shape["label"]
        if label not in CLASS_MAP:
            continue

        class_id = CLASS_MAP[label]
        points = shape["points"]

        if len(points) < 3:
            continue

        # Normalize and clamp polygon points
        norm_points = []
        for x, y in points:
            nx = max(0.0, min(1.0, x / img_w))
            ny = max(0.0, min(1.0, y / img_h))
            norm_points.append(f"{nx:.6f} {ny:.6f}")

        lines.append(f"{class_id} " + " ".join(norm_points))

    return lines


def convert_annotations(images_dir, output_dir):
    """Convert all labelme JSONs to YOLO segmentation .txt labels and split into train/val."""
    json_paths = sorted(glob.glob(os.path.join(images_dir, "*.json")))
    samples = []
    for jp in json_paths:
        stem = os.path.splitext(os.path.basename(jp))[0]
        img_path = os.path.join(images_dir, stem + ".png")
        if not os.path.isfile(img_path):
            print(f"Warning: no matching image for {jp}, skipping.")
            continue
        samples.append((img_path, jp))

    if not samples:
        raise FileNotFoundError(
            f"No valid image/json pairs found in {images_dir}"
        )

    print(f"Found {len(samples)} annotated images.")

    # Shuffle and split 80/20
    random.seed(42)
    random.shuffle(samples)
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    abs_output = os.path.abspath(output_dir)
    for split in ("train", "val"):
        os.makedirs(os.path.join(abs_output, "images", split), exist_ok=True)
        os.makedirs(os.path.join(abs_output, "labels", split), exist_ok=True)

    for split_name, split_samples in [("train", train_samples), ("val", val_samples)]:
        for img_path, json_path in split_samples:
            stem = os.path.splitext(os.path.basename(img_path))[0]

            dst_img = os.path.join(abs_output, "images", split_name, stem + ".png")
            shutil.copy2(img_path, dst_img)

            yolo_lines = labelme_to_yolo_seg(json_path)
            dst_label = os.path.join(abs_output, "labels", split_name, stem + ".txt")
            with open(dst_label, "w") as f:
                f.write("\n".join(yolo_lines))
                if yolo_lines:
                    f.write("\n")

    dataset_yaml = {
        "path": abs_output,
        "train": "images/train",
        "val": "images/val",
        "names": {0: "Gripper"},
    }
    yaml_path = os.path.join(abs_output, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"Dataset YAML written to {yaml_path}")
    return yaml_path


def train(yaml_path, model_name, epochs, output_dir):
    """Load a pretrained YOLO-seg model and train on the prepared dataset."""
    abs_output = os.path.abspath(output_dir)
    project_dir = os.path.join(abs_output, "runs")

    model = YOLO(model_name)
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=16,
        patience=50,
        project=project_dir,
        name="gripper_seg",
    )

    best_weights = os.path.join(
        project_dir, "gripper_seg", "weights", "best.pt"
    )
    if os.path.isfile(best_weights):
        print(f"\nTraining complete. Best weights saved to:\n  {best_weights}")
    else:
        found = glob.glob(os.path.join(project_dir, "gripper_seg*", "weights", "best.pt"))
        if found:
            best_weights = found[0]
            print(f"\nTraining complete. Best weights saved to:\n  {best_weights}")
        else:
            print("\nTraining complete. Could not locate best.pt automatically.")
            print(f"Check the runs directory: {project_dir}")

    return best_weights


def main():
    args = parse_args()
    yaml_path = convert_annotations(args.images_dir, args.output_dir)
    train(yaml_path, args.model, args.epochs, args.output_dir)


if __name__ == "__main__":
    main()
