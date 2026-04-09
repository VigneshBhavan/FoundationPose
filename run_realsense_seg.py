#!/usr/bin/env python3
"""Live FoundationPose 6DoF pose estimation with RealSense camera + YOLO segmentation mask."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from estimater import *
from datareader import *
import argparse
import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import json
import zmq
import time
from scipy.spatial.transform import Rotation


def get_realsense_pipeline():
    """Initialize RealSense pipeline with aligned depth."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 424, 240, rs.format.rgb8, 60)
    config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 60)
    profile = pipeline.start(config)

    # Get intrinsics
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()
    K = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ])

    # Align depth to color
    align = rs.align(rs.stream.color)

    return pipeline, align, K


def get_yolo_seg_mask(model, frame, conf=0.5):
    """Run YOLO segmentation and return pixel-level binary mask."""
    h, w = frame.shape[:2]
    results = model(frame, conf=conf, verbose=False)
    mask = np.zeros((h, w), dtype=bool)

    boxes = results[0].boxes
    masks = results[0].masks

    if boxes is not None and len(boxes) > 0 and masks is not None:
        best_idx = boxes.conf.argmax().item()
        best_conf = boxes.conf[best_idx].item()
        # Get the pixel-level segmentation mask and resize to frame dimensions
        seg_mask = masks.data[best_idx].cpu().numpy()
        seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = seg_mask > 0.5
        return mask, best_conf

    return mask, 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str,
                        default='/home/vbhavanantha/datacentertask/assets/h1_gripper.obj')
    parser.add_argument('--yolo_model', type=str,
                        default='/home/vbhavanantha/Desktop/PoseEstimation/FoundationPose/gripper_seg_dataset/runs/gripper_seg/weights/best.pt')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=1)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug_realsense_seg')
    parser.add_argument('--pose_log', type=str, default=None)
    parser.add_argument('--zmq_port', type=int, default=5555)
    parser.add_argument('--zmq_connect', type=str, default=None,
                        help='Connect to remote ZMQ XSUB (e.g. robot IP). If not set, binds locally.')
    parser.add_argument('--prune_keep_rate', type=float, default=1.0,
                        help='Score pruning: keep this fraction of hypotheses per refine iter (e.g. 0.5). 1.0=off (default).')
    parser.add_argument('--extrinsics', type=str, default=None,
                        help='Path to base_to_external_transform.json. If set, publishes poses in robot base frame.')
    args = parser.parse_args()
    if args.pose_log is None:
        args.pose_log = os.path.join(args.debug_dir, 'pose_log.json')

    set_logging_format()
    set_seed(0)

    # Load mesh (force ColorVisuals to avoid texture image requirement)
    mesh = trimesh.load(args.mesh_file)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    # Initialize FoundationPose
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=args.debug_dir,
        debug=args.debug,
        glctx=glctx,
    )
    logging.info("FoundationPose initialized")

    # Load YOLO segmentation model
    yolo = YOLO(args.yolo_model)
    logging.info(f"YOLO-seg loaded from {args.yolo_model}")

    # Start RealSense
    pipeline, align, K = get_realsense_pipeline()
    logging.info(f"RealSense started. K:\n{K}")

    os.makedirs(args.debug_dir, exist_ok=True)

    # ZMQ publisher — always bind locally, optionally also connect to robot
    zmq_ctx = zmq.Context()
    zmq_pub = zmq_ctx.socket(zmq.PUB)
    zmq_pub.bind(f"tcp://*:{args.zmq_port}")
    logging.info(f"ZMQ bound on tcp://*:{args.zmq_port}")
    if args.zmq_connect:
        zmq_pub.connect(f"tcp://{args.zmq_connect}:{args.zmq_port}")
        logging.info(f"ZMQ also connecting to tcp://{args.zmq_connect}:{args.zmq_port}")

    # Load extrinsics (camera-to-robot-base transform)
    T_base_cam = None
    if args.extrinsics:
        with open(args.extrinsics) as f:
            T_base_cam = np.array(json.load(f)["base_to_external_camera"])
        logging.info(f"Loaded extrinsics from {args.extrinsics} — publishing in robot base frame")
    else:
        logging.info("No extrinsics loaded — publishing in camera frame")

    pose = None
    center_pose = None
    pose_log = []
    frame_idx = 0
    fps_time = time.time()
    fps_count = 0
    fps = 0.0

    try:
        while True:
            t0 = time.time()

            # Get aligned frames
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())  # RGB
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0  # meters

            if pose is None:
                # Initial detection: use YOLO segmentation mask + FoundationPose register
                mask, conf = get_yolo_seg_mask(yolo, color[..., ::-1])  # YOLO expects BGR
                if conf >= 0.5:
                    logging.info(f"YOLO-seg detection conf={conf:.2f}, running initial pose estimation...")
                    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask,
                                        iteration=args.est_refine_iter,
                                        prune_keep_rate=args.prune_keep_rate)
                    center_pose = pose @ np.linalg.inv(to_origin)
                    log_entry = {
                        "frame": frame_idx,
                        "time": time.time(),
                        "pose_cam": pose.tolist(),
                        "type": "register",
                    }
                    if T_base_cam is not None:
                        log_entry["pose_base"] = (T_base_cam @ pose).tolist()
                    pose_log.append(log_entry)
                    logging.info(f"Initial pose:\n{pose}")
            else:
                pose = est.track_one(rgb=color, depth=depth, K=K,
                                     iteration=args.track_refine_iter)
                center_pose = pose @ np.linalg.inv(to_origin)
                log_entry = {
                    "frame": frame_idx,
                    "time": time.time(),
                    "pose_cam": pose.tolist(),
                    "type": "track",
                }
                if T_base_cam is not None:
                    log_entry["pose_base"] = (T_base_cam @ pose).tolist()
                pose_log.append(log_entry)

            # Debug: print raw camera-frame pose every 30 frames (~0.5s)
            if pose is not None and frame_idx % 30 == 0:
                t_cam = pose[:3, 3]
                print(f"\n>>> POSE [{frame_idx:5d}] cam_xyz=[{t_cam[0]:.4f} {t_cam[1]:.4f} {t_cam[2]:.4f}] <<<\n", file=sys.stderr, flush=True)

            # Publish pose over ZMQ
            if pose is not None:
                pub_pose = T_base_cam @ pose if T_base_cam is not None else pose
                t_vec = pub_pose[:3, 3].tolist()
                if frame_idx % 10 == 0:
                    c = pose[:3, 3]
                    print(f"[{frame_idx:5d}] cam=({c[0]:.4f},{c[1]:.4f},{c[2]:.4f}) base=({t_vec[0]:.4f},{t_vec[1]:.4f},{t_vec[2]:.4f})", flush=True)
                quat = Rotation.from_matrix(pub_pose[:3, :3]).as_quat().tolist()  # [x,y,z,w]
                zmq_pub.send_json({
                    "timestamp": time.time(),
                    "frame": frame_idx,
                    "frame_id": "pelvis" if T_base_cam is not None else "camera",
                    "position": {"x": t_vec[0], "y": t_vec[1], "z": t_vec[2]},
                    "orientation": {"x": quat[0], "y": quat[1], "z": quat[2], "w": quat[3]},
                })

            # Visualize — always show something
            vis = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            if center_pose is not None:
                vis_rgb = draw_posed_3d_box(K, img=color.copy(), ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(vis_rgb, ob_in_cam=center_pose, scale=0.05, K=K,
                                    thickness=2, transparency=0, is_input_rgb=True)
            else:
                cv2.putText(vis, "Waiting for detection...", (5, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # FPS counter
            fps_count += 1
            elapsed = time.time() - fps_time
            if elapsed >= 0.5:
                fps = fps_count / elapsed
                fps_time = time.time()
                fps_count = 0
            frame_ms = (time.time() - t0) * 1000
            cv2.putText(vis, f"FPS: {fps:.1f}  ({frame_ms:.0f}ms)", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('FoundationPose (seg)', vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                pose = None
                center_pose = None
                logging.info("Reset tracking. Will re-detect on next frame.")

            frame_idx += 1

            # Cap at 40 FPS
            elapsed = time.time() - t0
            min_dt = 1.0 / 40.0
            if elapsed < min_dt:
                time.sleep(min_dt - elapsed)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        zmq_pub.close()
        zmq_ctx.term()
        with open(args.pose_log, 'w') as f:
            json.dump(pose_log, f, indent=2)
        print(f"Saved {len(pose_log)} poses to {args.pose_log}")
