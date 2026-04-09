#!/usr/bin/env python3
"""
Scan an object with a RealSense camera and reconstruct a watertight mesh.

Usage:
    1. Run the script and point the camera at the object.
    2. Press SPACE to start recording.
    3. Move the camera slowly around the object.
    4. Press SPACE again to stop recording (or 'q' to quit).
    5. The script registers frames, runs TSDF integration, and saves the mesh.
"""

import argparse
import time

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs


def get_realsense_pipeline():
    """Initialize RealSense pipeline with aligned depth at 424x240 @ 60fps."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 424, 240, rs.format.rgb8, 60)
    config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 60)
    profile = pipeline.start(config)

    # Extract intrinsics from the color stream
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()
    K = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ])

    # Align depth frames to the color frame
    align = rs.align(rs.stream.color)

    return pipeline, align, K, intrinsics.width, intrinsics.height


def capture_frames(pipeline, align, capture_every, max_frames):
    """
    Live capture loop. Returns list of (rgb, depth) numpy array pairs.

    Controls:
        SPACE - toggle recording on/off
        q     - quit immediately
    """
    frames_captured = []
    recording = False
    total_seen = 0
    frame_counter = 0

    print("[Capture] Press SPACE to start recording, 'q' to quit.")

    while True:
        rs_frames = pipeline.wait_for_frames()
        aligned = align.process(rs_frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color = np.asanyarray(color_frame.get_data())       # H x W x 3, RGB
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0  # meters

        # Build display image (BGR for OpenCV)
        display = color[:, :, ::-1].copy()
        status = "RECORDING" if recording else "PAUSED"
        color_text = (0, 0, 255) if recording else (200, 200, 200)
        cv2.putText(display, f"{status}  Captured: {len(frames_captured)}/{max_frames}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 2)
        cv2.imshow("Scan Object - Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[Capture] Quit requested.")
            cv2.destroyAllWindows()
            return None  # signal to abort
        elif key == ord(' '):
            recording = not recording
            if recording:
                print("[Capture] Recording started. Move camera slowly around the object.")
            else:
                print(f"[Capture] Recording stopped. {len(frames_captured)} frames captured.")
                break

        if recording:
            total_seen += 1
            if total_seen % capture_every == 0:
                frames_captured.append((color.copy(), depth.copy()))
                frame_counter += 1
                if frame_counter % 20 == 0:
                    print(f"[Capture]   ... {len(frames_captured)} frames captured so far")
                if len(frames_captured) >= max_frames:
                    print(f"[Capture] Reached max frames ({max_frames}). Stopping.")
                    break

    cv2.destroyAllWindows()
    return frames_captured


def register_frames(frames, K, width, height):
    """
    Estimate camera poses via frame-to-frame RGBD odometry.

    Returns:
        poses          - list of 4x4 numpy arrays (camera-to-world)
        valid_indices  - list of frame indices that were successfully registered
    """
    print(f"\n[Registration] Processing {len(frames)} frames...")

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height,
                                                  K[0, 0], K[1, 1],
                                                  K[0, 2], K[1, 2])

    option = o3d.pipelines.odometry.OdometryOption()
    jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

    def make_rgbd(rgb, depth):
        color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,       # already in meters
            depth_trunc=1.5,       # ignore depth beyond 1.5m
            convert_rgb_to_intensity=True,
        )

    # First frame is the world origin
    current_pose = np.eye(4)
    poses = [current_pose.copy()]
    valid_indices = [0]

    prev_rgbd = make_rgbd(frames[0][0], frames[0][1])
    success_count = 1
    fail_count = 0

    for i in range(1, len(frames)):
        curr_rgbd = make_rgbd(frames[i][0], frames[i][1])

        success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            curr_rgbd, prev_rgbd,
            intrinsic,
            np.eye(4),   # initial guess: identity
            jacobian,
            option,
        )

        if success:
            # trans maps current -> previous; accumulate into world frame
            current_pose = current_pose @ np.linalg.inv(trans)
            poses.append(current_pose.copy())
            valid_indices.append(i)
            prev_rgbd = curr_rgbd
            success_count += 1
        else:
            fail_count += 1
            print(f"[Registration] WARNING: Odometry failed on frame {i}, skipping.")

        if (i + 1) % 25 == 0 or i == len(frames) - 1:
            print(f"[Registration]   Processed {i + 1}/{len(frames)} "
                  f"(success: {success_count}, failed: {fail_count})")

    print(f"[Registration] Done. {success_count}/{len(frames)} frames registered, "
          f"{fail_count} failed.")

    if success_count < 10:
        print("[Registration] WARNING: Very few frames registered. "
              "The mesh quality may be poor. Try capturing more slowly.")

    return poses, valid_indices


def integrate_tsdf(frames, poses, valid_indices, K, width, height, voxel_size):
    """Integrate registered RGBD frames into a TSDF volume and extract a mesh."""
    sdf_trunc = voxel_size * 5.0  # 5x voxel size is a reasonable truncation
    print(f"\n[TSDF] Integrating {len(valid_indices)} frames "
          f"(voxel={voxel_size:.4f}m, trunc={sdf_trunc:.4f}m)...")

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height,
                                                  K[0, 0], K[1, 1],
                                                  K[0, 2], K[1, 2])

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for idx_in_list, frame_idx in enumerate(valid_indices):
        rgb, depth = frames[frame_idx]
        color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,
            depth_trunc=1.5,
            convert_rgb_to_intensity=False,
        )

        extrinsic = np.linalg.inv(poses[idx_in_list])  # world-to-camera
        volume.integrate(rgbd, intrinsic, extrinsic)

        if (idx_in_list + 1) % 25 == 0 or idx_in_list == len(valid_indices) - 1:
            print(f"[TSDF]   Integrated {idx_in_list + 1}/{len(valid_indices)} frames")

    print("[TSDF] Extracting triangle mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    print(f"[TSDF] Raw mesh: {len(mesh.vertices)} vertices, "
          f"{len(mesh.triangles)} triangles")
    return mesh


def postprocess_mesh(mesh, max_triangles=200000):
    """Remove small components and optionally simplify."""
    print("\n[Post-processing] Removing disconnected components...")

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    if len(cluster_n_triangles) == 0:
        print("[Post-processing] WARNING: No triangle clusters found. Returning raw mesh.")
        return mesh

    largest_cluster = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()

    removed_components = len(cluster_n_triangles) - 1
    if removed_components > 0:
        print(f"[Post-processing] Removed {removed_components} small component(s).")

    print(f"[Post-processing] Cleaned mesh: {len(mesh.vertices)} vertices, "
          f"{len(mesh.triangles)} triangles")

    # Simplify if the mesh is very dense
    n_triangles = len(mesh.triangles)
    if n_triangles > max_triangles:
        print(f"[Post-processing] Simplifying from {n_triangles} to {max_triangles} triangles...")
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=max_triangles)
        mesh.compute_vertex_normals()
        print(f"[Post-processing] Simplified mesh: {len(mesh.vertices)} vertices, "
              f"{len(mesh.triangles)} triangles")

    return mesh


def save_mesh(mesh, output_path):
    """Save mesh as OBJ and PLY, print stats."""
    print(f"\n[Save] Writing mesh to {output_path}")
    o3d.io.write_triangle_mesh(output_path, mesh)

    # Also save PLY for visualization
    ply_path = output_path.rsplit('.', 1)[0] + '.ply'
    print(f"[Save] Writing PLY to {ply_path}")
    o3d.io.write_triangle_mesh(ply_path, mesh)

    # Print mesh statistics
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    extent = bounds_max - bounds_min

    print("\n===== Mesh Statistics =====")
    print(f"  Vertices:  {len(vertices)}")
    print(f"  Triangles: {len(triangles)}")
    print(f"  Bounds min: [{bounds_min[0]:.4f}, {bounds_min[1]:.4f}, {bounds_min[2]:.4f}]")
    print(f"  Bounds max: [{bounds_max[0]:.4f}, {bounds_max[1]:.4f}, {bounds_max[2]:.4f}]")
    print(f"  Extent:     [{extent[0]:.4f}, {extent[1]:.4f}, {extent[2]:.4f}] meters")
    print("===========================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Scan an object with RealSense and reconstruct a mesh via TSDF."
    )
    parser.add_argument('--output', type=str, default='scanned_mesh.obj',
                        help='Output mesh file path (default: scanned_mesh.obj)')
    parser.add_argument('--capture_every', type=int, default=5,
                        help='Capture every Nth frame to reduce redundancy (default: 5)')
    parser.add_argument('--voxel_size', type=float, default=0.002,
                        help='TSDF voxel size in meters (default: 0.002)')
    parser.add_argument('--max_frames', type=int, default=500,
                        help='Maximum number of frames to capture (default: 500)')
    args = parser.parse_args()

    # ---- Phase 0: RealSense init ----
    print("=" * 50)
    print("  Object Scanner - RealSense + TSDF Reconstruction")
    print("=" * 50)
    print("\n[Init] Starting RealSense pipeline (424x240 @ 60fps)...")

    pipeline, align, K, width, height = get_realsense_pipeline()
    print(f"[Init] Camera intrinsics:\n{K}")
    print(f"[Init] Resolution: {width}x{height}")

    try:
        # ---- Phase 1: Capture ----
        print("\n" + "-" * 50)
        print("  PHASE 1: CAPTURE")
        print("-" * 50)
        captured = capture_frames(pipeline, align, args.capture_every, args.max_frames)

        if captured is None or len(captured) == 0:
            print("[Capture] No frames captured. Exiting.")
            return

        print(f"[Capture] Total frames collected: {len(captured)}")

        if len(captured) < 20:
            print("[Capture] WARNING: Fewer than 20 frames captured. "
                  "Results may be poor. Consider re-scanning with slower motion.")

    finally:
        pipeline.stop()
        print("[Init] RealSense pipeline stopped.")

    # ---- Phase 2: Registration ----
    print("\n" + "-" * 50)
    print("  PHASE 2: FRAME-TO-FRAME REGISTRATION")
    print("-" * 50)

    t_start = time.time()
    poses, valid_indices = register_frames(captured, K, width, height)
    t_reg = time.time() - t_start
    print(f"[Registration] Took {t_reg:.1f}s")

    if len(valid_indices) < 5:
        print("[Registration] ERROR: Too few frames registered (<5). "
              "Cannot produce a usable mesh. Try scanning more slowly "
              "with more overlap between frames.")
        return

    # ---- Phase 3: TSDF Integration ----
    print("\n" + "-" * 50)
    print("  PHASE 3: TSDF INTEGRATION")
    print("-" * 50)

    t_start = time.time()
    mesh = integrate_tsdf(captured, poses, valid_indices, K, width, height, args.voxel_size)
    t_tsdf = time.time() - t_start
    print(f"[TSDF] Took {t_tsdf:.1f}s")

    if len(mesh.vertices) == 0:
        print("[TSDF] ERROR: Extracted mesh has no vertices. "
              "The object may have been too far or depth data was too noisy.")
        return

    # ---- Phase 4: Post-processing and Save ----
    print("\n" + "-" * 50)
    print("  PHASE 4: POST-PROCESSING")
    print("-" * 50)

    mesh = postprocess_mesh(mesh)

    if len(mesh.vertices) == 0:
        print("[Post-processing] ERROR: Mesh is empty after post-processing.")
        return

    save_mesh(mesh, args.output)
    print("Done.")


if __name__ == '__main__':
    main()
