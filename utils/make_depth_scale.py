import argparse
import json
import os
import sys

import cv2
import numpy as np
from joblib import Parallel, delayed

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text, read_intrinsics_binary, read_intrinsics_text, qvec2rotmat


def load_colmap_model(base_dir):
    sparse_dir = os.path.join(base_dir, "sparse", "0")
    images_bin = os.path.join(sparse_dir, "images.bin")
    cams_bin = os.path.join(sparse_dir, "cameras.bin")
    if os.path.exists(images_bin) and os.path.exists(cams_bin):
        images = read_extrinsics_binary(images_bin)
        cameras = read_intrinsics_binary(cams_bin)
    else:
        images = read_extrinsics_text(os.path.join(sparse_dir, "images.txt"))
        cameras = read_intrinsics_text(os.path.join(sparse_dir, "cameras.txt"))
    return cameras, images


def load_points_ordered(base_dir):
    sparse_dir = os.path.join(base_dir, "sparse", "0")
    points_bin = os.path.join(sparse_dir, "points3D.bin")
    points_txt = os.path.join(sparse_dir, "points3D.txt")

    points = {}
    if os.path.exists(points_bin):
        with open(points_bin, "rb") as fid:
            import struct

            def read_next_bytes(num_bytes, fmt, endian="<"):
                data = fid.read(num_bytes)
                return struct.unpack(endian + fmt, data)

            num_points = read_next_bytes(8, "Q")[0]
            for _ in range(num_points):
                row = read_next_bytes(43, "QdddBBBd")
                point_id = row[0]
                xyz = np.array(row[1:4], dtype=np.float64)
                track_length = read_next_bytes(8, "Q")[0]
                _ = read_next_bytes(8 * track_length, "ii" * track_length)
                points[point_id] = xyz
    elif os.path.exists(points_txt):
        with open(points_txt, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                elems = line.split()
                point_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])), dtype=np.float64)
                points[point_id] = xyz
    else:
        raise FileNotFoundError("Cannot find points3D.bin or points3D.txt under sparse/0")

    if not points:
        raise RuntimeError("No COLMAP points found.")

    max_id = max(points.keys())
    ordered = np.zeros((max_id + 1, 3), dtype=np.float64)
    for pid, xyz in points.items():
        ordered[pid] = xyz
    return ordered


def get_scales(key, cameras, images, points3d_ordered, depths_dir):
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]

    pts_idx = image_meta.point3D_ids
    mask = (pts_idx >= 0) & (pts_idx < len(points3d_ordered))
    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    if len(pts_idx) > 0:
        pts = points3d_ordered[pts_idx]
    else:
        pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    R = qvec2rotmat(image_meta.qvec)
    pts = np.dot(pts, R.T) + image_meta.tvec
    invcolmapdepth = 1.0 / pts[..., 2]

    n_remove = len(image_meta.name.split('.')[-1]) + 1
    depth_file = os.path.join(depths_dir, f"{image_meta.name[:-n_remove]}.png")
    invmonodepthmap = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    if invmonodepthmap is None:
        return None

    if invmonodepthmap.ndim != 2:
        invmonodepthmap = invmonodepthmap[..., 0]

    invmonodepthmap = invmonodepthmap.astype(np.float32) / (2 ** 16)
    s = invmonodepthmap.shape[0] / cam_intrinsic.height

    maps = (valid_xys * s).astype(np.float32)
    valid = (
        (maps[..., 0] >= 0)
        * (maps[..., 1] >= 0)
        * (maps[..., 0] < cam_intrinsic.width * s)
        * (maps[..., 1] < cam_intrinsic.height * s)
        * (invcolmapdepth > 0)
    )

    if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        maps = maps[valid, :]
        invcolmapdepth = invcolmapdepth[valid]
        invmonodepth = cv2.remap(
            invmonodepthmap,
            maps[..., 0],
            maps[..., 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )[..., 0]

        t_colmap = np.median(invcolmapdepth)
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))

        t_mono = np.median(invmonodepth)
        s_mono = np.mean(np.abs(invmonodepth - t_mono))
        scale = s_colmap / s_mono
        offset = t_colmap - t_mono * scale
    else:
        scale = 0
        offset = 0

    return {"image_name": image_meta.name[:-n_remove], "scale": float(scale), "offset": float(offset)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True, help="Path to COLMAP dataset root")
    parser.add_argument("--depths_dir", required=True, help="Path to generated depth maps")
    args = parser.parse_args()

    cameras, images = load_colmap_model(args.base_dir)
    points3d_ordered = load_points_ordered(args.base_dir)

    depth_param_list = Parallel(n_jobs=-1, backend="threading")(
        delayed(get_scales)(key, cameras, images, points3d_ordered, args.depths_dir) for key in images
    )

    depth_params = {
        depth_param["image_name"]: {"scale": depth_param["scale"], "offset": depth_param["offset"]}
        for depth_param in depth_param_list
        if depth_param is not None
    }

    output_path = os.path.join(args.base_dir, "sparse", "0", "depth_params.json")
    with open(output_path, "w") as f:
        json.dump(depth_params, f, indent=2)

    print(f"Saved depth params to {output_path}")


if __name__ == "__main__":
    main()
