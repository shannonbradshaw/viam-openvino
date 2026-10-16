"""Pure-numpy helpers for turning a segmentation mask plus an aligned depth image into a 3D point cloud:
Viam depth decoding, mask erosion, pinhole deprojection, depth filtering and PCD encoding. No Viam or
OpenVINO imports so this is trivially unit-testable.

Units follow the RDK: points and geometries are in millimeters in the camera frame (+Z out of the lens,
+X right, +Y down); PCD payloads are in meters, as the RDK's PCD reader expects.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

DEPTH_MAGIC = b"DEPTHMAP"


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def decode_viam_depth(data: bytes) -> np.ndarray:
    """Decode an ``image/vnd.viam.dep`` payload into a uint16 [H, W] array of millimeters.

    Layout: 8-byte ``DEPTHMAP`` magic, big-endian uint64 width and height, then big-endian uint16 pixels."""
    if len(data) < 24 or data[:8] != DEPTH_MAGIC:
        raise ValueError("depth image is not in Viam's image/vnd.viam.dep format")
    width, height = struct.unpack(">QQ", data[8:24])
    expected = 24 + width * height * 2
    if len(data) < expected:
        raise ValueError(f"depth image is truncated: expected {expected} bytes for {width}x{height}, got {len(data)}")
    return np.frombuffer(data, dtype=">u2", count=width * height, offset=24).reshape(height, width).astype(np.uint16)


def erode(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Binary erosion with a (2*pixels+1) square kernel. Mask edges are where depth and color disagree most."""
    if pixels <= 0:
        return mask
    out = mask.copy()
    for _ in range(pixels):
        shrunk = out.copy()
        shrunk[1:, :] &= out[:-1, :]
        shrunk[:-1, :] &= out[1:, :]
        shrunk[:, 1:] &= out[:, :-1]
        shrunk[:, :-1] &= out[:, 1:]
        out = shrunk
    return out


def mask_to_points(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    rgb: np.ndarray,
    intr: Intrinsics,
    depth_threshold_mm: float = 0.0,
    max_depth_mm: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deproject the masked pixels with valid depth. Returns (points [N, 3] float32 mm, colors [N, 3] uint8).

    ``depth_threshold_mm`` > 0 keeps only points within that distance of the mask's median depth, which drops
    background that bleeds in at the mask edge. ``max_depth_mm`` > 0 drops anything farther than that."""
    if mask.shape != depth_mm.shape:
        raise ValueError(f"mask shape {mask.shape} does not match depth shape {depth_mm.shape}")
    valid = mask & (depth_mm > 0)
    if max_depth_mm > 0:
        valid &= depth_mm <= max_depth_mm
    vs, us = np.nonzero(valid)
    z = depth_mm[vs, us].astype(np.float32)
    if depth_threshold_mm > 0 and len(z):
        keep = np.abs(z - np.median(z)) <= depth_threshold_mm
        vs, us, z = vs[keep], us[keep], z[keep]
    x = (us.astype(np.float32) - intr.cx) * z / intr.fx
    y = (vs.astype(np.float32) - intr.cy) * z / intr.fy
    return np.stack([x, y, z], axis=1), rgb[vs, us].astype(np.uint8)


def bounding_box(points_mm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box of ``points_mm``: (center [3], dims [3]) in mm."""
    lo, hi = points_mm.min(axis=0), points_mm.max(axis=0)
    return (lo + hi) / 2.0, hi - lo


def to_pcd(points_mm: np.ndarray, colors: Optional[np.ndarray] = None) -> bytes:
    """Encode points as a binary PCD (meters, little-endian), with packed ``rgb`` when colors are given,
    matching the RDK's ``pointcloud.ToPCD`` output."""
    n = len(points_mm)
    xyz = (np.asarray(points_mm, dtype=np.float32) / 1000.0).astype("<f4")
    if colors is None:
        header = "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        body = xyz.tobytes()
    else:
        header = "FIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F I\nCOUNT 1 1 1 1\n"
        c = np.asarray(colors, dtype=np.uint32)
        packed = ((c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]).astype("<u4")
        rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
        rec["x"], rec["y"], rec["z"], rec["rgb"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], packed
        body = rec.tobytes()
    head = (
        "VERSION .7\n" + header + f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n"
    ).encode()
    return head + body
