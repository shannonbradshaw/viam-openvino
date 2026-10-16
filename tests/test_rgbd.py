import struct

import numpy as np
import pytest

from models.rgbd import Intrinsics, bounding_box, decode_viam_depth, erode, mask_to_points, to_pcd


def encode_depth(depth: np.ndarray) -> bytes:
    h, w = depth.shape
    return b"DEPTHMAP" + struct.pack(">QQ", w, h) + depth.astype(">u2").tobytes()


def test_decode_viam_depth_roundtrip():
    depth = np.arange(12, dtype=np.uint16).reshape(3, 4) * 100
    out = decode_viam_depth(encode_depth(depth))
    assert out.dtype == np.uint16 and out.shape == (3, 4)
    np.testing.assert_array_equal(out, depth)


def test_decode_viam_depth_rejects_bad_payloads():
    with pytest.raises(ValueError, match="vnd.viam.dep"):
        decode_viam_depth(b"\xff\xd8 not a depth map at all....")
    with pytest.raises(ValueError, match="truncated"):
        decode_viam_depth(encode_depth(np.ones((4, 4), dtype=np.uint16))[:-2])


def test_erode_shrinks_square():
    m = np.zeros((10, 10), dtype=bool)
    m[2:8, 2:8] = True
    assert erode(m, 0) is m
    e = erode(m, 1)
    assert e.sum() == 16 and e[3:7, 3:7].all()
    assert erode(m, 3).sum() == 0


def test_mask_to_points_projection_and_filters():
    intr = Intrinsics(fx=100, fy=100, cx=2, cy=1, width=4, height=3)
    depth = np.array([[1000, 1000, 0, 1000], [1000, 2000, 1000, 1000], [1000, 1000, 5000, 1000]], dtype=np.uint16)
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    rgb[1, 1] = (10, 20, 30)
    mask = np.ones((3, 4), dtype=bool)
    pts, cols = mask_to_points(mask, depth, rgb, intr)
    assert len(pts) == 11  # the zero-depth pixel is dropped
    # pixel (u=1, v=1) at 2000 mm: X = (1-2)*2000/100 = -20, Y = 0
    i = int(np.argmax(pts[:, 2] == 2000))
    np.testing.assert_allclose(pts[i], [-20, 0, 2000])
    assert tuple(cols[i]) == (10, 20, 30)
    pts, _ = mask_to_points(mask, depth, rgb, intr, depth_threshold_mm=100)
    assert set(pts[:, 2].tolist()) == {1000.0}  # median 1000; 2000 and 5000 are outliers
    pts, _ = mask_to_points(mask, depth, rgb, intr, max_depth_mm=1500)
    assert pts[:, 2].max() == 1000
    with pytest.raises(ValueError, match="does not match"):
        mask_to_points(mask[:2], depth, rgb, intr)


def test_bounding_box():
    center, dims = bounding_box(np.array([[0, 0, 100], [10, -20, 140]], dtype=np.float32))
    np.testing.assert_allclose(center, [5, -10, 120])
    np.testing.assert_allclose(dims, [10, 20, 40])


def test_to_pcd_matches_rdk_binary_layout():
    pts = np.array([[1000, -500, 250], [0, 0, 1]], dtype=np.float32)
    cols = np.array([[255, 0, 1], [0, 128, 0]], dtype=np.uint8)
    data = to_pcd(pts, cols)
    head, body = data.split(b"DATA binary\n")
    assert b"FIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F I\n" in head
    assert b"WIDTH 2\nHEIGHT 1\n" in head and b"POINTS 2\n" in head
    rec = np.frombuffer(body, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
    np.testing.assert_allclose(rec["x"], [1.0, 0.0])  # meters
    np.testing.assert_allclose(rec["z"], [0.25, 0.001])
    assert rec["rgb"].tolist() == [0xFF0001, 0x008000]
    plain = to_pcd(pts)
    assert b"FIELDS x y z\n" in plain and len(plain.split(b"DATA binary\n")[1]) == 2 * 12
