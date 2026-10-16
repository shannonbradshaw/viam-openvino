import io
import struct

import numpy as np
import pytest
from PIL import Image
from viam.components.camera import Camera
from viam.media.video import CameraMimeType, NamedImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResponseMetadata
from viam.proto.component.camera import IntrinsicParameters
from viam.services.mlmodel import Metadata, MLModel, TensorInfo
from viam.utils import dict_to_struct

from models.yolo_decode import LetterboxInfo, decode_seg, masks_to_image
from models.yolo_seg_vision import YoloSegConfig, YoloSegVision

# A 64x64 model with 2 classes and 2 mask prototypes on a 16x16 grid (stride 4). Prototype 0 is "on" over the
# left half of the image, prototype 1 over the right half. Two objects sit side by side at the same depth with
# overlapping boxes: x 8..40 (class 0, mask = left half) and x 24..56 (class 1, mask = right half).
H = W = 64


def seg_outputs():
    protos = np.full((1, 2, 16, 16), -10.0, dtype=np.float32)
    protos[0, 0, :, :8] = 10.0
    protos[0, 1, :, 8:] = 10.0
    out = np.zeros((1, 4 + 2 + 2, 2), dtype=np.float32)
    out[0, :, 0] = [24, 32, 32, 40, 0.9, 0.0, 1.0, 0.0]  # cx cy w h | cls0 cls1 | coeff0 coeff1
    out[0, :, 1] = [40, 32, 32, 40, 0.0, 0.8, 0.0, 1.0]
    return out, protos


class FakeSegModel(MLModel):
    def __init__(self, name, label_path):
        super().__init__(name)
        self.out, self.protos = seg_outputs()
        self.label_path = label_path

    async def infer(self, input_tensors, *, extra=None, timeout=None):
        return {"output0": self.out, "output1": self.protos}

    async def metadata(self, *, extra=None, timeout=None):
        o0 = TensorInfo(name="output0", data_type="float32", shape=list(self.out.shape))
        o0.extra.CopyFrom(dict_to_struct({"labels": self.label_path}))
        o1 = TensorInfo(name="output1", data_type="float32", shape=list(self.protos.shape))
        return Metadata(name="fake", type="fake", input_info=[TensorInfo(name="images", data_type="float32", shape=[1, 3, H, W])], output_info=[o0, o1])


def png(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 100, 50)).save(buf, format="PNG")
    return buf.getvalue()


def depth_bytes(depth):
    h, w = depth.shape
    return b"DEPTHMAP" + struct.pack(">QQ", w, h) + depth.astype(">u2").tobytes()


class FakeRGBDCamera(Camera):
    def __init__(self, name, depth_shape=(H, W), depth_first=True):
        super().__init__(name)
        self.depth = np.full(depth_shape, 1000, dtype=np.uint16)
        self.depth_first = depth_first

    async def get_images(self, *, filter_source_names=None, extra=None, timeout=None, **kwargs):
        color = NamedImage("color", png(W, H), CameraMimeType.PNG)
        depth = NamedImage("depth", depth_bytes(self.depth), CameraMimeType.VIAM_RAW_DEPTH)
        return ([depth, color] if self.depth_first else [color, depth]), ResponseMetadata()

    async def get_properties(self, *, timeout=None, **kwargs):
        return Camera.Properties(
            supports_pcd=False,
            intrinsic_parameters=IntrinsicParameters(width_px=W, height_px=H, focal_x_px=50, focal_y_px=50, center_x_px=32, center_y_px=32),
            distortion_parameters=None,
            mime_types=[],
            frame_rate=0,
        )

    async def get_point_cloud(self, *, extra=None, timeout=None, **kwargs):
        raise NotImplementedError


@pytest.fixture
def seg_labels(tmp_path):
    p = tmp_path / "labels.txt"
    p.write_text("cup\nbottle\n")
    return str(p)


def make_service(seg_labels, cam=None, **attrs):
    ml = FakeSegModel("m", seg_labels)
    deps = {MLModel.get_resource_name("m"): ml}
    if cam is not None:
        deps[Camera.get_resource_name(cam.name)] = cam
    cfg = ComponentConfig(name="seg", api="rdk:service:vision", model="erh:openvino:yolo-seg",
                          attributes=dict_to_struct({"mlmodel_name": "m", "camera_name": "rs", **attrs}))
    return YoloSegVision.new(cfg, deps)


def test_decode_seg_and_masks():
    out, protos = seg_outputs()
    boxes, scores, cls, coeffs = decode_seg(out, 2, 0.5)
    assert cls.tolist() == [0, 1] and np.allclose(scores, [0.9, 0.8])
    np.testing.assert_allclose(boxes[0], [8, 12, 40, 52])
    with pytest.raises(ValueError, match="no room"):
        decode_seg(np.zeros((1, 6, 3), dtype=np.float32), 2, 0.5)
    lb = LetterboxInfo(orig_w=W, orig_h=H, scale=1.0, pad_x=0, pad_y=0, new_w=W, new_h=H, out_w=W, out_h=H)
    masks = masks_to_image(coeffs, protos, boxes, lb)
    assert masks.shape == (2, H, W)
    cols0, cols1 = np.nonzero(masks[0].any(axis=0))[0], np.nonzero(masks[1].any(axis=0))[0]
    assert cols0.min() == 8 and cols0.max() < 32  # left half, clipped to its box
    assert cols1.min() >= 32 and cols1.max() == 55
    assert not (masks[0] & masks[1]).any()


def test_masks_crop_letterbox_padding():
    # Wide 128x64 image letterboxed into 64x64: scale 0.5, pad_y 16. A prototype "on" everywhere must map to
    # the whole original image, not include the padding rows.
    protos = np.full((1, 1, 16, 16), 10.0, dtype=np.float32)
    lb = LetterboxInfo(orig_w=128, orig_h=64, scale=0.5, pad_x=0, pad_y=16, new_w=64, new_h=32, out_w=64, out_h=64)
    masks = masks_to_image(np.ones((1, 1), dtype=np.float32), protos, np.array([[0, 0, 128, 64]], dtype=np.float32), lb)
    assert masks.shape == (1, 64, 128) and masks.all()


def test_config_validation():
    assert YoloSegVision.validate_config(ComponentConfig(attributes=dict_to_struct({"mlmodel_name": "m", "camera_name": "rs"}))) == (["m"], ["rs"])
    cfg = YoloSegConfig({"mlmodel_name": "m"})
    assert (cfg.mask_threshold, cfg.depth_threshold_mm, cfg.min_points, cfg.mask_erode_px) == (0.5, 0.0, 50, 2)
    for bad in ({"yolo_version": "v5"}, {"min_points": -1}, {"mask_erode_px": 1.5}, {"mask_threshold": 2}):
        with pytest.raises(ValueError):
            YoloSegConfig({"mlmodel_name": "m", **bad})


async def test_touching_objects_get_separate_point_clouds(seg_labels):
    cam = FakeRGBDCamera("rs")
    v = make_service(seg_labels, cam, mask_erode_px=0, min_points=10)
    dets = await v.get_detections_from_camera("rs")
    assert [d.class_name for d in dets] == ["cup", "bottle"]
    objs = await v.get_object_point_clouds("rs")
    assert [o.geometries.geometries[0].label for o in objs] == ["cup", "bottle"]
    xs = []
    for o in objs:
        assert o.geometries.reference_frame == "rs"
        body = o.point_cloud.split(b"DATA binary\n")[1]
        rec = np.frombuffer(body, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
        assert np.allclose(rec["z"], 1.0)  # 1000 mm
        xs.append(rec["x"] * 1000)
    # Boxes overlap over x 24..40 px, but the masks split at x=32 px -> X = (32-32)*1000/50 = 0 mm.
    assert xs[0].max() < 0 <= xs[1].min()
    cup = objs[0].geometries.geometries[0]
    assert cup.box.dims_mm.x == pytest.approx((31 - 8) * 1000 / 50)
    assert cup.center.z == pytest.approx(1000)
    props = await v.get_properties()
    assert props.object_point_clouds_supported and props.default_camera == "rs"


async def test_capture_all_with_objects(seg_labels):
    v = make_service(seg_labels, FakeRGBDCamera("rs", depth_first=False), min_points=10)
    res = await v.capture_all_from_camera("rs", return_image=True, return_detections=True, return_object_point_clouds=True)
    assert res.image is not None and res.image.mime_type == CameraMimeType.PNG
    assert len(res.detections) == 2 and len(res.objects) == 2


async def test_unaligned_depth_is_rejected(seg_labels):
    v = make_service(seg_labels, FakeRGBDCamera("rs", depth_shape=(48, 64)))
    with pytest.raises(ValueError, match="align_color_depth"):
        await v.get_object_point_clouds("rs")


async def test_min_points_drops_small_objects(seg_labels):
    v = make_service(seg_labels, FakeRGBDCamera("rs"), min_points=100000)
    assert await v.get_object_point_clouds("rs") == []
