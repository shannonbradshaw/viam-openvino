"""``erh:openvino:yolo-seg`` - a vision service for YOLO v8/v11 instance segmentation models.

Detections work like ``erh:openvino:yolo``. With an RGBD camera (color plus depth aligned to it, e.g. a
RealSense with ``align_color_depth: true``), ``GetObjectPointClouds`` projects each instance's mask, not its
bounding box, through the depth image, so touching objects and background inside the box are kept apart.
Point clouds and geometries are returned in the camera's frame, ready for the motion service's world state.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Mapping, Optional, Tuple

import numpy as np
from viam.components.camera import Camera
from viam.media.video import CameraMimeType, ViamImage
from viam.proto.common import (
    GeometriesInFrame,
    Geometry,
    PointCloudObject,
    Pose,
    RectangularPrism,
    Vector3,
)
from viam.proto.service.vision import Detection, GetPropertiesResponse
from viam.resource.types import Model
from viam.services.vision import CaptureAllResult
from viam.utils import ValueTypes

from .rgbd import Intrinsics, bounding_box, decode_viam_depth, erode, mask_to_points, to_pcd
from .yolo_decode import LetterboxInfo, decode_seg, masks_to_image, nms, unletterbox
from .yolo_vision import YoloConfig, YoloVision, _float_attr, _pick_color_image


class YoloSegConfig(YoloConfig):
    def __init__(self, attrs: Mapping[str, Any]) -> None:
        super().__init__(attrs)
        attrs = dict(attrs or {})
        if self.yolo_version == "v5":
            raise ValueError("yolo-seg supports YOLO v8/v11 segmentation models; yolo_version v5 is not supported")
        self.mask_threshold = _float_attr(attrs, "mask_threshold", 0.5, 0.0, 1.0)
        self.depth_threshold_mm = _float_attr(attrs, "depth_threshold_mm", 0.0, 0.0, 100000.0)
        self.max_depth_mm = _float_attr(attrs, "max_depth_mm", 0.0, 0.0, 100000.0)
        self.min_points = _int_attr(attrs, "min_points", 50)
        self.mask_erode_px = _int_attr(attrs, "mask_erode_px", 2)


def _int_attr(attrs: Mapping[str, Any], key: str, default: int) -> int:
    v = attrs.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v or int(v) < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return int(v)


class YoloSegVision(YoloVision):
    MODEL: ClassVar[Model] = "erh:openvino:yolo-seg"
    CONFIG_CLASS = YoloSegConfig

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._intrinsics: Dict[str, Intrinsics] = {}

    def reconfigure(self, config, dependencies) -> None:
        super().reconfigure(config, dependencies)
        self._intrinsics = {}

    # ------------------------------------------------------------ segmentation
    async def _segment(
        self, image: ViamImage
    ) -> Tuple[List[Tuple[np.ndarray, float, str]], np.ndarray, np.ndarray, LetterboxInfo]:
        """Run the model and return ([(box xyxy original px, score, label)], masks [K, H, W], RGB, letterbox)."""
        assert self.cfg is not None
        cfg: YoloSegConfig = self.cfg  # type: ignore[assignment]
        outputs, lb, info, rgb = await self._infer(image)
        raw, protos = _pick_seg_outputs(outputs)
        num_masks = protos.shape[-3]
        boxes, scores, class_ids, coeffs = decode_seg(raw, num_masks, cfg.confidence_threshold)
        keep = nms(boxes, scores, class_ids, cfg.iou_threshold, cfg.max_detections)
        boxes, scores, class_ids, coeffs = boxes[keep], scores[keep], class_ids[keep], coeffs[keep]
        names = [self._label(info, cid) for cid in class_ids]
        if cfg.class_filter is not None:
            sel = np.array([n in cfg.class_filter for n in names], dtype=bool)
            boxes, scores, coeffs = boxes[sel], scores[sel], coeffs[sel]
            names = [n for n, s in zip(names, sel) if s]
        boxes = unletterbox(boxes, lb)
        masks = masks_to_image(coeffs, protos, boxes, lb, cfg.mask_threshold)
        return list(zip(boxes, scores.tolist(), names)), masks, rgb, lb

    async def _detect(self, image: ViamImage) -> List[Detection]:
        instances, _, _, lb = await self._segment(image)
        return [self._to_detection(box, score, name, lb) for box, score, name in instances]

    # ------------------------------------------------------------ RGBD
    async def _get_intrinsics(self, name: str, cam: Camera) -> Intrinsics:
        if name not in self._intrinsics:
            props = await cam.get_properties()
            ip = props.intrinsic_parameters
            if not ip or ip.focal_x_px <= 0 or ip.focal_y_px <= 0:
                raise ValueError(f"camera '{name}' does not report intrinsic parameters; they are needed for point clouds")
            self._intrinsics[name] = Intrinsics(
                ip.focal_x_px, ip.focal_y_px, ip.center_x_px, ip.center_y_px, ip.width_px, ip.height_px
            )
        return self._intrinsics[name]

    async def _get_rgbd(self, camera_name: str) -> Tuple[str, ViamImage, np.ndarray, Intrinsics]:
        name, cam = self._camera(camera_name)
        images, _ = await cam.get_images()
        color = _pick_color_image(images)
        depth = next(
            (i for i in images if i.mime_type == CameraMimeType.VIAM_RAW_DEPTH or "depth" in (i.name or "").lower()),
            None,
        )
        if color is None or depth is None:
            raise ValueError(
                f"camera '{name}' must return both a color and a depth image for point clouds; "
                f"got {[(i.name, i.mime_type) for i in images]}"
            )
        intr = await self._get_intrinsics(name, cam)
        return name, ViamImage(color.data, color.mime_type), decode_viam_depth(depth.data), intr

    async def _object_point_clouds(self, camera_name: str, image: ViamImage, depth: np.ndarray, intr: Intrinsics) -> List[PointCloudObject]:
        assert self.cfg is not None
        cfg: YoloSegConfig = self.cfg  # type: ignore[assignment]
        instances, masks, rgb, _ = await self._segment(image)
        if depth.shape != rgb.shape[:2]:
            raise ValueError(
                f"depth image {depth.shape[1]}x{depth.shape[0]} does not match color image {rgb.shape[1]}x{rgb.shape[0]}; "
                "the depth must be aligned to the color frame (on a RealSense set align_color_depth: true)"
            )
        objects: List[PointCloudObject] = []
        for (_, _, label), mask in zip(instances, masks):
            pts, colors = mask_to_points(
                erode(mask, cfg.mask_erode_px), depth, rgb, intr, cfg.depth_threshold_mm, cfg.max_depth_mm
            )
            if len(pts) < max(cfg.min_points, 1):
                continue
            center, dims = bounding_box(pts)
            geom = Geometry(
                center=Pose(x=float(center[0]), y=float(center[1]), z=float(center[2]), o_x=0, o_y=0, o_z=1, theta=0),
                box=RectangularPrism(dims_mm=Vector3(x=float(dims[0]), y=float(dims[1]), z=float(dims[2]))),
                label=label,
            )
            objects.append(
                PointCloudObject(
                    point_cloud=to_pcd(pts, colors),
                    geometries=GeometriesInFrame(reference_frame=camera_name, geometries=[geom]),
                )
            )
        return objects

    # ------------------------------------------------------------ vision API
    async def get_object_point_clouds(self, camera_name: str, *, extra: Optional[Mapping[str, ValueTypes]] = None, timeout: Optional[float] = None) -> List[PointCloudObject]:
        name, image, depth, intr = await self._get_rgbd(camera_name)
        return await self._object_point_clouds(name, image, depth, intr)

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        if not return_object_point_clouds:
            return await super().capture_all_from_camera(
                camera_name, return_image, return_classifications, return_detections, False, extra=extra, timeout=timeout
            )
        result = CaptureAllResult()
        name, image, depth, intr = await self._get_rgbd(camera_name)
        if return_image:
            result.image = image
        if return_detections:
            result.detections = await self._detect(image)
        result.objects = await self._object_point_clouds(name, image, depth, intr)
        return result

    async def get_properties(self, *, extra: Optional[Mapping[str, ValueTypes]] = None, timeout: Optional[float] = None) -> GetPropertiesResponse:
        return GetPropertiesResponse(
            classifications_supported=False,
            detections_supported=True,
            object_point_clouds_supported=True,
            default_camera=(self.cfg.camera_name or "") if self.cfg else "",
        )


def _pick_seg_outputs(outputs: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Split segmentation outputs into the detection tensor (rank 2/3) and the mask prototypes (rank 4)."""
    arrays = [np.asarray(v) for v in outputs.values()]
    raw = next((a for a in arrays if a.ndim in (2, 3)), None)
    protos = next((a for a in arrays if a.ndim == 4), None)
    if raw is None or protos is None:
        raise ValueError(
            "a YOLO segmentation model needs a detection output and a [1, M, H, W] mask prototype output; got "
            f"{[(k, list(np.shape(v))) for k, v in outputs.items()]}. For plain detection models use erh:openvino:yolo"
        )
    return raw, protos
