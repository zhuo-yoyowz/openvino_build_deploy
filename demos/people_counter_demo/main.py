import argparse
import json
import logging as log
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Tuple, List, Dict

import cv2
import numpy as np
import supervision as sv
import torch
from openvino import runtime as ov
from supervision import ColorLookup
from ultralytics import YOLO
from ultralytics.utils import ops

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils
from deep_sort_realtime.deepsort_tracker import DeepSort

CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def convert(model_name: str, model_dir: Path) -> tuple[Path, Path]:
    model_path = model_dir / f"{model_name}.pt"
    # create a YOLO object detection model
    yolo_model = YOLO(model_path)

    ov_model_path = model_dir / f"{model_name}_openvino_model"
    ov_int8_model_path = model_dir / f"{model_name}_int8_openvino_model"
    # export the model to OpenVINO format (FP16 and INT8)
    if not ov_model_path.exists():
        ov_model_path = yolo_model.export(format="openvino", dynamic=False, half=True)
    if not ov_int8_model_path.exists():
        ov_int8_model_path = yolo_model.export(format="openvino", dynamic=False, half=True, int8=True, data="coco128.yaml")
    return Path(ov_model_path) / f"{model_name}.xml", Path(ov_int8_model_path) / f"{model_name}.xml"


def letterbox(img: np.ndarray, new_shape: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[1::-1]  # current shape [width, height]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape != new_unpad:  # resize
        img = cv2.resize(img, dsize=new_unpad, interpolation=cv2.INTER_AREA)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))  # add border
    return img, ratio, (int(dw), int(dh))


def preprocess(image: np.ndarray, input_size: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    # add padding to the image
    image, _, padding = letterbox(image, new_shape=input_size)
    # convert to float32
    image = image.astype(np.float32)
    # normalize to (0, 1)
    image /= 255.0
    # changes data layout from HWC to CHW
    image = image.transpose((2, 0, 1))
    # add one more dimension
    image = np.expand_dims(image, axis=0)
    return image, padding


def postprocess(pred_boxes: np.ndarray, pred_masks: np.ndarray, input_size: Tuple[int, int], orig_img: np.ndarray, padding: Tuple[int, int], category_id: int,
                min_conf_threshold: float = 0.25, nms_iou_threshold: float = 0.75, agnostic_nms: bool = False, max_detections: int = 100) -> sv.Detections:
    nms_kwargs = {"agnostic": agnostic_nms, "max_det": max_detections}
    # non-maximum suppression
    pred = ops.non_max_suppression(torch.from_numpy(pred_boxes), min_conf_threshold, nms_iou_threshold, nc=80, **nms_kwargs)[0]

    # no predictions in the image
    if not len(pred):
        return sv.Detections.empty()

    masks = pred_masks
    if pred_masks is not None:
        # upscale masks
        masks = np.array(ops.process_mask(torch.from_numpy(pred_masks[0]), pred[:, 6:], pred[:, :4], input_size, upsample=True))
        masks = np.array([cv2.resize(mask[padding[1]:-padding[1] - 1, padding[0]:-padding[0] - 1], orig_img.shape[:2][::-1], interpolation=cv2.INTER_AREA) for mask in masks])
        masks = masks.astype(np.bool_)
    # transform boxes to pixel coordinates
    pred[:, :4] = ops.scale_boxes(input_size, pred[:, :4], orig_img.shape).round()
    # numpy array from torch tensor
    pred = np.array(pred)
    # create detections in supervision format
    det = sv.Detections(xyxy=pred[:, :4], mask=masks, confidence=pred[:, 4], class_id=pred[:, 5])
    # filter out other predictions than selected category
    return det[det.class_id == category_id]

def get_model(model_path: Path, device: str = "AUTO") -> ov.CompiledModel:
    # initialize OpenVINO
    core = ov.Core()
    # read the model from file
    model = core.read_model(model_path)
    # compile the model for latency mode
    model = core.compile_model(model, device_name=device, config={"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": "cache"})

    return model


def load_zones(json_path: str) -> List[np.ndarray]:
    # load json file
    with open(json_path) as f:
        zones_dict = json.load(f)

    # return a list of zones defined by points
    return [np.array(zone["points"], np.int32) for zone in zones_dict.values()]


def get_annotators(json_path: str, resolution_wh: Tuple[int, int], colorful: bool = False) -> Tuple[List, List, List, List, List]:
    # list of points
    polygons = load_zones(json_path)

    # colors for zones
    colors = sv.ColorPalette.DEFAULT

    zones = []
    zone_annotators = []
    box_annotators = []
    masks_annotators = []
    label_annotators = []
    for index, polygon in enumerate(polygons, start=1):
        # a zone to count objects in
        zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=resolution_wh)
        zones.append(zone)
        # the annotator - visual part of the zone
        zone_annotators.append(sv.PolygonZoneAnnotator(zone=zone, color=colors.by_idx(index), thickness=0))
        # box annotator, showing boxes around objects
        box_annotator = sv.BoxAnnotator(color_lookup=ColorLookup.INDEX) if colorful else sv.BoxAnnotator(color=colors.by_idx(index))
        box_annotators.append(box_annotator)
        # mask annotator, showing transparent mask
        mask_annotator = sv.MaskAnnotator(color_lookup=ColorLookup.INDEX) if colorful else sv.MaskAnnotator(color=colors.by_idx(index))
        masks_annotators.append(mask_annotator)
        # label annotator, showing people ids
        label_annotator = sv.LabelAnnotator(text_scale=0.7, color_lookup=ColorLookup.INDEX) if colorful else sv.LabelAnnotator(text_scale=0.7, color=colors.by_idx(index))
        label_annotators.append(label_annotator)

    return zones, zone_annotators, box_annotators, masks_annotators, label_annotators


def track_objects(frame: np.array, detections: sv.Detections, tracker: DeepSort) -> List:
    # Convert detections to the format required by the tracker
    detection_list = []
    for det in zip(detections.xyxy, detections.confidence):
        bbox = det[0].tolist()
        confidence = det[1]
        detection_list.append((bbox, confidence))
 
    # Update the tracker with the new detections
    tracks = tracker.update_tracks(detection_list, frame=frame)
    # detections are sorted by confidence so tracks must be also sorted the same way
    return list(sorted(tracks, key=lambda x: x.det_conf if x.det_conf is not None else 0.0, reverse=True))


def draw_annotations(frame: np.array, detections: sv.Detections, tracker: DeepSort, queue_count: Dict, object_limit: int, category:str,
                     zones: List, zone_annotators: List, box_annotators: List, masks_annotators: List, label_annotators: List) -> None:

    for zone_annotator in zone_annotators:
        # visualize polygon for the zone
        frame = zone_annotator.annotate(scene=frame)

    if detections:
        # uniquely track the objects
        tracks = track_objects(frame, detections, tracker)

        # annotate the frame with the detected persons within each zone
        for zone_id, (zone, box_annotator, masks_annotator, label_annotator) in enumerate(
                zip(zones, box_annotators, masks_annotators, label_annotators), start=1):

            # get detections relevant only for the zone
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            # visualize boxes around objects in the zone
            frame = masks_annotator.annotate(scene=frame, detections=detections_filtered)
            frame = box_annotator.annotate(scene=frame, detections=detections_filtered)
            # count how many objects detected
            det_count = len(detections_filtered)
            # Add track ID annotations
            label_annotator.annotate(scene=frame, detections=detections_filtered,
                                     labels=[f"ID: {track.track_id if track.time_since_update == 0 else ' '}" for
                                             track, in_zone in zip(tracks, mask) if in_zone])
            # add the count to the list
            queue_count[zone_id].append(det_count)
            # calculate the mean number of customers in the queue
            mean_customer_count = np.mean(queue_count[zone_id], dtype=np.int32)

            # add alert text to the frame if necessary, flash every second
            if mean_customer_count > object_limit and time.time() % 2 > 1:
                utils.draw_text(frame, text=f"Intel employee required in zone {zone_id}!", point=(20, 20),
                                font_color=(0, 0, 255))

            # print an info about number of customers in the queue, ask for the more assistants if required
            log.info(
                f"Zone {zone_id}, avg {category} count: {mean_customer_count} {'Intel employee required!' if mean_customer_count > object_limit else ''}")


def run(video_path: str, model_paths: Tuple[Path, Path], model_name: str = "", category: str = "person", zones_config_file: str = "",
        object_limit: int = 3, flip: bool = True, tracker_frames: int = 1800, colorful: bool = False, last_frames: int = 50) -> None:
    # set up logging
    log.getLogger().setLevel(log.INFO)

    model_mapping = {
        "FP16": model_paths[0],
        "INT8": model_paths[1],
    }

    device_mapping = utils.available_devices()

    model_type = "INT8"
    device_type = "AUTO"

    core = ov.Core()
    core.set_property({"CACHE_DIR": "cache"})
    # initialize and load model
    model = get_model(model_mapping[model_type], device_type)
    # input shape of the model (w, h, d)
    input_shape = tuple(model.inputs[0].shape)[:0:-1]

    # initialize video player to deliver frames
    if isinstance(video_path, str) and video_path.isnumeric():
        video_path = int(video_path)
    player = utils.VideoPlayer(video_path, size=(1920, 1080), fps=60, flip=flip)

    # get zones, and zone and box annotators for zones
    zones, zone_annotators, box_annotators, masks_annotators, label_annotators = get_annotators(json_path=zones_config_file, resolution_wh=(player.width, player.height), colorful=colorful)
    category_id = CATEGORIES.index(category)

    # object counter
    queue_count = defaultdict(lambda: deque(maxlen=last_frames))
    # keep at most 100 last times
    processing_times = deque(maxlen=100)

    # Initialize the tracker with a higher max_age
    tracker = DeepSort(max_age=tracker_frames, n_init=3)

    title = "Press ESC to Exit"
    cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL)
    cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # start a video stream
    player.start()
    while True:
        # Grab the frame.
        frame = player.next()
        if frame is None:
            print("Source ended")
            break
        # Get the results.
        frame = np.array(frame)
        f_height, f_width = frame.shape[:2]

        # preprocessing
        input_image, padding = preprocess(image=frame, input_size=input_shape[:2])
        # prediction
        start_time = time.time()
        results = model(input_image)
        processing_times.append(time.time() - start_time)
        boxes = results[model.outputs[0]]
        masks = results[model.outputs[1]] if len(model.outputs) > 1 else None
        # postprocessing
        detections = postprocess(pred_boxes=boxes, pred_masks=masks, input_size=input_shape[:2], orig_img=frame, padding=padding, category_id=category_id)

        # draw results
        draw_annotations(frame, detections, tracker, queue_count, object_limit, category, zones, zone_annotators, box_annotators, masks_annotators, label_annotators)

        # Mean processing time [ms].
        processing_time = np.mean(processing_times) * 1000

        fps = 1000 / processing_time
        utils.draw_text(frame, text=f"Inference time: {processing_time:.0f}ms ({fps:.1f} FPS)", point=(f_width * 3 // 5, 10))
        utils.draw_text(frame, text=f"Currently running {model_name} ({model_type}) on {device_type}", point=(f_width * 3 // 5, 50))

        utils.draw_control_panel(frame, device_mapping)
        utils.draw_ov_watermark(frame)
        # show the output live
        cv2.imshow(title, frame)
        key = cv2.waitKey(1)
        # escape = 27 or 'q' to close the app
        if key == 27 or key == ord('q'):
            break

        model_changed = False
        if key == ord('f'):
            model_type = "FP16"
            model_changed = True
        if key == ord('i'):
            model_type = "INT8"
            model_changed = True
        for i, dev in enumerate(device_mapping.keys()):
            if key == ord('1') + i:
                device_type = dev
                model_changed = True

        if model_changed:
            del model
            model = get_model(model_mapping[model_type], device_type)
            processing_times.clear()

    # stop the stream
    player.stop()
    # clean-up windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', default="0", type=str, help="Path to a video file or the webcam number")
    parser.add_argument("--model_name", type=str, default="yolo11n", help="Model version to be converted",
                        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x", "yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg",
                                 "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x", "yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg"])
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")
    parser.add_argument('--category', type=str, default="person", choices=CATEGORIES, help="The category to detect (from COCO dataset)")
    parser.add_argument('--zones_config_file', type=str, default="zones.json", help="Path to the zone config file (json)")
    parser.add_argument('--object_limit', type=int, default=3, help="The maximum number of objects in the area")
    parser.add_argument("--flip", type=bool, default=True, help="Mirror input video")
    parser.add_argument('--colorful', action="store_true", help="If objects should be annotated with random colors")
    parser.add_argument('--tracker_frames', type=int, default=1800, help="Maximum number of missed frames for the tracker")

    args = parser.parse_args()
    model_paths = convert(args.model_name, Path(args.model_dir))
    run(args.stream, model_paths, args.model_name, args.category, args.zones_config_file, args.object_limit, args.tracker_frames, args.flip, args.colorful)
