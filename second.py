import uuid
from collections import defaultdict
from typing import List, Tuple

import cv2
import numpy
import torch
from shapely.geometry import Polygon
from shapely.geometry.point import Point
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes


def to_numpy(poly: Polygon) -> List[numpy.ndarray]:
    return [numpy.array(poly.exterior.coords, dtype=numpy.int32)]


def draw_text(img: numpy.ndarray, text: str, point: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    cv2.putText(img, text, point, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=color, thickness=2,
                lineType=cv2.LINE_AA)


def draw_polygon(img: numpy.ndarray, polygon: Polygon, color: Tuple[int, int, int]) -> None:
    cv2.polylines(img, to_numpy(polygon), isClosed=True, color=color, thickness=10)


def generate_id() -> str:
    return str(uuid.uuid4())[:4]


print(torch.cuda.is_available())

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

left_screen: Polygon = Polygon(((0, 0), (0, 1045), (1830, 1045), (1830, 0)))

polygon_red_left: Polygon = Polygon(((626, 123), (770, 250), (630, 250), (552, 130)))
polygon_red_right: Polygon = Polygon(((2710, 200), (2832, 206), (2925, 68), (2840, 68)))

polygon_green_left: Polygon = Polygon(((638, 114), (745, 210), (789, 190), (682, 95)))
polygon_green_right: Polygon = Polygon(((2600, 190), (2700, 190), (2829, 58), (2755, 68)))

polygon_pairs = [
    {"left": polygon_green_left, "right": polygon_green_right, "color": GREEN},
    {"left": polygon_red_left, "right": polygon_red_right, "color": RED},
]

polygon_mask: Polygon = Polygon(((453, 0), (1220, 1045), (2570, 1045), (2943, 0), (2680, 0), (1830, 660), (646, 0)))

model: YOLO = YOLO('model_only_car.pt')
video = cv2.VideoCapture('video_obzor_new_cam_yos_2.avi')

h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(video.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))

mask = numpy.zeros((h, w), dtype=numpy.uint8)
cv2.fillPoly(mask, to_numpy(polygon_mask), 255)

tracked_ids = defaultdict()
total = set()

def process_frame(frame: numpy.ndarray):
    frame_masked = cv2.bitwise_and(frame, frame, mask=mask)

    results: Results = model.track(frame_masked, conf=0.55, persist=True, device='cuda')[0]

    for polygon in polygon_pairs:
        draw_polygon(frame, polygon["left"], polygon["color"])
        draw_polygon(frame, polygon["right"], polygon["color"])

    boxes: Boxes = results.boxes
    pair_detect = [{"left": None, "right": None} for _ in polygon_pairs]
    for box in boxes:
        if not box.is_track:
            continue
        track_id = int(box.id[0])

        x1, y1, x2, y2 = numpy.int32(box.xyxy[0])
        polygon_box = Polygon(((x1, y1), (x2, y1), (x2, y2), (x1, y2)))
        center = Point((x1 + x2) / 2, (y1+y2) / 2)
        if track_id not in tracked_ids and left_screen.intersects(center):
            code = generate_id()
            tracked_ids[track_id] = code

        for i, pair in enumerate(polygon_pairs):
            (left, right, color) = pair.values()

            if left.intersects(center):
                pair_detect[i]["left"] = track_id
            if track_id not in tracked_ids and right.intersects(center):
                pair_detect[i]["right"] = track_id

        draw_polygon(frame, polygon_box, BLUE)
        draw_text(frame, f"Id: {tracked_ids.get(track_id, "")}", (x1, y1), RED)

    for detect in pair_detect:
        left = detect["left"]
        right = detect["right"]
        if left is not None and right is not None:
            tracked_ids[right] = tracked_ids[left]
            total.add(left)


    count = len(total)
    draw_text(frame, f"Проехало машин: {count}", (100, 100), RED)


while video.isOpened():
    success, frame = video.read()
    if not success:
        break

    process_frame(frame)
    writer.write(frame)

    img_small = cv2.resize(frame, (1312, 384))
    cv2.imshow("yo", img_small)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()
writer.release()
