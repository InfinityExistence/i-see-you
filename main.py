from typing import List, Tuple

import cv2
import numpy
import torch
from shapely.geometry import Polygon
from torch import Tensor
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes


def to_numpy(poly: Polygon) -> List[numpy.ndarray]:
    return [numpy.array(poly.exterior.coords, dtype=numpy.int32)]


def draw_text(img: numpy.ndarray, text: str, point: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    cv2.putText(img, text, point, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=color, thickness=4, lineType=cv2.LINE_AA)


def draw_polygon(img: numpy.ndarray, polygon: Polygon, color: Tuple[int, int, int]) -> None:
    cv2.polylines(img, to_numpy(polygon), isClosed=True, color=color, thickness=10)


print(torch.cuda.is_available())

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

polygon_red: Polygon = Polygon(((255, 1079), (680, 280), (980, 280), (980, 1079)))
polygon_green: Polygon = Polygon(((549, 700), (438, 950), (904, 950), (904, 700)))

model: YOLO = YOLO('yolo11n.pt')
video = cv2.VideoCapture('Counting_cars.mp4')

h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(video.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))

mask = numpy.zeros((h, w), dtype=numpy.uint8)
cv2.fillPoly(mask, to_numpy(polygon_red), 255)

tracked_ids = set()


def process_frame(frame: numpy.ndarray):
    frame_masked = cv2.bitwise_and(frame, frame, mask=mask)

    results : Results = model.track(frame_masked, conf=0.7, classes=[2], persist=True, device='cuda')[0]

    draw_polygon(frame, polygon_red, RED)
    draw_polygon(frame, polygon_green, GREEN)

    boxes : Boxes = results.boxes
    for box in boxes:
        if not box.is_track:
            continue
        track_id = int(box.id[0])
        conf : Tensor = box.conf[0]
        x1, y1, x2, y2 = numpy.int32(box.xyxy[0])
        polygon_box = Polygon(((x1, y1), (x2, y1), (x2, y2), (x1, y2)))

        if track_id not in tracked_ids and polygon_green.intersects(polygon_box):
            tracked_ids.add(track_id)

        draw_polygon(frame, polygon_box, BLUE)
        draw_text(frame, f"Id: {track_id}, Conf: {conf:.2f}", (x1, y1), RED)

    count = len(tracked_ids)
    draw_text(frame, f"Проехало машин: {count}", (0, 100), RED)


while video.isOpened():
    success, frame = video.read()
    if not success:
        break

    process_frame(frame)
    writer.write(frame)

    img_small = cv2.resize(frame, (640, 384))
    cv2.imshow("yo", img_small)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()
writer.release()
