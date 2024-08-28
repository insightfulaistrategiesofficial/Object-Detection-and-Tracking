from ultralytics import YOLO
from config.config import yolo_model_path

# load yolov8 large model
yolo = YOLO(yolo_model_path)