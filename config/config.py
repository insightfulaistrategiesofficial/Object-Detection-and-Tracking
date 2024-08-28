import cv2
import torch

# Paths
deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7' # Path to the DeepSort model
video_path = 'data/video.mp4'
output_path = 'output.mp4' # Path where the video with the tracked objects will be saved
yolo_model_path = 'yolov8l.pt' # Type of the YOLO model

results_yolo_path = "data/results/results_yolo.csv"
results_frcnn_path = "data/results/results_faster_rcnn.csv"
results_ssd_path = "data/results/results_ssd.csv"
results_tracking_path = "data/results/tracking_results.csv"

# Variables
max_age = 70 # Maximum number of missed misses before a track is deleted

# Open the video file for processing
cap = cv2.VideoCapture(video_path)

detected_object_index = 0 # Index of the detected object in the class_names list

confidence = 0.8 # Confidence threshold

unique_track_ids = set() # Set of unique track IDs

frames = [] # List of all frames

# Retrieve and store the width of the video frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Retrieve and store the height of the video frames
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Retrieve and store the frames per second (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

device = 'cuda' if torch.cuda.is_available() else 'cpu' # check if GPU is available

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))