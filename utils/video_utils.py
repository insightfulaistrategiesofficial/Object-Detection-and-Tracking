import cv2
import torch.nn.functional as F
from time import time
from config.config import cap, unique_track_ids, frames, device, detected_object_index, confidence, out
from utils.transformations import transform
from models.yolo_model import yolo
from models.coco_names import class_names
from models.deep_sort_model import tracker
from models.ssd_model import ssd
from models.faster_rcnn_model import faster_rcnn
import pandas as pd
import torch

def video_inference_detection(model=yolo):
    """
    Function to perform object detection on a video using the specified model.

    Parameters:
    - model (str): The model to use for detection. Options are 'yolo', 'faster_rcnn', and 'ssd'.

    """
    # Initialize a dictionary to store results
    results = {
        'inference_time (millisec)': [],  # To store the time taken for inference
        'conf_scores': [],     # To store confidence scores of detections
        'class_names': [],     # To store class names for each detection
        'bboxes': []           # To store bounding boxes of detections
    }

    # Counter for frames
    i = 0

    while True:
        # Read a frame from the video
        success, img = cap.read()

        # Break the loop if no frame is captured (end of video)
        if not success:
            break

        transformed_img = transform(img)

        if model == 'yolo':
            # Resize the image dimensions to be multiples of 32 for YOLO
            new_height = (transformed_img.shape[1] // 32) * 32
            new_width = (transformed_img.shape[2] // 32) * 32
            
            # Use bilinear interpolation to resize the image
            transformed_img_resized = F.interpolate(
                transformed_img.unsqueeze(dim=0), 
                size=(new_height, new_width), 
                mode='bilinear', 
                align_corners=False
            )

        # Start timing the inference process
        start_time = time()
        
        if model == 'yolo':
        # Perform object detection on the resized image
            preds = yolo(transformed_img_resized)
        elif model == 'faster_rcnn':
            preds = faster_rcnn([transformed_img])
        elif model =='ssd':
            preds = ssd([transformed_img])
        
        # End timing the inference process
        end_time = time()

        # Initialize lists to store results for the current frame
        classes = []
        boxes = []
        conf_scores = []

        if model == 'yolo':
            # Process each prediction
            for pred in preds:
                # Extract bounding boxes from the prediction
                box = pred.boxes
                
                for b in box:
                    # Extract confidence score and convert to percentage
                    conf_score = round(b.conf[0].cpu().numpy() * 100)
                    
                    # Filter out predictions with low confidence
                    if conf_score > 80:
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype('int')
                        boxes.append([x1, y1, x2, y2])
                        
                        # Draw the bounding box on the original image
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                        
                        # Append confidence score and class name to lists
                        conf_scores.append(conf_score)
                        class_name = class_names[int(b.cls[0].cpu().numpy())]
                        classes.append(class_name)
                        
                        # Put text with the class name on the image
                        cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                        
        else:
            # Extract bounding boxes, labels, and scores from the prediction
            bboxes, labels, scores = preds[0]['boxes'], preds[0]['labels'], preds[0]['scores']

            # Filter out predictions with scores below the threshold (0.8 in this case)
            num = torch.argwhere(scores > 0.8).shape[0]
            
            for i in range(num):
                # Convert bounding box coordinates to integers
                x1, y1, x2, y2 = bboxes[i].detach().numpy().astype('int')
                boxes.append([x1, y1, x2, y2])
                
                # Draw the bounding box on the original image
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                
                # Get the class name from the label
                class_name = class_names[labels.detach().numpy()[i] - 1]
                classes.append(class_name)
                
                # Calculate and round the confidence score
                conf_score = round(scores.detach().numpy()[i] * 100)
                conf_scores.append(conf_score)
                
                # Put text with the class name on the image
                cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # End timing the inference process

        # Append results of the current frame to the results dictionary
        results['bboxes'].append(boxes)
        results['conf_scores'].append(conf_scores)
        results['class_names'].append(classes)
        
        if model == 'yolo':
            results['inference_time (millisec)'].append(end_time - start_time)
        else:
            results['inference_time (millisec)'].append((end_time - start_time) * 1000)

        # Display the frame with bounding boxes and labels
        cv2.imshow('Frame', img)

        # Break the loop if 'c' or 'C' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') or key == ord('C'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    return pd.DataFrame(results)


def video_inference_tracking():
    # Initialize variables for tracking FPS and elapsed time
    i = 0
    counter, fps, elapsed = 0, 0, 0
    start_time = time()  # Record the start time for FPS calculation

    # Dictionary to store tracking results, including confidence scores, class names, bounding boxes, and track IDs
    results_track = {
        'conf_scores': [],
        'class_names': [],
        'bboxes': [],
        'track_id': []
    }

    # Start reading the video frames
    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame from the video

        if ret:
            # Convert the frame from BGR to RGB format (YOLO typically expects RGB)
            og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = og_frame.copy()

            # Run the model on the current frame with specified settings (device, classes, confidence threshold)
            results = yolo(frame, device=0, classes=0, conf=0.8)

            # Process the results from YOLO for each detection
            for result in results:
                boxes = result.boxes  # Extract bounding boxes
                cls = boxes.cls.tolist()  # Extract class indices
                xyxy = boxes.xyxy  # Extract bounding box coordinates (x1, y1, x2, y2)
                conf = boxes.conf  # Extract confidence scores
                xywh = boxes.xywh  # Extract bounding box coordinates (x, y, width, height)
                
                # Map class indices to class names
                for class_index in cls:
                    class_name = class_names[int(class_index)]

            # Convert predictions to numpy arrays and detach from computation graph
            conf = conf.detach().cpu().numpy()
            xyxy = xyxy.detach().cpu().numpy()
            bboxes_xywh = xywh.cpu().numpy()

            # Store the tracking results in the results_track dictionary
            results_track['conf_scores'].extend(conf.tolist())
            results_track['class_names'].extend([class_names[int(index)] for index in cls])
            results_track['bboxes'].extend(xyxy.tolist())
            
            # Update the tracker with new bounding boxes and confidence scores
            tracks = tracker.update(bboxes_xywh, conf, og_frame)

            # Iterate over the tracks to draw bounding boxes and track IDs on the frame
            for track in tracker.tracker.tracks:
                track_id = track.track_id  # Retrieve the track ID
                hits = track.hits  # Number of hits for the track (how often it was detected)
                x1, y1, x2, y2 = track.to_tlbr()  # Convert bounding box format to (top-left, bottom-right)
                w = x2 - x1  # Calculate the width of the bounding box
                h = y2 - y1  # Calculate the height of the bounding box

                # Define colors for bounding boxes based on track ID
                red_color = (0, 0, 255)  # Red color
                blue_color = (255, 0, 0)  # Blue color
                green_color = (0, 255, 0)  # Green color

                # Cycle through colors based on track ID
                color_id = track_id % 3
                if color_id == 0:
                    color = red_color
                elif color_id == 1:
                    color = blue_color
                else:
                    color = green_color

                # Draw the bounding box with the appropriate color
                cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

                # Draw the class name and track ID above the bounding box
                text_color = (0, 0, 0)  # Black color for text
                cv2.putText(og_frame, f"{class_name}-{track_id}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
                
                # Add the track ID to the set of unique track IDs
                unique_track_ids.add(track_id)

            # Count the number of unique persons being tracked
            person_count = len(unique_track_ids)

            # Update the FPS calculation based on elapsed time
            current_time = time()
            elapsed = (current_time - start_time)
            counter += 1
            if elapsed > 1:
                fps = counter / elapsed
                counter = 0
                start_time = current_time

            # Display the person count on the frame
            cv2.putText(og_frame, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Store the processed frame for output
            frames.append(og_frame)

            # Write the frame to the output video file
            out.write(cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))

            # Display the frame with bounding boxes and tracking info
            cv2.imshow("Video", og_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
                break
    
    return results_track