import argparse
from utils.video_utils import *
from config.config import results_yolo_path, results_frcnn_path, results_ssd_path, results_tracking_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object Detection and Tracking')
    parser.add_argument('mode', choices=['detection', 'tracking'], help='Mode to run: detection or tracking')
    parser.add_argument('--model', choices=['yolo', 'faster_rcnn', 'ssd'], default='yolo', help='Model to use for detection (only applicable for detection mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'detection':
        results = video_inference_detection(model=args.model)
        if args.model == 'yolo':
            results.to_csv(results_yolo_path, index=False)
        elif args.model == 'faster_rcnn':
            results.to_csv(results_frcnn_path, index=False)
        elif args.model == 'ssd':
            results.to_csv(results_ssd_path, index=False)
    elif args.mode == 'tracking':
        results = video_inference_tracking()
        results.to_csv(results_tracking_path, index=False)
        cap.release()
        out.release()
        cv2.destroyAllWindows()
