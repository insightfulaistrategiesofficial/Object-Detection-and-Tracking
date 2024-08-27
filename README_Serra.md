# 🎥 Object Detection and Tracking Framework

Welcome to the **Object Detection and Tracking Framework**! This project allows you to perform object detection and tracking on video streams using state-of-the-art models like YOLO, Faster R-CNN, and SSD. Whether you're detecting objects or tracking them across frames, this framework has you covered! 🚀

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Detection Mode](#detection-mode)
  - [Tracking Mode](#tracking-mode)
- [Project Structure](#project-structure)
- [Author](#author)
- [Acknowledgements](#acknowledgements)

## 📝 Overview

This project is designed to provide an easy-to-use interface for running object detection and tracking on video files. You can choose between various models for detection and get detailed results with bounding boxes, class names, confidence scores, and more!

## ✨ Features

- **Multiple Detection Models:** Choose between YOLO, Faster R-CNN, and SSD for object detection. 🦾
- **Real-time Tracking:** Track objects across frames with unique IDs and see how they move over time. 👀
- **Customizable:** Easily swap out models or adjust parameters to fit your needs.

## 🛠 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/object-detection-tracking.git
   cd object-detection-tracking
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main script:**
   ```bash
   python main.py
   ```

## 🚀 Usage

You can run the script in two modes: detection or tracking. Use the `--model` argument to specify the detection model when in detection mode.

### 🔍 Detection Mode

```bash
python main.py detection --model [yolo | faster_rcnn | ssd]
```

- **YOLO:** Real-time detection with high accuracy and speed.
- **Faster R-CNN:** Highly accurate detection, best for scenarios where speed is less critical.
- **SSD:** A good balance between speed and accuracy.

### 🏃‍♂️ Tracking Mode

```bash
python main.py tracking
```

Track objects across video frames with unique IDs, and see how they move over time!

## 📁 Project Structure

```
📂 Object-Detection-and-Tracking
 ┣ 📂 config
 ┃ ┣ 📜 config.py            # Configuration settings for paths and parameters
 ┣ 📂 models
 ┃ ┣ 📜 yolo_model.py        # YOLO model implementation
 ┃ ┣ 📜 faster_rcnn_model.py # Faster R-CNN model implementation
 ┃ ┣ 📜 ssd_model.py         # SSD model implementation
 ┃ ┣ 📜 deep_sort_model.py   # Deep SORT tracker implementation
 ┃ ┣ 📜 coco_names.py        # COCO class names
 ┣ 📂 utils
 ┃ ┣ 📜 video_utils.py       # Video processing and inference functions
 ┃ ┣ 📜 transformations.py   # Image transformations
 ┣ 📜 main.py                # Main script for running detection and tracking
 ┣ 📜 requirements.txt       # Python dependencies
 ┗ 📜 README.md              # This file
```

## Author
**Serra**  
Email: [serurays@gmail.com](mailto:serurays@gmail.com)  
GitHub: [serra-github](https://github.com/Serurays)

## 🙌 Acknowledgements

A big thanks to the Insightful AI Strategies for giving me this project! 🤗

Happy coding and enjoy exploring the world of object detection and tracking! 🎉