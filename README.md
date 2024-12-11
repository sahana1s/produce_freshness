# Produce Freshness Detection App

This Streamlit app performs freshness detection on images using a custom-trained YOLO (You Only Look Once) object detection model. It allows users to upload images of products (e.g., fruits, vegetables) and automatically classifies them as "Fresh" or "Rotten" based on detected objects. The results are displayed on the app, and users can view processed product data on a separate page.

## Features

- **Freshness Detection**:
  - Upload an image and run object detection to classify items as "Fresh" or "Rotten."
  - View processed images with bounding boxes around detected objects.
  - Freshness status and object count are displayed for each detected item.
  
- **Processed Products**:
  - A table displaying freshness detection data, including timestamps, object class labels, and freshness status.

## Tech Stack

- **Streamlit**: Web framework for building the interactive app.
- **OpenCV**: For image processing and displaying results.
- **YOLO**: Custom-trained object detection model used for freshness detection.
- **Pandas**: To handle and display data in tabular format.
- **NumPy**: For numerical operations.
- **Ultralytics YOLO**: Library for using the YOLO object detection model.

## Getting Started

### Prerequisites

- Python 3.x
- `streamlit`, `opencv-python`, `pandas`, `numpy`, `ultralytics`

### Installation

1. Clone or download the repository.
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
3. Run the Streamlit application:

   ```bash
   streamlit run app.py

### Access the App
You can access the app here: [Produce Freshness Detection App](https://appucefreshness-apdutdse8fbqv8ghvr6noe.streamlit.app/)
