import cv2
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import streamlit as st

# Define a multipage app
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", ["Freshness Detection", "Processed Products"])

# Initialize session state for tables
if 'freshness_data' not in st.session_state:
    st.session_state.freshness_data = []

# Helper function for timestamp
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Freshness Detection Page
if page == "Freshness Detection":
    st.title("Freshness Detection")

    @st.cache_resource
    def load_freshness_model():
        from ultralytics import YOLO
        return YOLO("best.pt")  # Replace with your YOLO model path

    model = load_freshness_model()

    def process_and_display_image(image_path):
        results = model.predict(source=image_path, conf=0.3, imgsz=640)
        image = cv2.imread(image_path)
        counts = Counter()

        for result in results:
            for box in result.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = box
                class_name = model.names[int(class_id)]
                counts[class_name] += 1
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        st.image(image, caption="Processed Image", use_column_width=True)
        st.write("### Detected Objects")
        for class_name, count in counts.items():
            freshness_status = "Fresh" if "fresh" in class_name.lower() else "Rotten"
            timestamp = get_timestamp()
            st.session_state.freshness_data.append({
                "Timestamp": timestamp,
                "Class Label": class_name,
                "Freshness Status": freshness_status,
                "Count": count
            })

    uploaded_file = st.file_uploader("Upload an image for detection", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        temp_file = "temp_image.jpg"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Run Detection"):
            with st.spinner("Processing the image..."):
                process_and_display_image(temp_file)

# Processed Products Page
elif page == "Processed Products":
    st.title("Processed Products")

    st.write("### Freshness Detection Data")
    st.dataframe(pd.DataFrame(st.session_state.freshness_data).style.set_table_styles([{
        "selector": "table",
        "props": [("overflow-x", "scroll")]
    }]))