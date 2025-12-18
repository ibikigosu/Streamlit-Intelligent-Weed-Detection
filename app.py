import io
import os
from datetime import datetime, timezone

import cv2
import gridfs
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from pymongo import MongoClient
from ultralytics import YOLO

# Pulls in environment variables from the .env file so the app can access secrets securely
load_dotenv()

# Grabs the MongoDB connection string from the environment
MONGO_URI = os.getenv("MONGO_URI")


@st.cache_resource
def setup_mongodb():
    try:
        client = MongoClient(MONGO_URI)
        db = client["weed_detection"]
        fs = gridfs.GridFS(db)

        if "uploads" not in db.list_collection_names():
            db.create_collection("uploads")
        if "detections" not in db.list_collection_names():
            db.create_collection("detections")

        return db, fs
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {str(e)}")
        return None, None


@st.cache_resource
def instantiate_model():
    model = YOLO("model/best.pt")
    return model


top_image = Image.open("static/banner_top.png")
bottom_image = Image.open("static/banner_bottom.png")
main_image = Image.open("static/main_banner.png")

upload_path = "uploads/"
download_path = "downloads/"
model = instantiate_model()

# Sets up the database connection and GridFS for file storage
db, fs = setup_mongodb()

if db is None:
    st.error("Could not connect to database. Please check your connection settings.")
    st.stop()

st.image(main_image, use_container_width="auto")
st.title(" Intelligent Weed Detection 🔎🌱")
st.sidebar.image(top_image, use_container_width="auto")
st.sidebar.header("Input 🛠")
selected_type = st.sidebar.selectbox(
    "Please select an activity type 🚀", ["Upload Image", "Live Video Feed"]
)
st.sidebar.image(bottom_image, use_container_width="auto")


if selected_type == "Upload Image":
    st.info("✨ Supports all popular image formats 📷 - PNG, JPG, BMP")
    uploaded_file = st.file_uploader(
        "Upload Image of Weed 🌱", type=["png", "jpg", "bmp", "jpeg"]
    )

    if uploaded_file is not None:
        try:
            # Converts the uploaded file into a numpy array so OpenCV can work with it
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Builds the metadata object to track upload details in the database
            upload_info = {
                "filename": uploaded_file.name,
                "upload_date": datetime.now(timezone.utc),
                "file_type": uploaded_file.type,
            }

            # Saves the original file to GridFS, resetting the pointer first since it was already read above
            uploaded_file.seek(0)
            file_id = fs.put(
                uploaded_file.getvalue(),
                filename=uploaded_file.name,
                metadata=upload_info,
            )

            # Runs the YOLO model on the image to detect weeds
            results = model.predict(source=image, conf=0.25, iou=0.45)
            res_plotted = results[0].plot()

            # OpenCV uses BGR by default, but Streamlit expects RGB for display
            res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

            # Prepares the processed image for storage in the database
            processed_img = Image.fromarray(res_plotted_rgb)
            processed_bytes = io.BytesIO()
            processed_img.save(
                processed_bytes, format=uploaded_file.type.split("/")[-1].upper()
            )

            # Creates the metadata linking the processed image back to its original
            detection_info = {
                "original_file_id": file_id,
                "detection_date": datetime.now(timezone.utc),
                "filename": f"processed_{uploaded_file.name}",
            }

            processed_id = fs.put(
                processed_bytes.getvalue(),
                filename=f"processed_{uploaded_file.name}",
                metadata=detection_info,
            )

            # Shows the user the image with detection boxes drawn on it
            st.markdown("---")
            st.image(processed_img, caption="Processed Image with Detections")

            # Lets the user download the processed image to their device
            if st.download_button(
                label="Download Output Image 📷",
                data=processed_bytes.getvalue(),
                file_name=f"output_{uploaded_file.name}",
                mime=uploaded_file.type,
            ):
                st.success("✅ Download Successful")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    else:
        st.warning("⚠ Please upload your Image")


else:
    st.info("✨ The Live Feed from Web-Camera will take some time to load up 🎦")
    live_feed = st.checkbox("Start Web-Camera ✅")
    FRAME_WINDOW = st.image([])

    try:
        # Loops through common camera indices since different systems assign webcams differently
        camera_indices = [0, 1, 2]
        cap = None

        for idx in camera_indices:
            cap = cv2.VideoCapture(idx)
            if cap is not None and cap.isOpened():
                break

        if cap is None or not cap.isOpened():
            st.error(
                "❌ Unable to access webcam. Please check your camera permissions and connection."
            )
            st.info(
                "Make sure to:\n"
                "1. Allow camera access in your browser\n"
                "2. Connect a webcam to your device\n"
                "3. Make sure no other application is using the camera"
            )
        else:
            if live_feed:
                try:
                    while cap.isOpened():
                        success, frame = cap.read()
                        if success:
                            results = model.predict(source=frame, conf=0.25, iou=0.45)
                            annotated_frame = results[0].plot()
                            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            FRAME_WINDOW.image(rgb_frame)
                        else:
                            break
                except Exception as e:
                    st.error(f"❌ Error during video processing: {str(e)}")
                finally:
                    if cap is not None:
                        cap.release()
            else:
                if cap is not None:
                    cap.release()
                st.warning("⚠ The Web-Camera is currently disabled. 😯")

    except Exception as e:
        st.error(f"❌ Error initializing camera: {str(e)}")
