import streamlit as st
import os
from PIL import Image
import cv2
from ultralytics import YOLO
from pymongo import MongoClient
import gridfs
import io
from datetime import datetime, timezone
import numpy as np

# MongoDB connection setup
MONGO_URI = os.getenv('MONGO_URI', 'mongodb+srv://ibiki:qFymvxsK6PKMzoon@machinelearning.bmu3u.mongodb.net/?retryWrites=true&w=majority&appName=MachineLearning')

@st.cache_resource
def setup_mongodb():
    try:
        client = MongoClient(MONGO_URI)
        db = client['weed_detection']
        fs = gridfs.GridFS(db)
        
        if 'uploads' not in db.list_collection_names():
            db.create_collection('uploads')
        if 'detections' not in db.list_collection_names():
            db.create_collection('detections')
            
        return db, fs
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {str(e)}")
        return None, None

@st.cache_resource
def instantiate_model():
    model = YOLO('model/best.pt')
    return model

top_image = Image.open('static/banner_top.png')
bottom_image = Image.open('static/banner_bottom.png')
main_image = Image.open('static/main_banner.png')

upload_path = "uploads/"
download_path = "downloads/"
model = instantiate_model()

# Initialize MongoDB connection
db, fs = setup_mongodb()

if db is None:
    st.error("Could not connect to database. Please check your connection settings.")
    st.stop()

st.image(main_image, use_container_width='auto')
st.title(' Intelligent Weed Detection 🔎🌱')
st.sidebar.image(top_image, use_container_width='auto')
st.sidebar.header('Input 🛠')
selected_type = st.sidebar.selectbox('Please select an activity type 🚀', ["Upload Image", "Live Video Feed"])
st.sidebar.image(bottom_image, use_container_width='auto')

if selected_type == "Upload Image":
    st.info('✨ Supports all popular image formats 📷 - PNG, JPG, BMP')
    uploaded_file = st.file_uploader("Upload Image of Weed 🌱", type=["png","jpg","bmp","jpeg"])

    if uploaded_file is not None:
        try:
            # Convert uploaded file to numpy array
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Store original image metadata
            upload_info = {
                'filename': uploaded_file.name,
                'upload_date': datetime.now(timezone.utc),
                'file_type': uploaded_file.type
            }
            
            # Store original file
            uploaded_file.seek(0)  # Reset file pointer
            file_id = fs.put(
                uploaded_file.getvalue(),
                filename=uploaded_file.name,
                metadata=upload_info
            )
            
            # Process image with YOLO
            results = model.predict(source=image, conf=0.5, iou=0.45)
            res_plotted = results[0].plot()
            
            # Convert BGR to RGB for display
            res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            # Store processed image
            processed_img = Image.fromarray(res_plotted_rgb)
            processed_bytes = io.BytesIO()
            processed_img.save(processed_bytes, format=uploaded_file.type.split('/')[-1].upper())
            
            # Store detection results
            detection_info = {
                'original_file_id': file_id,
                'detection_date': datetime.now(timezone.utc),
                'filename': f"processed_{uploaded_file.name}"
            }
            
            processed_id = fs.put(
                processed_bytes.getvalue(),
                filename=f"processed_{uploaded_file.name}",
                metadata=detection_info
            )
            
            # Display processed image
            st.markdown("---")
            st.image(processed_img, caption='Processed Image with Detections')
            
            # Download button
            if st.download_button(
                label="Download Output Image 📷",
                data=processed_bytes.getvalue(),
                file_name=f"output_{uploaded_file.name}",
                mime=uploaded_file.type
            ):
                st.success('✅ Download Successful')
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    else:
        st.warning('⚠ Please upload your Image')


else:
    st.info('✨ The Live Feed from Web-Camera will take some time to load up 🎦')
    live_feed = st.checkbox('Start Web-Camera ✅')
    FRAME_WINDOW = st.image([])
    
    try:
        # Try different camera indices
        camera_indices = [0, 1, 2]
        cap = None
        
        for idx in camera_indices:
            cap = cv2.VideoCapture(idx)
            if cap is not None and cap.isOpened():
                break
        
        if cap is None or not cap.isOpened():
            st.error("❌ Unable to access webcam. Please check your camera permissions and connection.")
            st.info("Make sure to:\n"
                   "1. Allow camera access in your browser\n"
                   "2. Connect a webcam to your device\n"
                   "3. Make sure no other application is using the camera")
        else:
            if live_feed:
                try:
                    while cap.isOpened():
                        success, frame = cap.read()
                        if success:
                            results = model.predict(source=frame, conf=0.5, iou=0.45)
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
                st.warning('⚠ The Web-Camera is currently disabled. 😯')
                
    except Exception as e:
        st.error(f"❌ Error initializing camera: {str(e)}")




