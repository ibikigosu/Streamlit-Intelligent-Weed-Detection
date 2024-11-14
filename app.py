import streamlit as st
import os
from PIL import Image
import cv2
from ultralytics import YOLO

st.set_page_config(
    page_title="Weed Detection",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def instantiate_model():
    model = YOLO('model/best.pt')
    return model

@st.cache_data
def download_success():
    st.balloons()
    st.success('✅ Download Successful !!')

top_image = Image.open('static/banner_top.png')
bottom_image = Image.open('static/banner_bottom.png')
main_image = Image.open('static/main_banner.png')

upload_path = "uploads/"
download_path = "downloads/"
model = instantiate_model()

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
        with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
            f.write((uploaded_file).getbuffer())
        with st.spinner(f"Working... 💫"):
            uploaded_image = os.path.abspath(os.path.join(upload_path,uploaded_file.name))
            downloaded_image = os.path.abspath(os.path.join(download_path,str("output_"+uploaded_file.name)))

            results = model.predict(source=uploaded_image, conf=0.5, iou=0.45)
            
            res_plotted = results[0].plot()
            im_array = Image.fromarray(res_plotted)
            im_array.save(downloaded_image)

            final_image = Image.open(downloaded_image)
            st.markdown("---")
            st.image(final_image, caption='This is how your final image looks like')
            with open(downloaded_image, "rb") as file:
                if uploaded_file.name.endswith('.jpg') or uploaded_file.name.endswith('.JPG'):
                    if st.download_button(
                                            label="Download Output Image 📷",
                                            data=file,
                                            file_name=str("output_"+uploaded_file.name),
                                            mime='image/jpg'
                                         ):
                        download_success()
                if uploaded_file.name.endswith('.jpeg') or uploaded_file.name.endswith('.JPEG'):
                    if st.download_button(
                                            label="Download Output Image 📷",
                                            data=file,
                                            file_name=str("output_"+uploaded_file.name),
                                            mime='image/jpeg'
                                         ):
                        download_success()

                if uploaded_file.name.endswith('.png') or uploaded_file.name.endswith('.PNG'):
                    if st.download_button(
                                            label="Download Output Image 📷",
                                            data=file,
                                            file_name=str("output_"+uploaded_file.name),
                                            mime='image/png'
                                         ):
                        download_success()

                if uploaded_file.name.endswith('.bmp') or uploaded_file.name.endswith('.BMP'):
                    if st.download_button(
                                            label="Download Output Image 📷",
                                            data=file,
                                            file_name=str("output_"+uploaded_file.name),
                                            mime='image/bmp'
                                         ):
                        download_success()
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


