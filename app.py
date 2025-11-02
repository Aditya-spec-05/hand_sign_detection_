import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- 1. Load Your Trained YOLOv8 Model ---
@st.cache_resource
def load_yolo_model():
    # --- THIS IS THE UPDATED PATH ---
    model_path = r"D:\projects\handsigndetection\best.pt"
    # --------------------------------
    
    try:
        model = YOLO(model_path)
        st.success(f"Model '{model_path}' loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None

model = load_yolo_model()

# --- 2. Define the Video Transformer Class ---
# This class will process each frame from the webcam
class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_yolo_model()

    def transform(self, frame):
        if self.model is None:
            # If model failed to load, return the original frame
            return frame.to_ndarray(format="bgr24")

        # Convert the frame to a format OpenCV can use
        img = frame.to_ndarray(format="bgr24")

        # --- Run Inference ---
        try:
            # Run YOLOv8 inference on the frame
            results = self.model(img, verbose=False) # verbose=False to reduce logs

            # Get the annotated frame (with boxes and labels)
            annotated_frame = results[0].plot()

            return annotated_frame
        
        except Exception as e:
            print(f"Error during inference: {e}")
            return img # Return original image on error

# --- 3. Set Up the Streamlit Page ---
st.set_page_config(page_title="Hand Sign Detection", layout="centered")
st.title("Live Hand Sign Detection 🖐️")

if model is None:
    st.warning("Model could not be loaded. Please check the file path.")
else:
    st.success("Model loaded. Click 'START' below to access your webcam.")

    # --- 4. Start the WebRTC Streamer ---
    webrtc_streamer(
        key="hand-sign-detector",
        mode=WebRtcMode.SENDRECV,
        # Use the class we defined above to process frames
        video_processor_factory=YOLOVideoTransformer, 
        media_stream_constraints={
            "video": {
                "width": 640,
                "height": 480
            },
            "audio": False
        },
        async_processing=True,
    )

st.caption("This app uses a YOLOv8 model trained to detect 16 hand signs.")
