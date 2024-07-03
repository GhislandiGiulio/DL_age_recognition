import cv2
import streamlit as st
import numpy as np

import os
# impostazione della cartella di esecuzione corretta
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

# caricamento del modello
from tensorflow.keras.models import load_model

model = load_model("age_prediction_model.keras")

# titolo dell'applicazione
st.title("Age Prediction")

# Variable to store the webcam capture object
cap = None

# Variable to store the last captured frame
captured_frame = None

# Start and stop buttons
start_button = st.button("Start")
stop_button = st.button("Stop")

# frame webcam
stframe = st.empty()

take_picture_button = None

captured_frame = None


def video_capture():
    
    global take_picture_button
    global captured_frame

    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        st.error("Could not open webcam.")
        cap.release()
        st.stop()

    # Frame streaming loop
    while cap.isOpened():

        # Read a frame from the webcam stream
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image.")
            break

        # Get the dimensions of the frame
        height, width, _ = frame.shape

        # Define the center and axes of the oval
        center = (width // 2, height // 2)
        axes = (width // 6, height // 3)  # Half the width and height of the frame

        # Define the color and thickness of the oval
        color = (0, 255, 0)  # Green color in BGR format
        thickness = 2

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb_flipped = cv2.flip(frame_rgb, 1)
        
        frame_with_oval = frame_rgb_flipped

        cv2.ellipse(frame_with_oval, center, axes, angle=0, startAngle=0, endAngle=360, color=color, thickness=thickness)

        # Display the frame
        stframe.image(frame_with_oval, channels="RGB")

        process_frame(frame_rgb_flipped)

        # Add a small delay to make the loop run at a reasonable speed
        cv2.waitKey(1)

        
    cap.release()


def process_frame(frame):
    
    height, width, _ = frame.shape

    # Crop the image
    # Define the crop box (left, upper, right, lower)
    upper = height // 6
    lower = height - (height // 6)
    crop_height = lower - upper

    # Calculate left and right to make the crop box square
    center_x = width // 2
    half_crop_width = crop_height // 2
    left = max(0, center_x - half_crop_width)
    right = min(width, center_x + half_crop_width)

    # Define the crop box (left, upper, right, lower)
    crop_box = (left, upper, right, lower)

    cropped_image_np = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
    

if start_button:
    video_capture()


if stop_button:
    cap = None