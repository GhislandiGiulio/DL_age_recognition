import cv2
import streamlit as st
import numpy as np

from PIL import Image 

import os
# impostazione della cartella di esecuzione corretta
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)


from tensorflow.keras.models import load_model

# caricamento del modello
need_loading = 1

if need_loading:

    model = load_model("age_prediction_model.keras")

    need_loading += 1

# titolo dell'applicazione
st.title("Age Prediction")

# Variable to store the webcam capture object
cap = None

# Variable to store the last captured frame
captured_frame = None

# frame webcam
stframe = st.empty()

take_picture_button = None

captured_frame = None


def video_capture():

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

        # prediction of age of frame
        resized_gray_image = process_frame(frame_rgb_flipped)

        pred_age = model.predict(resized_gray_image, verbose=0)

        pred_age = np.round(pred_age).astype(int)
        pred_age = pred_age.reshape(-1)

        show_age(pred_age, frame_with_oval)

        # Display the frame
        stframe.image(frame_with_oval, channels="RGB")



        # Add a small delay to make the loop run at a reasonable speed
        cv2.waitKey(1)

        
    cap.release()

def show_age(pred_age, frame):

    text = f"Age: {pred_age//100}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1

    # Choose the thickness of the line
    thickness = 2

    # Determine the size of the text
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate the position to put the text (bottom-left corner)
    text_x = 10
    text_y = frame.shape[0] - 10  # Adjust the y-coordinate as needed

    # Add text to the image
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)


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

    cropped_image = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]

    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    resized_gray_image = cv2.resize(gray_image, (128, 128), interpolation=cv2.INTER_LANCZOS4)
    
    np.save("./output.jpg", frame)

    resized_gray_image = np.expand_dims(resized_gray_image, axis=-1)

    resized_gray_image = np.expand_dims(resized_gray_image, axis=0)

    return resized_gray_image
    

video_capture()