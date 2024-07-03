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

# frame webcam
stframe = st.empty()


def video_capture():

    # Open the webcam
    cap = cv2.VideoCapture(0)

    # check cattura webcam
    if not cap.isOpened():
        st.error("Could not open webcam.")
        cap.release()
        st.stop()

    # loop di streaming dei frame
    while cap.isOpened():

        # lettura di un frame della webcam
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image.")
            break

        # conversione a formato rgb
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb_flipped = cv2.flip(frame_rgb, 1)
        
        # aggiunta dell'ovale al frame
        frame_with_oval = frame_rgb_flipped
        add_oval(frame_with_oval)

        # pre-processing del frame per la predizione
        resized_gray_image = process_frame(frame_rgb_flipped)

        # previsione dell'età del frame
        pred_age = model.predict(resized_gray_image, verbose=0)

        # trasformazione delle età predette da float a intero
        pred_age = np.round(pred_age).astype(int)
        pred_age = pred_age.reshape(-1)

        # chiamata funzione per aggiungere età predetta a frame
        show_age(pred_age, frame_with_oval)

        # stampa del frame
        stframe.image(frame_with_oval, channels="RGB")

        # sleep
        cv2.waitKey(1)

        
    cap.release()

def add_oval(frame):

    # dimensioni del frame
    height, width, _ = frame.shape

    # definizione centro e assi dell'ovale
    center = (width // 2, height // 2)
    axes = (width // 6, height // 3) 

    # definizione colore e spessore ovale
    color = (0, 255, 0)  # verde in formato BGR
    thickness = 2
    
    cv2.ellipse(frame, center, axes, angle=0, startAngle=0, endAngle=360, color=color, thickness=thickness)

def show_age(pred_age, frame):

    # variabile contenente il testo da mostrare
    text = f"Age: {pred_age//100}"

    # impostazione del font
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.7
    thickness = 1

    # posizionamento del testo
    text_x = 10
    text_y = frame.shape[0] - 10  # Adjust the y-coordinate as needed

    # aggiunta del testo all'immagine
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)


def process_frame(frame):
    
    # dimensioni del frame della webcam
    height, width, _ = frame.shape

    # Crop the image
    # definzione della crop_box sopra e sotto
    upper = height // 6
    lower = height - (height // 6)
    crop_height = lower - upper

    # calcolo di sinistra e destra per rendere l'immagine quadrata
    center_x = width // 2
    half_crop_width = crop_height // 2
    left = max(0, center_x - half_crop_width)
    right = min(width, center_x + half_crop_width)

    # definizione della crop_box
    crop_box = (left, upper, right, lower)

    # applicazione delle trasformazioni all'immagine
    cropped_image = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]] # cropping
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) # grayscaling
    resized_gray_image = cv2.resize(gray_image, (128, 128), interpolation=cv2.INTER_LANCZOS4) # resizing
    resized_gray_image = np.expand_dims(resized_gray_image, axis=-1) # aumento di dimensioni
    resized_gray_image = np.expand_dims(resized_gray_image, axis=0)

    return resized_gray_image
    

video_capture()