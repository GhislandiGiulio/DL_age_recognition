import streamlit as st
import cv2
from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.capture = cv2.VideoCapture(0)

    def transform(self, frame):
        ret, frame = self.capture.read()
        if not ret:
            st.error("Failed to capture frame from webcam.")
            return frame

        return frame


def main():
    st.title("Webcam Picture Capture")

    # Create a WebRTC streamer with custom settings
    webrtc_ctx = webrtc_streamer(
        key="snapshot",
        mode=WebRtcMode.SENDRECV,
        client_settings=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True},
        ),
        video_transformer_factory=VideoTransformer,
        async_transform=True,
    )

    if webrtc_ctx.video_transformer:
        st.image(webrtc_ctx.video_transformer.frame, channels="BGR")


if __name__ == "__main__":
    main()
