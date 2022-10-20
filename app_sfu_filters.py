import urllib.request
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_server_state import server_state, server_state_lock
from streamlit_webrtc import ClientSettings, WebRtcMode, webrtc_streamer

cv2_path = Path(cv2.__file__).parent


def imread_from_url(url: str):
    req = urllib.request.urlopen(url)
    encoded = np.asarray(bytearray(req.read()), dtype="uint8")
    image_bgra = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)

    return image_bgra


def overlay_bgra(background: np.ndarray, overlay: np.ndarray, roi):
    roi_x, roi_y, roi_w, roi_h = roi
    roi_aspect_ratio = roi_w / roi_h

    # Calc overlay x, y, w, h that cover the ROI keeping the original aspect ratio
    ov_org_h, ov_org_w = overlay.shape[:2]
    ov_aspect_ratio = ov_org_w / ov_org_h

    if ov_aspect_ratio >= roi_aspect_ratio:
        ov_h = roi_h
        ov_w = int(ov_aspect_ratio * ov_h)
        ov_y = roi_y
        ov_x = int(roi_x - (ov_w - roi_w) / 2)
    else:
        ov_w = roi_w
        ov_h = int(ov_w / ov_aspect_ratio)
        ov_x = roi_x
        ov_y = int(roi_y - (ov_h - roi_h) / 2)

    resized_overlay = cv2.resize(overlay, (ov_w, ov_h))

    # Cut out the pixels of the overlay image outside the background frame.
    margin_x0 = -min(0, ov_x)
    margin_y0 = -min(0, ov_y)
    margin_x1 = max(background.shape[1], ov_x + ov_w) - background.shape[1]
    margin_y1 = max(background.shape[0], ov_y + ov_h) - background.shape[0]

    resized_overlay = resized_overlay[
        margin_y0 : resized_overlay.shape[0] - margin_y1,
        margin_x0 : resized_overlay.shape[1] - margin_x1,
    ]
    ov_x += margin_x0
    ov_w -= margin_x0 + margin_x1
    ov_y += margin_y0
    ov_h -= margin_y0 + margin_y1

    # Overlay
    foreground = resized_overlay[:, :, :3]
    mask = resized_overlay[:, :, 3]

    overlaid_area = background[ov_y : ov_y + ov_h, ov_x : ov_x + ov_w]
    overlaid_area[:] = np.where(mask[:, :, np.newaxis], foreground, overlaid_area)


@st.experimental_singleton
def get_face_classifier():
    return cv2.CascadeClassifier(
        str(cv2_path / "data/haarcascade_frontalface_alt2.xml")
    )


@st.experimental_singleton
def get_filters():
    return {
        "ironman": imread_from_url(
            "https://i.pinimg.com/originals/0c/c0/50/0cc050fd99aad66dc434ce772a0449a9.png"  # noqa: E501
        ),
        "laughing_man": imread_from_url(
            "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/3a17e5a4-9610-4fa3-a4bd-cb7d94d6f7e1/darwcty-d989aaf1-3cfa-4576-b2ac-305209346162.png/v1/fill/w_944,h_847,strp/laughing_man_logo_by_aggressive_vector_darwcty-pre.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9OTE5IiwicGF0aCI6IlwvZlwvM2ExN2U1YTQtOTYxMC00ZmEzLWE0YmQtY2I3ZDk0ZDZmN2UxXC9kYXJ3Y3R5LWQ5ODlhYWYxLTNjZmEtNDU3Ni1iMmFjLTMwNTIwOTM0NjE2Mi5wbmciLCJ3aWR0aCI6Ijw9MTAyNCJ9XV0sImF1ZCI6WyJ1cm46c2VydmljZTppbWFnZS5vcGVyYXRpb25zIl19.5SDBnNZF6ktZM7Mk5gJfpHNQswRba3eqpvUn6FMHyW4"  # noqa: E501
        ),
        "cat": imread_from_url(
            "https://i.pinimg.com/originals/29/cd/fd/29cdfdf2248ce2465598b2cc9e357579.png"  # noqa: E501
        ),
    }


def main():
    if "webrtc_contexts" not in server_state:
        server_state["webrtc_contexts"] = []

    face_cascade = get_face_classifier()
    filters = get_filters()

    filter_type = st.radio(
        "Select filter type",
        ("ironman", "laughing_man", "cat"),
        key="filter-type",
    )
    draw_rect = st.checkbox("Draw rect (for debug)")

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.11, minNeighbors=3, minSize=(30, 30)
        )

        overlay = filters[filter_type]

        for (x, y, w, h) in faces:
            # Ad-hoc adjustment of the ROI for each filter type
            if filter_type == "ironman":
                roi = (x, y, w, h)
            elif filter_type == "laughing_man":
                roi = (x, y, int(w * 1.15), h)
            elif filter_type == "cat":
                roi = (x, y - int(h * 0.3), w, h)
            overlay_bgra(img, overlay, roi)

            if draw_rect:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    self_ctx = webrtc_streamer(
        key="self",
        mode=WebRtcMode.SENDRECV,
        client_settings=ClientSettings(
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": True, "audio": True},
        ),
        video_frame_callback=video_frame_callback,
        sendback_audio=False,
    )

    with server_state_lock["webrtc_contexts"]:
        webrtc_contexts = server_state["webrtc_contexts"]
        if self_ctx.state.playing and self_ctx not in webrtc_contexts:
            webrtc_contexts.append(self_ctx)
            server_state["webrtc_contexts"] = webrtc_contexts
        elif not self_ctx.state.playing and self_ctx in webrtc_contexts:
            webrtc_contexts.remove(self_ctx)
            server_state["webrtc_contexts"] = webrtc_contexts

    active_other_ctxs = [
        ctx for ctx in webrtc_contexts if ctx != self_ctx and ctx.state.playing
    ]

    for ctx in active_other_ctxs:
        webrtc_streamer(
            key=str(id(ctx)),
            mode=WebRtcMode.RECVONLY,
            client_settings=ClientSettings(
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                media_stream_constraints={
                    "video": True,
                    "audio": True,
                },
            ),
            source_audio_track=ctx.output_audio_track,
            source_video_track=ctx.output_video_track,
            desired_playing_state=ctx.state.playing,
        )


if __name__ == "__main__":
    main()
