import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import time
from datetime import datetime
import pandas as pd
import os


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


st.set_page_config(layout='wide')
st.title("🏋️ AI Fitness Coach")
st.markdown("Real-time pose detection and rep counting using webcam + MediaPipe")

exercise = st.sidebar.selectbox("Choose Exercise", ["Bicep Curl"])
log_enabled = st.sidebar.checkbox("Save Workout Log")
start_btn = st.sidebar.button("▶️ Start Workout")

frame_placeholder = st.empty()
rep_display = st.sidebar.empty()


counter = 0
stage = None
logs = []
start_time = time.time()

if start_btn:
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Webcam not accessible")
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                landmarks = results.pose_landmarks.landmark

                
                shoulder = [landmarks[12].x, landmarks[12].y]
                elbow = [landmarks[14].x, landmarks[14].y]
                wrist = [landmarks[16].x, landmarks[16].y]

                angle = calculate_angle(shoulder, elbow, wrist)

                
                if angle > 160:
                    stage = "down"
                if angle < 30 and stage == "down":
                    stage = "up"
                    counter += 1
                    logs.append({
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'exercise': exercise,
                        'reps': counter
                    })

                
                cv2.putText(frame, f'Angle: {int(angle)}', (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            
            elapsed = int(time.time() - start_time)
            cv2.putText(frame, f'Reps: {counter}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, f'Time: {elapsed}s', (10, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

            frame_placeholder.image(frame, channels='BGR')
            rep_display.markdown(f"### ✅ Reps: {counter}")
            rep_display.markdown(f"⏱️ Time: {elapsed} seconds")

            if st.sidebar.button("❌ Stop Workout"):
                break

        cap.release()
        cv2.destroyAllWindows()

        
        if log_enabled and logs:
            df = pd.DataFrame(logs)
            os.makedirs("logs", exist_ok=True)
            filename = f"logs/workout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            st.success(f"✅ Workout log saved: `{filename}`")
