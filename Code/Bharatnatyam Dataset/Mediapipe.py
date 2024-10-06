# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/gdrive')

!pip install mediapipe

# Define the mapping from Mediapipe keypoints to SMPL keypoints
mediapipe_to_smpl_mapping = {
    0: 0,    # Nose -> Nose
    1: 8,    # Left Eye Inner -> Left Eye
    2: 9,    # Left Eye -> Right Eye
    3: 7,    # Right Eye Inner -> Right Eye
    4: 12,   # Left Ear -> Left Shoulder
    5: 13,   # Left Ear Tip -> Left Elbow
    6: 15,   # Right Ear -> Right Shoulder
    7: 16,   # Right Ear Tip -> Right Elbow
    8: 1,    # Left Shoulder -> Left Hip
    9: 2,    # Left Elbow -> Left Knee
    10: 3,   # Left Wrist -> Left Ankle
    11: 4,   # Right Shoulder -> Right Hip
    12: 5,   # Right Elbow -> Right Knee
    13: 6,   # Right Wrist -> Right Ankle
    14: 17,  # Left Hip -> Left Big Toe
    15: 18,  # Left Knee -> Left Small Toe
    16: 19,  # Left Ankle -> Left Heel
    17: 20,  # Right Hip -> Right Big Toe
    18: 21,  # Right Knee -> Right Small Toe
    19: 22,  # Right Ankle -> Right Heel
    20: -1,  # Left Pinky Finger -> Not available (set as -1)
    21: -1,  # Left Index Finger -> Not available (set as -1)
    22: -1,  # Right Pinky Finger -> Not available (set as -1)
    23: -1,  # Right Index Finger -> Not available (set as -1)
}

import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose.Pose()

# Define the mapping from Mediapipe keypoints to SMPL keypoints
mediapipe_to_smpl_mapping = {
    # Mapping dictionary goes here...
}

videos_dir = '/content/gdrive/MyDrive/Mediapipe/Videos'
outputs_dir = '/content/gdrive/MyDrive/Mediapipe/outputs'

# Create the outputs directory if it doesn't exist
os.makedirs(outputs_dir, exist_ok=True)

# Iterate over the videos in the videos directory
for video_file in os.listdir(videos_dir):
    # Get the full path of the video file
    video_path = os.path.join(videos_dir, video_file)

    # Open the video file for reading
    cap = cv2.VideoCapture(video_path)

    media_keypoints = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the image to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect the keypoints using MediaPipe Pose
        results = mp_pose.process(frame)

        # Extract the landmarks from the results
        landmarks = results.pose_landmarks

        # Store the landmarks for this frame
        media_keypoints.append(landmarks)

    cap.release()

    # Convert Mediapipe keypoints to SMPL keypoints
    smpl_keypoints = []
    for landmarks in media_keypoints:
        smpl_keypoint_coords = []
        for mp_idx, smpl_idx in mediapipe_to_smpl_mapping.items():
            if smpl_idx != -1 and landmarks is not None and landmarks.landmark[mp_idx].visibility > 0.5:
                x = landmarks.landmark[mp_idx].x
                y = landmarks.landmark[mp_idx].y
                smpl_keypoint_coords.append([x, y])
            else:
                smpl_keypoint_coords.append([0.0, 0.0])  # Missing keypoints are represented as [0, 0]
        smpl_keypoints.append(smpl_keypoint_coords)

    # Convert the keypoints list to a numpy array
    smpl_keypoints = np.array(smpl_keypoints)
    print(smpl_keypoints)

    # Save the SMPL keypoints as a text file
    #output_file = os.path.join(outputs_dir, f'{video_file[:-4]}.txt')
    #np.savetxt(output_file, smpl_keypoints, delimiter=',')

    #print(f"SMPL keypoints saved for {video_file}")

print("All videos processed.")

