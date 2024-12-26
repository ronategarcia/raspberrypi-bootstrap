# Imports
import mediapipe as mp
from picamera2 import Picamera2
import numpy as np
import time
import cv2

# Initialize the pi camera
pi_camera = Picamera2()
# Convert the color mode to RGB
config = pi_camera.create_preview_configuration(main={"format": "RGB888"})
pi_camera.configure(config)

# Start the pi camera and give it a second to set up
pi_camera.start()
time.sleep(1)

def draw_pose(image, landmarks):
    ''' 
    Draw circles on the landmarks and lines connecting the landmarks then return the image.

    Use the cv2.line and cv2.circle functions.

    landmarks is a collection of 33 dictionaries with the following keys:
        x: float values in the interval of [0.0,1.0]
        y: float values in the interval of [0.0,1.0]
        z: float values in the interval of [0.0,1.0]
        visibility: float values in the interval of [0.0,1.0]
    '''

    # Copy the image
    landmark_image = image.copy()

    # Get the dimensions of the image
    height, width, _ = image.shape
    
    # Draw landmarks and connections
    for landmark in landmarks.landmark:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        visibility = landmark.visibility
        if visibility > 0.5:  # Only draw visible landmarks
            cv2.circle(landmark_image, (x, y), 5, (255, 0, 255), -1)

    # Draw connections based on the MediaPipe Pose connections
    mp_pose = mp.solutions.pose
    for connection in mp_pose.POSE_CONNECTIONS:
        start = connection[0]
        end = connection[1]
        start_point = (
            int(landmarks.landmark[start].x * width),
            int(landmarks.landmark[start].y * height),
        )
        end_point = (
            int(landmarks.landmark[end].x * width),
            int(landmarks.landmark[end].y * height),
        )
        cv2.line(landmark_image, start_point, end_point, (0, 255, 255), 2)

    return landmark_image

def main():
    '''
    Show a video feed using the Pi camera and process the frames.
    '''
    # Create a pose estimation model 
    mp_pose = mp.solutions.pose

    # Start detecting the pose
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        while True:
            # Capture a frame from the Pi Camera
            frame = pi_camera.capture_array()

            # Convert the frame from RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # To improve performance, mark the image as not writable to pass by reference.
            frame.flags.writeable = False

            # Get the landmarks
            results = pose.process(frame)

            if results.pose_landmarks:
                # Draw the pose landmarks on the frame
                frame = draw_pose(frame, results.pose_landmarks)

            # Display the frame
            cv2.imshow('Pose Detection', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    print('done')
