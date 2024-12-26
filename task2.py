# Imports
import mediapipe as mp
from picamera2 import Picamera2
import time
import cv2
import numpy as np

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
    TODO Task 3
        modify this function further to loop and show a video
    '''
    # Create a pose estimation model 
    mp_pose = mp.solutions.pose

    # Open a window with a countdown timer
    for i in range(3, 0, -1):
        countdown_image = 255 * np.ones((500, 500, 3), dtype=np.uint8)
        cv2.putText(countdown_image, str(i), (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 5)
        cv2.imshow('Countdown', countdown_image)
        cv2.waitKey(1000)

    cv2.destroyWindow('Countdown')

    # Capture an image from the Pi Camera
    frame = pi_camera.capture_array()

    # Convert the frame from RGB to BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Start detecting the pose
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        # To improve performance, mark the image as not writable to pass by reference.
        frame.flags.writeable = False

        # Get the landmarks
        results = pose.process(frame)

        if results.pose_landmarks:
            # Draw the pose landmarks on the frame
            result_image = draw_pose(frame, results.pose_landmarks)

            # Save the result image
            cv2.imwrite('output2.png', result_image)
            print('Pose landmarks detected and saved as output.png')
        else:
            print('No Pose Detected')

if __name__ == "__main__":
    main()
    print('done')
