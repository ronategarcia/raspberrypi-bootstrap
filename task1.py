# Imports
import mediapipe as mp
from picamera2 import Picamera2
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
    TODO Task 1

    Code to this fucntion to draw circles on the landmarks and lines
    connecting the landmarks then return the image.

    Use the cv2.line and cv2.circle functions.

    landmarks is a collection of 33 dictionaries with the following keys
        x: float values in the interval of [0.0,1.0]
        y: float values in the interval of [0.0,1.0]
        z: float values in the interval of [0.0,1.0]
        visibility: float values in the interval of [0.0,1.0]
        
    References:
    https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
    https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    '''

    # copy the image
    landmark_image = image.copy()

    # get the dimensions of the image
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
    TODO Task 2
        modify this fucntion to take a photo uses the pi camera instead 
        of loading an image

    TODO Task 3
        modify this function further to loop and show a video
    '''

    # Create a pose estimation model 
    mp_pose = mp.solutions.pose

    # start detecting the poses
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        # load test image
        image = cv2.imread("person.png")

        # To improve performance, optionally mark the image as not 
        # writeable to pass by reference.
        image.flags.writeable = False
        
        # get the landmarks
        results = pose.process(image)
        
        if results.pose_landmarks != None:
            result_image = draw_pose(image, results.pose_landmarks)
            cv2.imwrite('output.png', result_image)
            print(results.pose_landmarks)
        else:
            print('No Pose Detected')


if __name__ == "__main__":
    main()
    print('done')

