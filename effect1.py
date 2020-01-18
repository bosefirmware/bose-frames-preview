import cv2
import numpy as np
import dlib
from math import hypot

# Loading Camera and Nose image and Creating mask
frameImg = cv2.imread("boseframesnolens.png")
cap = cv2.VideoCapture(0)

# Loading Face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def applyFilter(frame):
    rows, cols, _ = frame.shape
    frameMask = np.zeros((rows, cols), np.uint8)

    frameMask.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)

        # Nose coordinates
        topFrame = (landmarks.part(27).x, landmarks.part(27).y)
        centerFrame = (landmarks.part(28).x, landmarks.part(28).y)
        leftFrame = (landmarks.part(45).x, landmarks.part(45).y)
        rightFrame = (landmarks.part(36).x, landmarks.part(36).y)

        frameWidth = int(hypot(leftFrame[0] - rightFrame[0],
                            leftFrame[1] - rightFrame[1]) * 1.7)
        frameHeight = int(frameWidth * 0.3375)

        # New nose position
        top_left = (int(centerFrame[0] - frameWidth / 2),
                    int(centerFrame[1] - frameHeight / 2))
        bottom_right = (int(centerFrame[0] + frameWidth / 2),
                        int(centerFrame[1] + frameHeight / 2))

        # Adding the new nose
        resizedGlasses = cv2.resize(frameImg, (frameWidth, frameHeight))
        resizedGlassesGray = cv2.cvtColor(resizedGlasses, cv2.COLOR_BGR2GRAY)
        
        _, frameMask = cv2.threshold(
            resizedGlassesGray, 1, 255, cv2.THRESH_BINARY_INV)

        frameArea = frame[top_left[1]: top_left[1] + frameHeight,
                        top_left[0]: top_left[0] + frameWidth]
        frameAreaNoFrame = cv2.bitwise_and(
            frameArea, frameArea, mask=frameMask)



        final_nose = cv2.add(frameAreaNoFrame, resizedGlasses)
        frame[top_left[1]: top_left[1] + frameHeight,
            top_left[0]: top_left[0] + frameWidth] = final_nose
        
    
    return frame

if __name__ == "__main__":
     while True:
        _, curFrame = cap.read()
        cv2.imshow("Frame", applyFilter(curFrame))



        key = cv2.waitKey(1)
        if key == 27:
            break
