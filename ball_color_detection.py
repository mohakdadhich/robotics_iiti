import cv2 as cv
import numpy as np

def detect_balls(image_path):
    img = cv.imread(image_path)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    red1_lower = np.array([0,60,50])       # max can be [179, 255,255]
    red1_upper = np.array([10,255,255])

    red2_lower = np.array([169,100,100])
    red2_upper = np.array([179,255,255])

    blue_lower = np.array([100,50,50])
    blue_upper = np.array([130,255,255])

    red_mask1 = cv.inRange(hsv, red1_lower, red1_upper)
    red_mask2 = cv.inRange(hsv, red2_lower, red2_upper)
    red_mask = cv.bitwise_or(red_mask1, red_mask2)
    blue_mask = cv.inRange(hsv, blue_lower, blue_upper)

    red_contour, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    blue_contour, _ = cv.findContours(blue_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in red_contour:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.putText(img, "Red", (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    for contour in blue_contour:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.putText(img, "Blue", (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    cv.imshow("Detected balls", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

image_path = "Ball color detection\color_ball1.jpeg"

detect_balls(image_path)

