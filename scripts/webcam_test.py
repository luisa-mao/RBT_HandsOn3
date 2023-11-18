import cv2
from matplotlib import pyplot as plt

# make a sift detector
sift = cv2.xfeatures2d.SIFT_create(50)

# function to find the blue dot
def find_dot(frame):
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # find the keypoints and descriptors with sift
    kp = sift.detect(gray, None)
    dot = None
    max_blue = 0
    # iterate through the keypoints and find the blue dot
    for i in range(len(kp)):
        x = int(kp[i].pt[0])
        y = int(kp[i].pt[1])
        b = frame[y, x, 0]
        g = frame[y, x, 1]
        r = frame[y, x, 2]
        if b > max_blue and b > g and b > r:
            max_blue = b
            dot = kp[i]
    return dot

# def dog(img):
#     # Apply 3x3 and 7x7 Gaussian blur
#     low_sigma = cv2.GaussianBlur(img,(3,3),0)
#     high_sigma = cv2.GaussianBlur(img,(5,5),0)
    
#     # Calculate the DoG by subtracting
#     dog = low_sigma - high_sigma
#     return dog

cap = cv2.VideoCapture(0)
# ret, frame = cap.read()

# video capture
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # kp = find_dot(frame)
        # frame = cv2.drawKeypoints(frame, kp, None, (255, 0, 0), 4)
        dot = find_dot(frame)
        if dot is None:
            continue
        # draw a bounding box around the dot
        x = int(dot.pt[0])
        y = int(dot.pt[1])
        frame = cv2.rectangle(frame, (x-10, y-10), (x+10, y+10), (0, 255, 0), 4)
        # frame = cv2.drawKeypoints(frame, [dot], None, (0, 0, 255), 4)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# cv2.imwrite('test.jpg', frame)