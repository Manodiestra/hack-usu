import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
import cv2

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '_data/pose_deploy.prototxt'
PRETRAINED = '_data/pose_iter_102000.caffemodel'

nPoints = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]

threshold = 0.2

# load the model
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
print("successfully loaded classifier")

cap = cv2.VideoCapture(0)
hasFrame, frame = cap.read()

frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

aspect_ratio = frameWidth/frameHeight

inHeight = 368
inWidth = int(((aspect_ratio*inHeight)*8)//8)

while True:
    hasFrame, frame = cap.read()
    frameCopy = np.copy(frame)
    if not hasFrame:
        cv2.waitKey()
        break
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).aastype(np.float32) / 255.0
    net.blobs['data'].data[...] = image
    net.forward()

    output = net.blobs['prob'].data

    # Empty list to store the detected keypoints
    points = []

    canvas = np.zeros((frameHeight, frameWidth), dtype=np.uint8)

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))
        grayProbMap = (np.interp(probMap, [0.0, 1.0], [0, 255])).astype(np.uint8)
        canvas = cv2.bitwise_or(canvas, grayProbMap)

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold :
            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 6, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1])))
        else :
            points.append(None)

    cv2.imshow("Probability Map", canvas)
    key = cv2.waitKey(1)
    if key == 27:
        break

