# Useful links
# http://www.pygame.org/wiki/OBJFileLoader
# https://rdmilligan.wordpress.com/2015/10/15/augmented-reality-using-opencv-opengl-and-blender/
# https://clara.io/library

# http://www.philipzucker.com/aruco-in-opencv/


import argparse

import cv2
import cv2.aruco as aruco

import numpy as np
import math
import os
from objloader_simple import *

# Minimum number of matches that have to be found
# to consider the recognition valid
MIN_MATCHES = 10  


def main():
    """
    This functions loads the target surface image,
    """
    #1,2,17,19

    protoFile = "./_data/pose_deploy.prototxt"
    weightsFile = "./_data/pose_iter_102000.caffemodel"
    nPoints = 22
    POSE_PAIRS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]

    threshold = 0.2

    homography = None 
    # matrix of camera parameters (made up but works quite well for me) 
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

    # # load the reference surface that will be searched in the video stream
    dir_name = os.getcwd()

    # # Load 3D model from OBJ file
    obj = OBJ(os.path.join(dir_name, 'obj_files/Drachen_1.0_obj.obj'), swapyz=True)  
    # init video capture
    cap = cv2.VideoCapture(0)
    hasFrame, frame = cap.read()

    if not hasFrame:
        return

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    aspect_ratio = frameWidth/frameHeight

    inHeight = 368
    inWidth = int(((aspect_ratio*inHeight)*8)//8)

    cv2.namedWindow('frame')
    # cv2.setMouseCallback("frame",on_click)
    boxSiz = 50

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    while True:
        # read the current frame
        hasFrame, frame = cap.read()
        frameCopy = np.copy(frame)
        if not hasFrame:
            cv2.waitKey()
            break

        frame.shape[0] #rows y 
        src_pts = np.array([
            [[frame.shape[1]/2 -boxSiz, frame.shape[0]/2- boxSiz]],   # Top LEft point
            [[frame.shape[1]/2 +boxSiz, frame.shape[0]/2 -boxSiz]],  # Top right point
            [[frame.shape[1]/2- boxSiz, frame.shape[0]/2 + boxSiz]],  # BOttem left
            [[frame.shape[1]/2 +boxSiz, frame.shape[0]/2+ boxSiz]],  # BOttem Right
            ], dtype='float32')

        # print(src_pts)

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        # parameters =  aruco.DetectorParameters_create()
        # corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        # # print(corners)
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)

        net.setInput(inpBlob)

        output = net.forward()

        # Empty list to store the detected keypoints
        points = []
        corners = []

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
                cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2, lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(point[0]), int(point[1])))
                if i in (1,2,17,5):
                    corners.append((int(point[0]), int(point[1])))
            else :
                points.append(None)
        cv2.imshow("Probability Map", canvas)

        dst_pts = None
        if len(corners) == 4:
            dst_pts = np.float32([
                [corners[0]],
                [corners[1]],
                [corners[3]],
                [corners[2]],
                ])

        # print(dst_pts)
        # if dst_pts is not None:
        if dst_pts is not None:
            for i in range(len(dst_pts)):
                cv2.line(frameCopy,tuple(src_pts[i][0]),tuple(dst_pts[i][0]),(0,255,0),2)
                cv2.drawMarker(frameCopy,tuple(dst_pts[i][0]),[0,255,255],cv2.MARKER_CROSS,30,2)

            homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
        else:
            homography = None
        # if a valid homography matrix was found render cube on model plane
        if homography is not None:
            try:
                # obtain 3D projection matrix from homography matrix and camera parameters
                projection = projection_matrix(camera_parameters, homography)  
                # project cube or model
                render(frameCopy, obj, projection, frame, False)
                #frame = render(frame, model, projection)
            except Exception as e:
                print("Error {}".format(e))
        
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(frameCopy, points[partA], points[partB], (0, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.circle(frameCopy, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frameCopy, points[partB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        # show result
        cv2.imshow('frame', frameCopy)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('-'):
            boxSiz += 20
        elif key == ord("+"):
            boxSiz -= 20

    cap.release()
    cv2.destroyAllWindows()
    return 0

def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape[:2]

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))


if __name__ == '__main__':
    main()
