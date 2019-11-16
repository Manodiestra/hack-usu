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
    # point_List = []
    # clickPT = (0,0)
    # def on_click(event, x, y, flags, param):
    #     global point_List
    #     if event == cv2.EVENT_LBUTTONUP:
    #         clickPT = (x, y)

    homography = None 
    # matrix of camera parameters (made up but works quite well for me) 
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

    # # load the reference surface that will be searched in the video stream
    dir_name = os.getcwd()

    # # Load 3D model from OBJ file
    obj = OBJ(os.path.join(dir_name, 'obj_files/Drachen_1.0_obj.obj'), swapyz=True)  
    # init video capture
    cap = cv2.VideoCapture(0)

    cv2.namedWindow('frame')
    # cv2.setMouseCallback("frame",on_click)

    countY =0
    while True:
        if countY >100:
            countY =0
        countY +=1
        # read the current frame
        ret, frame = cap.read()

        frame.shape[0] #rows y 
        boxSiz = 50
        src_pts = np.float32([
            [[frame.shape[1]/2 -boxSiz, frame.shape[0]/2- boxSiz]],   # Top LEft point
            [[frame.shape[1]/2 +boxSiz, frame.shape[0]/2 -boxSiz]],  # Top right point
            [[frame.shape[1]/2- boxSiz, frame.shape[0]/2 + boxSiz]],  # BOttem left
            [[frame.shape[1]/2 +boxSiz, frame.shape[0]/2+ boxSiz]],  # BOttem Right
            ])

        # print(src_pts)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters =  aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        # print(corners)
        SPEED = 25
        dst_pts = None
        if len(corners) !=0:
            dst_pts = np.float32([
                [corners[0][0][0]],
                [corners[0][0][1]],
                [corners[0][0][3]],
                [corners[0][0][2]],
                ])

        # print(dst_pts)
        if dst_pts is not None:
            for i in range(len(dst_pts)):
                cv2.line(frame,tuple(src_pts[i][0]),tuple(dst_pts[i][0]),(0,255,0),2)

                cv2.drawMarker(frame,tuple(dst_pts[i][0]),[0,255,255],cv2.MARKER_CROSS,30,2)

            homography = cv2.getPerspectiveTransform(
                src_pts,
                dst_pts,
                 ) 
            # if a valid homography matrix was found render cube on model plane
            if homography is not None:
                try:
                    # obtain 3D projection matrix from homography matrix and camera parameters
                    projection = projection_matrix(camera_parameters, homography)  
                    # project cube or model
                    frame = render(frame, obj, projection, frame, False)
                    #frame = render(frame, model, projection)
                except Exception as e:
                    print("Error {}".format(e))
        # show result
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        else:
            pass
            # print ()"Not enough matches found - %d/%d" % (len(matches), MIN_MATCHES)

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
