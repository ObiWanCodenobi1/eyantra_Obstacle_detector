import cv2
import numpy as np
import cv2.aruco as aruco
import argparse

class Arena:

    def __init__(self, image_path):
        self.width = 1000
        self.height = 1000
        self.image_path = image_path
        self.detected_markers = []
        self.obstacles = 0
        self.total_area = 0

    def identification(self):

        # Read the image
        frame = cv2.imread(self.image_path)
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        ###################################
        # Identify the Aruco ID's in the given image
        # define names of each possible ArUco tag OpenCV supports
        ARUCO_DICT = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
            "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
            "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
            "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
        }

        # loop over the types of ArUco dictionaries
        for (arucoName, arucoDict) in ARUCO_DICT.items():
            # load the ArUCo dictionary, grab the ArUCo parameters, and
            # attempt to detect the markers for the current dictionary
            arucoDict = cv2.aruco.getPredefinedDictionary(arucoDict)
            arucoParams = cv2.aruco.DetectorParameters()
            (self.corners, self.ids, self.rejected) = cv2.aruco.detectMarkers(gray_image, arucoDict, parameters=arucoParams)
            # if at least one ArUco marker was detected display the ArUco
            # name to our terminal
            if len(self.corners) > 0:
                break

        

        self.detected_markers.extend(map(int,self.ids))

        '''for i in self.ids:
            self.detected_markers.append(i)'''
        ###################################
        # Apply Perspective Transform
        self.ref = []

        #hardcoding the corners used as reference, may change it to a funtion later
        self.ref.append(self.corners[0][0][2])
        self.ref.append(self.corners[1][0][3])
        self.ref.append(self.corners[2][0][1])
        self.ref.append(self.corners[3][0][0])
        
        self.ref = np.float32(self.ref)

        w = self.width + 30
        h = self.height + 30

        dst = np.float32([[w,h],[0,h],[w,0],[0,0]])
        matrix = cv2.getPerspectiveTransform(self.ref, dst)
        result = cv2.warpPerspective(frame, matrix, (w, h))
        
        transformed_image = result
        ###################################
        # Use the transformed image to find obstacles and their area

        gray = cv2.cvtColor(transformed_image,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(transformed_image, contours, -1, (0,255,0), 3)
        

        self.obstacles = len(contours)
        self.total_area = 0
        for obstacle in contours:
            self.total_area += cv2.contourArea(obstacle)
        ###################################


    def text_file(self):
        with open("obstacles.txt", "w") as file:
            file.write(f"Aruco ID: {self.detected_markers}\n")
            file.write(f"Obstacles: {self.obstacles}\n")
            file.write(f"Area: {self.total_area}\n")

    def chooseCorners(self): #function to choose the corner of Aruco marker to use as reference
        pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    image_path = args.image

    arena = Arena(image_path)
    arena.identification()
    arena.text_file()
