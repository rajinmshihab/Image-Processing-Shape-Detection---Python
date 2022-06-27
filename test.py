# import the necessary packages
import argparse
import imutils
import cv2
from imutils import contours
import numpy as np

# detect a shape of object
# para: contour of object
def detectShape(c):
    # initialize the shape name and approximate the contour
    approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
    # if the shape has 3 vertices, it is a triangle
    if len(approx) == 3:
        shape = "triangle"
    # if the shape has 4 vertices, it is a rectangle
    elif len(approx) == 4:
        shape = "rectangle"
    # otherwise,assume it is a circle
    elif len(approx) >15:
        shape = "circle"
    #otherwise, we assume the shape is undefined
    else:
        shape = "undefined"
    # return the name of the shape
    #print(len(approx))
    return shape

def driver(image):
    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    # image = cv2.pyrMeanShiftFiltering(image, 21, 51)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (3, 3), (image.shape[1]-3, image.shape[0]-3), 255, -1)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # assume that color of background is white-like. and remove the background
    (T, thresh) = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('threshold', thresh)
    
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((2,2), np.uint8)
    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image
    thresh = cv2.erode(thresh, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    masked = cv2.bitwise_and(thresh, thresh, mask=mask)
 
    edged = cv2.Canny(masked, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # find contours in the edge map
    contour = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)
    # sort the contours from left-to-right
    (contour, _) = contours.sort_contours(contour)
    
    idToshape = [] # id - shape map array
    id = 0
    triangle_cnt = 0
    maxArea = 0
    minArea = 99999
    maxObject = None
    minObject = None
    # loop over the contours
    for c in contour:
        # if the contour is not sufficiently large, ignore it
        cntArea = cv2.contourArea(c)
        # find a maximum object
        if cntArea > maxArea:
            maxArea = cntArea
            maxObject = id
        # find a minimum object
        if cntArea < minArea:
            minArea = cntArea
            minObject = id
        # determine a shape of contour
        shape = detectShape(c)
        # add shape along with id into map array
        if shape == 'triangle':
            triangle_cnt+=1
        idToshape.append(shape)

        # draw the contours and the name of the shape on the image
        # cv2.drawContours(image, [c.astype("int")], -1, (0, 255, 0), 1)
      
        # compute the center of the contour
        m = cv2.moments(c)
        cX = int((m["m10"] / m["m00"]))
        cY = int((m["m01"] / m["m00"]))
        # put labels of id into every object's center
        cv2.putText(image, str(id), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0,255,0), 2)
        id+=1

    # show the output image
    cv2.imshow("Image", image)
    # print details
    print('1. The number of objects in the scene is     : ', id)
    print('2. The smallest object in the scene is id    : ', minObject)
    print('3. The largest object in the scene is id     : ', maxObject)
    print('5. The shapes of object in the scene         : ')
    id = 0   
    for  shape in idToshape:
        print('\t{0} - {1}'.format(id, shape))
        id+=1
   
 
 
if __name__ == '__main__':  
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to the input image")
    args = vars(ap.parse_args())
    # load the image and resize it to a smaller factor so that
    # the shapes can be approximated better
    image = cv2.imread(args["image"])
    driver(image)
    cv2.waitKey(0)