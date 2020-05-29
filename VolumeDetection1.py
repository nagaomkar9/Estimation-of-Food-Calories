
# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import values as v
import food_predict as item

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# load the image, convert it to grayscale, and blur it slightly
dict={}
def sizeOfObject(image,flag):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    cv2.imshow("Image", edged)
    cv2.waitKey(0)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None
    #print(cnts)
    
    list=[]
    count=0
    order=1
    # loop over the contours individually
    for c in cnts:
            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 150:
                    continue

            # cropping the image
            l=[]
            if pixelsPerMetric is not None and flag==0:
                    x,y,w,h=cv2.boundingRect(c)
                    cropped_image=image[y:y+h,x:x+w]
                    cv2.imwrite('crop.jpeg',cropped_image)
                    name=item.Predict()
                    l.append(name)
                    dict[order]=l
                    #order=order+1
                    #print(name)
                    

            # compute the rotated bounding box of the contour
            orig = image.copy()
            box = cv2.minAreaRect(c)
            #print(box)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            #print(box)
            

            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

            # loop over the original points and draw them
            for (x, y) in box:
                    cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)

            # compute the midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-righ and bottom-right
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # draw the midpoints on the image
            cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

            # draw lines between the midpoints
            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                    (255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                    (255, 0, 255), 2)

            # compute the Euclidean distance between the midpoints
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            # if the pixels per metric has not been initialized, then
            # compute it as the ratio of pixels to supplied metric
            # (in this case, inches)
            width=0.9
            if pixelsPerMetric is None:
                    pixelsPerMetric = dB / width

            # compute the size of the object
            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric
            if count != 0 and flag==0:
                    l.append(dimA*dimB)
                    dict[order]=l
                    order=order+1
                            
            elif count!=0 and flag==1:
                    list.append(dimA)
                    #list.append(dimB)
            
            # draw the object sizes on the image
            cv2.putText(orig, "{:.1f}in".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 0, 0), 2)
            cv2.putText(orig, "{:.1f}in".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 0, 0), 2)

            # show the output image
            cv2.imshow("Image", orig)
            cv2.waitKey(0)
            count=count+1
    if flag==0:
            return dict
    elif flag==1:
            return list
       


image = cv2.imread('example6.jpeg')
flag=0
TopView={}
TopView=sizeOfObject(image,flag)
#print(TopView)
flag=1
SideView=[]
image1 = cv2.imread('example5.jpeg')
SideView=sizeOfObject(image1,flag)
#print(SideView)
#calculating calories o food item

for key in TopView:
    val=TopView[key]
    name=v.Food(val[0])
    density=v.Density(val[0])
    caloriespergram=v.Calories(val[0])
    volume=val[1]*SideView[key-1]*2.54*2.54*2.54
    mass=density*volume
    calories=caloriespergram*mass
    print("Food item is ",name,", weights up to ",mass," grams and number of calories are ",calories)
