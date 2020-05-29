# USAGE
# python object_size.py --image images/example_01.png --width 0.955
# python object_size.py --image images/example_02.png --width 0.955
# python object_size.py --image images/example_03.png --width 3.5

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import values
import food_predict as item

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# load the image, convert it to grayscale, and blur it slightly
dict={}
OrderOfObjects={}
def sizeOfObject(image,flag):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

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
            if cv2.contourArea(c) < 300:
                    continue

            # cropping the image
            if pixelsPerMetric is not None and flag==0:
                    x,y,w,h=cv2.boundingRect(c)
                    cropped_image=image[y:y+h,x:x+w]
                    cv2.imwrite('crop.jpeg',cropped_image)
                    name=item.Predict()
                    OrderOfObjects[order]=name
                    order=order+1
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
                    if name not in dict:
                            l=[]
                            l.append(dimA*dimB)
                            dict[name]=l
                    else:
                            l=dict[name]
                            l.append(dimA*dimB)
                            dict[name]=l
                            
            elif count!=0 and flag==1:
                    list.append(dimA)
                    #list.append(dimB)
            
            # draw the object sizes on the image
            cv2.putText(orig, "{:.1f}in".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
            cv2.putText(orig, "{:.1f}in".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

            # show the output image
            cv2.imshow("Image", orig)
            cv2.waitKey(0)
            count=count+1
    if flag==0:
            return dict
    elif flag==1:
            return list
       


image = cv2.imread('example_12.jpg')
flag=0
TopView={}
TopView=sizeOfObject(image,flag)
#print(TopView)
flag=1
#SideView=[]
image1 = cv2.imread('example_13.jpg')
SideView=sizeOfObject(image1,flag)
#print(SideView)
#print(OrderOfObjects)
#volume of the object
result={}
c=0
for order in OrderOfObjects:
        item=OrderOfObjects[order]
        name=values.Food(item)
        density=values.Density(item)
        calories=values.Calories(item)
        if name not in result:
                l=TopView[item]
                result[name]=(l.pop(0)*SideView[order-1]*1.5*1.5*1.5*density*calories)/12
                print(result[name])
        else:
                l=TopView[item]
                result[name]=result[name]+((l.pop(0)*SideView[order-1]*1.5*1.5*1.5*density*calories)/12)
                print(result[name])
        #result[name]=(TopView[FoodItem]*SideView[c]*1.5*1.5*1.5*density)/12
for name in result:
        print("Food item is ",name," and number of calories are ",result[name])

