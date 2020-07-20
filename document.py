import cv2
import numpy as np
import utils
 
 
########################################################################
webCamFeed = False
pathImage = "2.jpg"
cap = cv2.VideoCapture(0)
cap.set(10,160)
heightImg = 480
widthImg  = 640
########################################################################
 

count=0
 
while True:
 
    if webCamFeed: # FOR WEBCAM
    	success, img = cap.read()
    else:
    	img = cv2.imread(pathImage) # IF NOT USING WEBCAM
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
    imgCanny=cv2.Canny(imgBlur,100,150) # ADD Canny to detect Edges
    kernel=np.ones((5,5))	
    imgDilate=cv2.dilate(imgCanny,kernel,iterations=2) # ADDING DILATION
    imgErode=cv2.erode(imgDilate,kernel,iterations=1) # ADDING ERODE

    contours,hierarchy=cv2.findContours(imgErode,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) # FINDING CONTOURS
    contours=sorted(contours,key=cv2.contourArea,reverse=True) # SORTING CONTOURS TO GET THE BIGGEST CONTOUR

    for c in contours:
    	p=cv2.arcLength(c,True) # FINDING ARC LENGTH
    	approx=cv2.approxPolyDP(c,0.02*p,True)  
    	
    	if len(approx)==4:	# LENGTH IS 4 FOR RECTANGLE
    		target=approx
    		break

    		
    approx=utils.reorder(target) # REORDERING THE POINTS 
    #cv2.drawContours(img,target,-1,(0,255,0),20) # TO DISPLAY CONTOURS
    print(approx) # DISPLAYS THE POINTS 
    pts1=np.float32(approx)
    pts2=np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])

    matrix=cv2.getPerspectiveTransform(pts1,pts2)
    imgWarpColored=cv2.warpPerspective(img,matrix,(widthImg,heightImg)) # WARPS THE IMAGE

    cv2.imshow('Original',img)
    cv2.imshow('Erode',imgErode)
    cv2.imshow('WarpCol',imgWarpColored)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
    	break
 
   
 
	
		

