import cv2                                          #importing libraries of python OpenCV
detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")     #Using our classifier

imp_img = cv2.VideoCapture("elon.jpg")                        #we can use 0 or 1 instead of file name if we 
                                                                #are using source of video capture as our webcam

res, img = imp_img.read()                                      #it gives us two returns 1. if this code has read the image(true or false)
                                                                    #2. dimensions {res will store true and false and img will store coordinates}

"""Haarcascade classifier is trained for grey scale images. So we have to change our image into Greyscale"""
grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

"""Detect faces of different sizes in the input image"""
faces = detect.detectMultiScale(grey,1.3,5)                    #(grey_image,Scale factor,minNeighbor)
                                                               #we will get 4 coordinates in faces variable: x,y,width,height
                                                               #xy will be origin and width and height will be from origin
                                                               
"""Drawing rectangle over detected face"""                                                               
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)                                          #(image,pt1,pt2,color,thickness)

cv2.imshow("Elon image",img)                  #showing the image(any image tile,image variable)
                                              #3 things alwyas needs to be done while showing image: 1.Wait key 2. Release of window 3. Destroying of window
cv2.waitKey(0)                                  #milliseconds needs to be passed(here 0 milliseconds is passed)
imp_img.release()
cv2.destroyAllWindows()
