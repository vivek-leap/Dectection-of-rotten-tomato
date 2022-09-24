import os
import matplotlib.pyplot as plt
from rembg.bg import remove
import numpy as np
import io
from PIL import Image
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = os.path.join(folder,filename)
        if img is not None:
            images.append(img)
    return images
result12=[]
for i in load_images_from_folder(r"C:\Users\Acer\Documents\R"):
    org=cv2.imread(r"C:\Users\Admin\Downloads\final.jpg")
    org = cv2.resize(org, (0,0), fx=0.1, fy=0.1)
    cv2.imshow("original",org)
    cv2.waitKey(0) 
    input_path = r"C:\Users\Admin\Downloads\final.jpg"

# ******************************************************** remove background and make it black***************************************
f = np.fromfile(input_path)
result = remove(f)
img = Image.open(io.BytesIO(result)).convert("RGBA")
# img.save(output_path)
imcv = np.asarray(img)[:,:,::-1].copy()
# Or
imcv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
imcv = cv2.resize(imcv, (0,0), fx=0.1, fy=0.1)
cv2.imshow("after removal",imcv)
cv2.waitKey(0) 

# *************************************************glare*********************************************************
gray = cv2.cvtColor(imcv, cv2.COLOR_BGR2GRAY)
grayimg = gray

GLARE_MIN = np.array([0, 0, 100],np.uint8)
GLARE_MAX = np.array([0, 0, 150],np.uint8)

hsv = cv2.cvtColor(imcv, cv2.COLOR_BGR2HSV)

#HSV
frame_threshed = cv2.inRange(hsv, GLARE_MIN, GLARE_MAX)


#INPAINT
mask1 = cv2.threshold(grayimg , 160, 255, cv2.THRESH_BINARY)[1]
imcv1 = cv2.inpaint(imcv, mask1, 0.3, cv2.INPAINT_TELEA) 

cv2.imshow("glare removal",imcv1)
cv2.waitKey(0) 
imcv=imcv1
# ******************************************************* crop*****************************************************************
original = imcv.copy()
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
canny = cv2.Canny(blurred,10,200)
kernel = np.ones((5,5),np.uint8)
dilate = cv2.dilate(canny, kernel, iterations=1)
cv2.imshow("dilated",dilate)
cv2.waitKey(0) 

# Find contours
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# Iterate thorugh contours and filter for ROI

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)
    ROI = original[y:y+h, x:x+w]
    ROI1 = dilate[y:y+h, x:x+w]
cv2.imshow("cropped",ROI1)
cv2.waitKey(0) 
number_of_white_pix = np.sum(ROI1 > 0)
number_of_black_pix = np.sum(ROI1 == 0)


print((number_of_white_pix*100)/(number_of_white_pix+number_of_black_pix))
result12.append((number_of_white_pix*100)/(number_of_white_pix+number_of_black_pix))

# for i in result:
#     print(i)
plt.plot(range(1,len(result12)+1),(result12))
plt.plot(range(1,len(result12)+1),sorted(result12))
plt.xlabel("No. of images with increasing time")
plt.ylabel("% of white pixels")
plt.title("% of white pixels for canny edge")
plt.show()