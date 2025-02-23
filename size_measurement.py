# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 11:23:08 2025

@author: hui.ma
"""

import cv2
import numpy as np
from tkinter import Tk, filedialog

roi = None
drawing = False
x_start, y_start, x_end, y_end = -1, -1, -1, -1

def select_roi(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, drawing, roi

    if event == cv2.EVENT_LBUTTONDOWN:  
        drawing = True
        x_start, y_start = x, y

    elif event == cv2.EVENT_MOUSEMOVE:  
        if drawing:
            temp_img = image.copy()
            cv2.rectangle(temp_img, (x_start, y_start), (x, y), (0, 0, 255), 2)
            cv2.imshow("Select ROI", temp_img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_end, y_end = x, y
        roi = (x_start, y_start, x_end, y_end)
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.imshow("Select ROI", image)


def detect_rectangles_in_roi():
    if roi is None:
        print("No ROI selected!")
        return []

    x1, y1, x2, y2 = roi
    roi_img = gray[y1:y2, x1:x2]
    edges = cv2.Canny(roi_img, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_rectangles = []  

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            corners = [tuple(point[0] + [x1, y1]) for point in approx]  
            corners = sorted(corners, key=lambda p: (p[1], p[0]))
            top_two = sorted(corners[:2], key=lambda p: p[0])
            bottom_two = sorted(corners[2:], key=lambda p: p[0])  

            top_left, top_right = top_two
            bottom_left, bottom_right = bottom_two
            
            width1 = np.sqrt((top_left[0] - top_right[0])**2 + (top_left[1] - top_right[1])**2)
            width2 = np.sqrt((bottom_left[0] - bottom_right[0])**2 + (bottom_left[1] - bottom_right[1])**2)
            height1 = np.sqrt((top_left[0] - bottom_left[0])**2 + (top_left[1] - bottom_left[1])**2)
            height2 = np.sqrt((top_right[0] - bottom_right[0])**2 + (top_right[1] - bottom_right[1])**2)
            width = (width1+width2)/2
            height = (height1+height2)/2


            detected_rectangles.append({
                "corners": [top_left, top_right, bottom_right, bottom_left],
                "width": width,
                "height": height
            })


            cv2.drawContours(image, [np.array([top_left, top_right, bottom_right, bottom_left])], -1, (255, 0, 0), 2)
    cv2.imshow("Detected Rectangles", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detected_rectangles  


Tk().withdraw()  
file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])

if not file_path:
    print("No file selected!")
    exit()

image = cv2.imread(file_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Select ROI", image)
cv2.setMouseCallback("Select ROI", select_roi)

cv2.waitKey(0)  
cv2.destroyAllWindows()

detected_rects = detect_rectangles_in_roi()

numrect = len(detected_rects)
all_heights = np.zeros([numrect])
for i in np.arange(0,numrect):
    all_heights[i] = detected_rects[i]["height"]
pixel=np.mean(all_heights)

group = int(input("Enter Group number: "))
element = int(input("Enter Element number: "))
lpmm = 2 ** (group + (element - 1)/6)
period_mm = 1 / lpmm  # Period in millimeters

pixelsize = period_mm/(2*pixel)

def draw_line(image_path):
    image = cv2.imread(image_path)
    image_copy = image.copy()
    drawing = False
    start_point = None
    end_point = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, end_point  
        if event == cv2.EVENT_LBUTTONDOWN:
            start_point = (x, y)
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            temp_image = image.copy()
            cv2.line(temp_image, start_point, (x, y), (0, 255, 0), 2)
            cv2.imshow("Draw a Line", temp_image)
        elif event == cv2.EVENT_LBUTTONUP:
            end_point = (x, y)
            drawing = False
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)
            cv2.imshow("Draw a Line", image)
          
    cv2.namedWindow("Draw a Line")
    cv2.setMouseCallback("Draw a Line", mouse_callback)
    cv2.imshow("Draw a Line", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    length = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
    return start_point, end_point, length

root = Tk()
root.withdraw()
img_path = filedialog.askopenfilename(title="Select Image",
                                     filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
root.destroy() 
start, end, length = draw_line(img_path)
actual = pixelsize*length
print(f"Start: {start}, End: {end}, Length: {length:.2f} pixels,Pixel size: {pixelsize:.2f}, Actual length: {actual}")
