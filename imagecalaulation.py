# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:09:50 2022

@author: hui.ma
"""


import numpy as np
import matplotlib.pyplot as plt
import csv



RVITM=np.load('TM210225_n17_o16_1ms_1.npy')
output=np.loadtxt('output_o16_n17_210225_1ms_1.csv', delimiter=',')
inp=np.loadtxt('input_o16_n17_210225_1ms_1.csv', delimiter=',')
outputc = output[:,200:1280]
outputmodified=np.zeros((270,270))
for i in range (270):
    for j in range (270):
        outputmodified[i,j]=outputc[4*i:4*i+3,4*j:4*j+1].mean()
        
outline=outputmodified.reshape(270*270,1)     
RVITMI=np.linalg.pinv(RVITM) 
Iimg=np.dot(RVITMI,outline)
img=Iimg.reshape(16,16)


def find_best_threshold(float_matrix, binary_matrix):
    best_threshold = 0
    best_score = float('inf')
    
    for t in np.linspace(float_matrix.min(), float_matrix.max(), 1000):  # Try 100 thresholds
        binarized = (float_matrix >= t).astype(int)
        score = np.sum(binarized != binary_matrix)  # Count mismatches
        
        if score < best_score:
            best_score = score
            best_threshold = t

    return best_threshold

threshold = find_best_threshold(img, inp)
binary_estimate = (img >= threshold).astype(int)

print("Optimal threshold:", threshold)
print("Binary matrix after thresholding:\n", binary_estimate)



# threshold=0.
# n=8
# binarr = np.where(img>threshold, 1, 0)
diff=inp-binary_estimate
error=np.count_nonzero(diff)    
print(error,'and',threshold)
plt.figure(1)
plt.imshow(binary_estimate)
plt.figure(2)
plt.imshow(inp)
plt.figure(3)
plt.imshow(diff)
plt.figure(4)
plt.imshow(img)
plt.figure(5)
plt.imshow(output,cmap='gray',vmin=0,vmax=1023)

# with open("inputring1.csv", "w", newline="") as f1:
#     writer = csv.writer(f1)
#     writer.writerows(binarr)

# plt.figure(1)
# plt.imshow(output,cmap='gray',vmin=0,vmax=1023)

# plt.figure(2)
# plt.imshow(outputc,cmap='gray',vmin=0,vmax=1023)

# plt.figure(3)
# plt.imshow(outputmodified,cmap='gray',vmin=0,vmax=1023)