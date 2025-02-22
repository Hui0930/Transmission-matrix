# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 09:28:16 2022

@author: hui.ma
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 15:16:01 2022

@author: hui.ma
"""

try:
    # if on Windows, use the provided setup script to add the DLLs folder to the PATH
    from windows_setup import configure_path
    configure_path()
except ImportError:
    configure_path = None

import numpy as np
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, OPERATION_MODE
import matplotlib.pyplot as plt
from PIL import Image
from scipy.linalg import hadamard
import math
import time
import slmpy
import csv
from datetime import datetime

now = datetime.now()
starttime=now.strftime("%H:%M:%S")

order=256
n=17
A=hadamard(order).astype('int8')
A1=(A+1)//2
A2=(-A+1)//2
B1=A1.reshape((order,int(math.sqrt(order)),int(math.sqrt(order))))
B2=A2.reshape((order,int(math.sqrt(order)),int(math.sqrt(order))))
B=np.concatenate((B1, B2))
C=np.zeros((B.shape[0],n*B.shape[1],n*B.shape[2]), dtype='uint8')
for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        for k in range(B.shape[2]):
            C[i,n*k:n*k+n,n*j:n*j+n]=B[i,k,j]

D=np.zeros((C.shape[0],1080,1920), dtype='uint8')
D[:,540-C.shape[1]//2:540+C.shape[1]//2,960-C.shape[2]//2:960+C.shape[2]//2]=C
D=255*D
D=np.uint8(D)

slm = slmpy.SLMdisplay(isImageLock = True)
output=[]
with TLCameraSDK() as sdk:
    available_cameras = sdk.discover_available_cameras()
    if len(available_cameras) < 1:
        print("no cameras detected")
        
    with sdk.open_camera(available_cameras[0]) as camera:
        camera.exposure_time_us = 2000  # set exposure to 11 ms
        camera.frames_per_trigger_zero_for_unlimited = 0  # start camera in continuous mode
        camera.image_poll_timeout_ms = 2000  # 1 second polling timeout
        old_roi = camera.roi  # store the current roi
        
        camera.arm(2)
        camera.issue_software_trigger()
        #for i in range(10):
        for i in range(D.shape[0]):
            testIMG=D[i]
            slm.updateArray(testIMG)
            time.sleep(1)
            frame = camera.get_pending_frame_or_null()
            if frame is not None:
                frame.image_buffer
                image_buffer_copy = np.copy(frame.image_buffer)
                output.append(image_buffer_copy)
                print(i)

            else:
                print("timeout reached during polling, program exiting...")
        
        randommatrix=np.random.randint(2,size=order)
        inputmatrix=randommatrix.reshape((int(math.sqrt(order)),int(math.sqrt(order))))
        magainput=np.zeros((int(n*math.sqrt(order)),int(n*math.sqrt(order))))
        for i in range (int(math.sqrt(order))):
            for j in range (int(math.sqrt(order))):
                magainput[n*i:n*i+n,n*j:n*j+n]=inputmatrix[i,j]
        T=np.zeros((1080,1920), dtype='uint8')
        T[540-magainput.shape[0]//2:540+magainput.shape[0]//2,960-magainput.shape[1]//2:960+magainput.shape[1]//2]=magainput
        T=255*T
        T=np.uint8(T)
        slm.updateArray(T)
        time.sleep(1.5)
        frame = camera.get_pending_frame_or_null()
        if frame is not None:
            frame.image_buffer
            image_buffer_copy = np.copy(frame.image_buffer)
        
        with open("input_o16_n17_170225_2ms_1.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(inputmatrix)

        print('save the test input')
        
        with open("output_o16_n17_170225_2ms_1.csv", "w", newline="") as f1:
            writer = csv.writer(f1)
            writer.writerows(image_buffer_copy)

        print('save the test output')        
        
        camera.disarm()
        camera.roi = old_roi  # reset the roi back to the original roi
        
slm.close()
outputarray=np.asarray(output)
outputarraycroped=outputarray[:,:,200:1280]
compactarray=np.zeros((2*order,270,270))
for k in range (2*order):
    for i in range (270):
        for j in range (270):
            compactarray[k,i,j]=outputarraycroped[k,4*i:4*i+3,4*j:4*j+3].mean()

reoutput=compactarray.reshape(2*order,270*270,1)
finaloutput=np.zeros((270*270,2*order))
for i in range (2*order):
    finaloutput[:,i]=reoutput[i,:,0]

w=finaloutput[:,0]
modifiedoutput=2*finaloutput-w[:,None]
modifiedinput=np.transpose(np.concatenate((A,-A),1))
RVITM=np.dot(modifiedoutput,modifiedinput)
print('finish calculation')
np.save('TM210225_n17_o16_2ms_1.npy',RVITM)
print('save the TM')
np.save('output210225_n17_o16_2ms_1.npy',outputarray)
print('save the output')
end = datetime.now()
stoptime=end.strftime("%H:%M:%S")

print(starttime)
print(stoptime)
            