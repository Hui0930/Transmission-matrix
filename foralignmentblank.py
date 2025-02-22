# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:34:29 2022

@author: hui.ma
"""
# create a square for the system alignment
import numpy as np
import slmpy
import matplotlib.pyplot as plt
import time

D=np.zeros((1080,1920),dtype='uint8')
D[340:740,760:1160]=255
D=np.uint8(D)
slm = slmpy.SLMdisplay()
slm.updateArray(D)
time.sleep(10000)
slm.close()