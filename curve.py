# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:04:21 2015

@author: drizardnicolas
"""
#Code to build some curves and to interpret some results

size=[]
PP_time=[]
PP_time_5=[]
PP_time_square=[]
PM_time=[]
PM_time_5=[]
PM_time_square=[]

size.append(i1*j1)

PP_time.append(pre_processing_time)
PM_time.append(pre_processing_time)

PP_time_5.append(pre_processing_time)
PM_time_5.append(pre_processing_time)

PP_time_square.append(3.8977)
PM_time_square.append(39.756)

plt.plot(size, PP_time, 'ro')
plt.plot(size, PP_time_5, 'bs')
plt.plot(size, PP_time_square, 'g^')
plt.show()

from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(size,PP_time,'ro',label="Model 1")
line2, = plt.plot(size,PP_time_5,'bs',label="Model 2")
line3, = plt.plot(size,PP_time_square,'g^', label="Model 2")

plt.title('Pre-processing time')
plt.xlabel('Image size')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=1),line2: HandlerLine2D(numpoints=1),line3: HandlerLine2D(numpoints=1)},loc=2)

plt.show()

line1, = plt.plot(size,PM_time,'ro',label="Model 1")
line2, = plt.plot(size,PM_time_5,'bs',label="Model 2")
line3, = plt.plot(size,PM_time_square,'g^', label="Model 2")

plt.title('Patch-matching time')
plt.xlabel('Image size')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=1),line2: HandlerLine2D(numpoints=1),line3: HandlerLine2D(numpoints=1)},loc=2)

plt.show()

square_result=[5, 4, 1, 4, 2, 3]
param_result=[4.5, 3, 1, 3, 2, 1.5]
color_result=[4, 2, 0.5, 2, 1, 0]
eye_true=[12, 8, 12, 12, 8, 8]
images=[1,2,3,4,5,6]


line1, = plt.plot(images,color_result,'ro',label="Model 1")
line2, = plt.plot(images,param_result,'bs',label="Model 2")
line3, = plt.plot(images,square_result,'g^', label="Model 2")
line4, = plt.plot(images,eye_true,'r--', label="Real image")

plt.axis([0.5, 6.5, 0, 14])
plt.title('Prediction accuracy of the models')
plt.xlabel('Image number')
plt.ylabel('Number of eyes predicted against the reality')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=1),line2: HandlerLine2D(numpoints=1),line3: HandlerLine2D(numpoints=1)},loc=5)

plt.show()

sum(color_result)/35.0