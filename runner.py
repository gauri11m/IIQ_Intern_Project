import numpy as np
import cv2
import datetime
import time
from Image_segmentation import *

from tracker import *

import mrcnn.model as modellib
from mrcnn.config import Config
from visualise import random_colors, get_mask_contours, draw_mask
import smtplib

import cv2

seg = image()

cap = cv2.VideoCapture(r"C:\Users\Hp\Downloads\Telegram Desktop\Test_Trim01_Trim (2).mp4")

tracker=EuclideanDistTracker()

# cap = cv2.VideoCapture(0)
wht = 320
thresh = 0.5
nmsthresh = 0.2


modelConfig =  'yolov3_testing.cfg'
modelWeights = 'yolov3_training_last.weights'

# modelConfig =  'yolov4-tiny-custom.cfg'
# modelWeights = 'yolov4-tiny-custom_last.weights'

net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

id_list = []

def find_objects(outputs,img):

    L = []
    ht,wt,ct = img.shape
    box = []
    class_ids = []
    confs = []
    area = 0
    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > thresh:
                w,h = int(det[2]*wt), int(det[3]*ht)
                x,y = int(det[0]*wt - w/2), int(det[1]*ht - h/2)
                if w/h>3:
                    box.append([x,y,w,h])
                    L.append([x,y,w,h])
                    boxids = tracker.update(L)
                    class_ids.append(class_id)
                    confs.append(float(confidence))
                    area = area + (w-30)*(h)
                    x, y, w, h, id = boxids[0]
                    if id not in id_list:
                        id_list.append(id)
                        seg.seg(img, id_list, id)

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

    blob = cv2.dnn.blobFromImage(img,1/255,(wht,wht),(0,0,0),1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = []
    # print(layerNames)
    for i in net.getUnconnectedOutLayers():
        outputNames.append(layerNames[i - 1])
    # outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)

    outputs = net.forward(outputNames)
    # print(outputs[0].shape)

    find_objects(outputs, img)

    cv2.imshow('Image',img)
    cv2.waitKey(1)
