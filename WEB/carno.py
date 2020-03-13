
from threading import Lock, Thread
import os.path
import glob
from pspconfig import PspConfig 
import Cstool
from  reco import Carno
import cv2 as cv
import tensorflow as tf

global lock
lock = Lock()
carno = Carno('c')
dnn_process_count = 0
def getDnnProcessCount(gb):
    global dnn_process_count
    lock.acquire()
    dnn_process_count += gb
    rcnt  = dnn_process_count
    lock.release()
    return rcnt

def find_process(file_name):
    try:
        car_no = None
        rcnt = getDnnProcessCount(1)
        print('find_process', rcnt)
        if  rcnt == 1:
            car_no, xml = carno.findProcess(file_name)
            if car_no == None :
                car_no = 'Not found'
            return  car_no
        return "동시에 2개을 분석하지 못해요."
    finally :
        getDnnProcessCount(-1)
    return "처리중 오류가 발생했습니다."
# -*- coding: utf-8 -*-
from io import BytesIO
import sys, os
from pspconfig import PspConfig
def call_psp_(http_handler, args):
    _psp_out_ = BytesIO()
    
    plate_no_text='no image'
    print('args', args)
    if "image" in args:
        display_image=args["image"]
        plate_no_text=find_process(PspConfig.img_path+"/"+display_image)
        print('carno.psp>>>>>', plate_no_text)
    
    _psp_out_.write(plate_no_text.encode())
    return _psp_out_
